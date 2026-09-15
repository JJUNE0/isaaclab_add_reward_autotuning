"""Wolf v2 standard PPO baseline with symmetric oracle observations."""
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from lab.wolf.assets.wolf_v2 import wolf_v2_cfg
from ..mdp import gait, oracle
from ..terrain.stairs_cfg import STAIRS_CFG

JOINT_CFG = SceneEntityCfg("robot", joint_names=oracle.JOINTS, preserve_order=True)

@configclass
class WolfSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = wolf_v2_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="generator", terrain_generator=STAIRS_CFG,
        max_init_terrain_level=1, collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.4, 0.45)),
        debug_vis=False,
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.4, 0.0, 20.0)), ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        mesh_prim_paths=["/World/ground"], debug_vis=False,
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))

@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot", resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0, rel_heading_envs=1.0, heading_command=True,
        heading_control_stiffness=0.5, debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.15, 0.45), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.3, 0.3), heading=(0.0, 0.0)),
    )


@configclass
class GaitCommandsCfg(CommandsCfg):
    """Velocity command plus a fixed trot command for gait V0."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=None,
        ),
    )
    gait = gait.FixedGaitCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        theta1=0.5,
        theta2=0.0,
        theta3=0.0,
        frequency=2.0,
        duty_factor=0.5,
    )


@configclass
class GaitFrequencyRandomizedCommandsCfg(GaitCommandsCfg):
    """Trot commands with one frequency sampled at the start of each episode."""

    gait = gait.FixedGaitCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        theta1=0.5,
        theta2=0.0,
        theta3=0.0,
        frequency=2.0,
        frequency_range=(1.0, 3.0),
        duty_factor=0.5,
    )

@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=oracle.JOINTS, preserve_order=True,
        use_default_offset=True, scale={"HAA_.*": 0.4, "HFE_.*": 1.0, "KFE_.*": 0.8, "AFE_.*": 0.8},
        clip={"HAA_front_.*": (-0.43633, 0.5236), "HAA_back_.*": (-0.7854, 0.7854),
              "HFE_front_.*": (-3.14, 1.2217), "HFE_back_.*": (-1.2217, 1.2217),
              "KFE_front_.*": (-1.1, 0.87267), "KFE_back_.*": (-0.96, 1.22), "AFE_.*": (-0.5236, 1.4835)},
    )

@configclass
class FrameCfg(ObservationGroupCfg):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": JOINT_CFG})
    joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": JOINT_CFG}, scale=0.1)
    angular_velocity = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
    gravity = ObsTerm(func=mdp.projected_gravity)
    actions = ObsTerm(func=mdp.last_action)
    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True

@configclass
class OracleCfg(ObservationGroupCfg):
    commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    linear_velocity = ObsTerm(func=mdp.base_lin_vel)
    height_scan = ObsTerm(func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.0}, clip=(-3.0, 3.0))
    feet = ObsTerm(func=oracle.foot_state)
    contacts = ObsTerm(func=oracle.contact_forces, scale=0.01)
    torque = ObsTerm(func=mdp.joint_effort, params={"asset_cfg": JOINT_CFG}, scale=0.01)
    masses = ObsTerm(func=oracle.body_masses, scale=0.1)
    difficulty = ObsTerm(func=oracle.terrain_level)
    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class GaitOracleCfg(OracleCfg):
    """Oracle group extended with gait command and phase clock."""

    gait_command = ObsTerm(func=gait.command)
    gait_clock = ObsTerm(func=gait.clock)


@configclass
class ObservationsCfg:
    stack_policy: FrameCfg = FrameCfg()
    none_stack_policy: OracleCfg = OracleCfg()
    stack_critic: FrameCfg = FrameCfg()
    none_stack_critic: OracleCfg = OracleCfg()

@configclass
class GaitObservationsCfg(ObservationsCfg):
    none_stack_policy: GaitOracleCfg = GaitOracleCfg()
    none_stack_critic: GaitOracleCfg = GaitOracleCfg()

@configclass
class EventsCfg:
    reset_base = EventTermCfg(func=mdp.reset_root_state_uniform, mode="reset", params={
        "pose_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15), "yaw": (-0.05, 0.05)},
        "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                           "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)}})
    reset_joints = EventTermCfg(func=mdp.reset_joints_by_offset, mode="reset", params={
        "position_range": (-0.02, 0.02), "velocity_range": (0.0, 0.0)})

@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_contact = TerminationTermCfg(func=mdp.illegal_contact, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0})
    overturned = TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": 1.2})

@configclass
class WolfOracleEnvCfg(ManagerBasedRLEnvCfg):
    scene: WolfSceneCfg = WolfSceneCfg(num_envs=1024, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class FlatWolfSceneCfg(WolfSceneCfg):
    """Wolf scene using a generated flat mesh instead of stair sub-terrains."""

    def __post_init__(self):
        super().__post_init__()
        self.terrain.terrain_generator.sub_terrains = {"flat": MeshPlaneTerrainCfg(proportion=1.0)}
        self.terrain.max_init_terrain_level = 0


@configclass
class WolfGaitFlatEnvCfg(WolfOracleEnvCfg):
    """Flat-terrain fixed-trot environment for gait V0."""

    scene: FlatWolfSceneCfg = FlatWolfSceneCfg(num_envs=1024, env_spacing=3.0)
    observations: GaitObservationsCfg = GaitObservationsCfg()
    commands: GaitCommandsCfg = GaitCommandsCfg()
