"""Repository Wolf v2 URDF plant for the stairs project (no stair task yet)."""
from pathlib import Path
import fcntl
import hashlib
import os
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg


def _spawn_wolf_urdf(prim_path, cfg, *args, **kwargs):
    # The importer uses shared temporary mesh filenames. Serialize cache writers.
    with (Path(cfg.usd_dir) / ".conversion.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return sim_utils.spawn_from_urdf(prim_path, cfg, *args, **kwargs)


def wolf_v2_cfg() -> ArticulationCfg:
    """Resolve source meshes and cache conversion by URDF + mesh content.

    WOLF_URDF_PATH supports standalone checkouts. Gains are initial simulation
    values, not calibrated hardware parameters or a validated standing policy.
    """
    repo = Path(__file__).resolve().parents[3]
    source = Path(os.environ.get(
        "WOLF_URDF_PATH", str(repo.parent.parent / "URDF/Wolf_v_2_0_0/urdf/Wolf_v_2_0_0.urdf")
    )).resolve()
    tree = ET.parse(source)
    digest = hashlib.sha256(source.read_bytes())
    for mesh in tree.iter("mesh"):
        name = mesh.attrib["filename"]
        if name.startswith("package://Wolf_v_2_0_0/"):
            path = source.parent.parent / name.removeprefix("package://Wolf_v_2_0_0/")
        else:
            path = source.parent / name
        path = path.resolve(strict=True)
        data = path.read_bytes()
        if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"Git LFS mesh missing: {path}; run git lfs pull in the parent repository")
        digest.update(str(path).encode())
        digest.update(data)
        mesh.set("filename", str(path))
    cache = Path(os.environ.get("WOLF_ASSET_CACHE", str(repo / "outputs/wolf_assets"))) / digest.hexdigest()[:16]
    cache.mkdir(parents=True, exist_ok=True)
    normalized = cache / "Wolf_v_2_0_0.urdf"
    content = ET.tostring(tree.getroot(), encoding="utf-8", xml_declaration=True)
    if not normalized.exists() or normalized.read_bytes() != content:
        normalized.write_bytes(content)
    joints = [j for j in tree.findall("joint") if j.get("type") != "fixed"]
    actuators = {}
    for joint in joints:
        name = joint.attrib["name"]
        limit = joint.find("limit")
        effort, velocity = float(limit.attrib["effort"]), float(limit.attrib["velocity"])
        actuators[name] = IdealPDActuatorCfg(
            joint_names_expr=[name], stiffness=80.0, damping=1.0, armature=0.01,
            effort_limit=effort, effort_limit_sim=effort,
            velocity_limit=velocity, velocity_limit_sim=velocity,
        )
    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            func=_spawn_wolf_urdf,
            asset_path=str(normalized), usd_dir=str(cache), usd_file_name="Wolf_v_2_0_0.usd",
            fix_base=False, merge_fixed_joints=False, make_instanceable=True,
            activate_contact_sensors=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                target_type="none", gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.9), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        soft_joint_pos_limit_factor=0.8,
        actuators=actuators,
    )
