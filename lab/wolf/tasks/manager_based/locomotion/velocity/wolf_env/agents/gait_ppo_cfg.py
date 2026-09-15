"""PPO configuration for the 323-dimensional fixed-trot gait V0."""

from isaaclab.utils import configclass

from .co_rl_cfg import WolfStairPPORunnerCfg


@configclass
class WolfGaitPPORunnerCfg(WolfStairPPORunnerCfg):
    max_iterations = 1500
    experiment_name = "Wolf_v2_Flat_Gait_Trot_PPO"
    experiment_description = "Fixed trot gait command, 2 Hz, 50 percent duty, flat terrain"
