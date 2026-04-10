# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher
import matplotlib.pyplot as plt

# local imports
import cli_args  # isort: skip
from scripts.co_rl.core.runners import OffPolicyRunner
from scripts.co_rl.core.modules.teacher_student import RMAStudent
from scripts.co_rl.core.utils.str2bool import str2bool

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with CO-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--algo", type=str, default="ppo", help="Name of the task.")
parser.add_argument("--stack_frames", type=int, default=None, help="Number of frames to stack.")
parser.add_argument("--plot", type=str2bool, default="False", help="Plot the data.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

parser.add_argument("--num_policy_stacks", type=int, default=None, help="Number of policy stacks.")
parser.add_argument("--num_critic_stacks", type=int, default=None, help="Number of critic stacks.")
parser.add_argument(
    "--store_expert_data", 
    action='store_true', 
    help="이 플래그를 사용하면 오프라인 데이터를 저장합니다."
)
parser.add_argument("--zarr_filename", type=str, default="expert_data", help="저장될 Zarr 파일의 이름입니다.")
parser.add_argument(
    "--store_training_data", 
    action='store_true', 
    help="이 플래그를 사용하면 오프라인 데이터를 저장합니다."
)

# append CO-RL cli arguments
cli_args.add_co_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from scripts.co_rl.core.runners import OnPolicyRunner, SRMOnPolicyRunner, MOO_OnPolicyRunner
from isaaclab.utils.dict import print_dict

from scripts.co_rl.core.wrapper import (
    CoRlPolicyRunnerCfg,
    CoRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    export_srm_as_onnx,
    export_student_onnx
    
)

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint


# Import extensions to set up environment tasks
import lab.flamingo.tasks  # noqa: F401
from lab.flamingo.isaaclab.isaaclab.envs import ManagerBasedConstraintRLEnv, ManagerBasedConstraintRLEnvCfg

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

from scripts.co_rl.core.storage import SequenceDataStorage

def _get_privileged_obs(obs_dict, device):
    """[Helper] obs_dict에서 priv_ 키들을 찾아 합쳐서 반환합니다."""
    if obs_dict is None:
        return None
    
    # 1. 'priv_'로 시작하는 키 찾기 (Sorted로 순서 보장 필수)
    priv_keys = sorted([k for k in obs_dict.keys() if k.startswith("priv_")])
    
    if len(priv_keys) > 0:
        priv_tensors = [obs_dict[k] for k in priv_keys]
        return torch.cat(priv_tensors, dim=-1).to(device)
    
    # 2. 기존 방식 호환
    elif "privileged" in obs_dict:
        return obs_dict["privileged"].to(device)
    
    return None

def main():
    """Play with CO-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: CoRlPolicyRunnerCfg = cli_args.parse_co_rl_cfg(args_cli.task, args_cli)
    agent_cfg.num_policy_stacks = args_cli.num_policy_stacks if args_cli.num_policy_stacks is not None else agent_cfg.num_policy_stacks
    agent_cfg.num_critic_stacks = args_cli.num_critic_stacks if args_cli.num_critic_stacks is not None else agent_cfg.num_critic_stacks
    agent_cfg.store_training_data = args_cli.store_training_data
    
    is_off_policy = False if agent_cfg.to_dict()["algorithm"]["class_name"] in ["PPO", "SRMPPO", "ACAPSPPO", "SIEPPO", "MOOPPO"] else True
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "co_rl", agent_cfg.experiment_name, args_cli.algo)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("co_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if isinstance(env.unwrapped, ManagerBasedConstraintRLEnv):
        agent_cfg.use_constraint_rl = True

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for co-rl
    env = CoRlVecEnvWrapper(env, agent_cfg)

        # --- SequenceDataStorage 초기화 코드 추가 ---
    storage = None
    if args_cli.store_expert_data:
        zarr_path = os.path.join(log_dir, args_cli.zarr_filename)
        print(f"[INFO] 오프라인 데이터를 다음 경로에 저장합니다: {zarr_path}")

        storage = SequenceDataStorage(
            num_envs=args_cli.num_envs,
            obs_dim=env.num_obs,
            action_dim=env.num_actions,
            zarr_path=zarr_path,
            high_reward_episodes=500, # 플레이 데이터는 보통 고품질이므로 많이 저장
            low_reward_episodes=0
        )
        # --- 카운터 및 저장 주기 설정 ---
        finished_episodes_count = 0
        stored_episodes = 0
        max_stored_episodes = 1000


    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    if is_off_policy:
        runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        if args_cli.algo == "mooppo":
            runner = MOO_OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif args_cli.algo == "ppo":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    runner.load(resume_path)
    
    
    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if is_off_policy:
        export_policy_as_jit(runner.alg, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            runner.alg, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )
    else:
        export_policy_as_jit(
            runner.alg.actor_critic, runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
        )
        if isinstance(runner.alg.actor_critic, RMAStudent):
            # 기존 개별 export 대신 통합 export 호출
            export_policy_as_onnx(
                runner.alg.actor_critic, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
            )
        
            export_student_onnx(runner.alg.actor_critic, export_model_dir, device=agent_cfg.device)
        else:
            # Student가 아닌 일반 PPO 등은 기존 방식대로
            export_policy_as_onnx(
                runner.alg.actor_critic, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
            )
        
        if args_cli.algo == "srmppo":
            export_srm_as_onnx(
                runner.alg.srm, runner.alg.srm_fc, device=agent_cfg.device, path=export_model_dir, filename="srm.onnx"
            )


    # TODO Should Generalize this using argparser
    # Initialize lists to store data
    joint_pos_list = []
    target_joint_pos_list = []
    joint_velocity_obs_list = []
    target_joint_velocity_list = []

    # reset environment
    obs, extras = env.get_observations()
    privileged_obs = _get_privileged_obs(extras.get("observations"), agent_cfg.device)
        
    if isinstance(runner.alg.actor_critic, RMAStudent):
        print("[Inference] Initializing Student History Buffer...")
        # dones=1로 주어 "새로 태어난 상태"로 인식시켜 버퍼를 현재 obs로 도배
        runner.alg.actor_critic.update_history(obs, dones=torch.ones(env.num_envs, device=env.device))    
    
    timestep = 0
    width = 80
    pad = 35
    # Simulate environment and collect data
    while simulation_app.is_running():
        prev_obs = obs
        with torch.inference_mode():
            if isinstance(runner.alg.actor_critic, RMAStudent):
                current_dones = dones if 'dones' in locals() else torch.zeros(env.num_envs, device=env.device)
                runner.alg.actor_critic.update_history(obs, current_dones)

            if privileged_obs is not None:
                try:
                    actions = policy(obs, privileged_info = privileged_obs)
                except TypeError:
                    actions = policy(obs)
            else:
                # Student (Blind) -> 내부 버퍼 사용
                actions = policy(obs)

            obs, rewards, dones, infos = env.step(actions)
            
            if "observations" in infos:
                privileged_obs = _get_privileged_obs(infos["observations"], agent_cfg.device)
            else:
                privileged_obs = None
            
            if hasattr(runner.moo_reward_manager, "debug_metrics"):
                moo_metrics = runner.moo_reward_manager.debug_metrics
            else:
                moo_metrics = None
            
            if moo_metrics is not None:
                moo_str = (f"""{'-' * width}\n""" \
                  f"""{'MOO/ADD Metrics'.center(width, ' ')}\n""" \
                  f"""{f'Mean D(Δ) Prob:':>{pad}} {moo_metrics['mean_discriminator_prob']:.4f}\n""" \
                  f"""{f'Mean Raw Reward:':>{pad}} {moo_metrics['mean_raw_reward']:.4f}\n""" \
                  f"""{f'Mean Normed |Δ|:':>{pad}} {moo_metrics['mean_norm_delta_abs']:.4f}\n""" \
                )
                if "per_feature_abs_mean" in moo_metrics :
                    moo_str += f"""{f'Mean Raw |Δ| per feature:':>{pad}}\n"""
                    
                    for key, value in moo_metrics['per_feature_abs_mean'].items():
                        key_str = f"    {key}:" 
                        moo_str += f"""{f'{key_str}':>{pad}} {value:.4f}\n"""
                #print(moo_str)
            
            if args_cli.store_expert_data and storage is not None:
                real_dones = dones
                real_timeouts = infos["time_outs"]
                
                storage.add_step_data(
                    prev_obs, actions, rewards, real_dones, real_timeouts, obs
                )
                
                num_episodes_ended = torch.sum(dones).item()
                if num_episodes_ended > 0:
                    finished_episodes_count += num_episodes_ended
                    if finished_episodes_count > storage.high_reward_capacity * 3 and max_stored_episodes > stored_episodes:
                        stored_episodes += storage.high_reward_capacity
                        finished_episodes_count = 0
                        print(f"\n[INFO] {stored_episodes}개의 에피소드 데이터를 저장합니다...")
                        storage.flush_to_zarr()
            
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        # Extract the relevant slices and convert to numpy
        joint_pos = obs[0, :6].cpu().numpy()
        target_joint_pos = (obs[0, 20:26] * 2.0).cpu().numpy()
        joint_velocity_obs = obs[0, 12:14].cpu().numpy()
        target_joint_velocity = (obs[0, 26:28] * 20.0).cpu().numpy()

        # Store the data
        joint_pos_list.append(joint_pos)
        target_joint_pos_list.append(target_joint_pos)
        joint_velocity_obs_list.append(joint_velocity_obs)
        target_joint_velocity_list.append(target_joint_velocity)
        

                

    env.close()

    if args_cli.plot == "True":
        plt.figure(figsize=(14, 16))

        for i in range(6):
            plt.subplot(4, 2, i + 1)
            plt.plot([step[i] for step in joint_pos_list], label=f"Joint Position {i+1}")
            plt.plot([step[i] for step in target_joint_pos_list], label=f"Target Joint Position {i+1}", linestyle="--")
            plt.title(f"Joint Position {i+1} and Target Joint Position", fontsize=10, pad=10)
            plt.legend()

        for i in range(2):
            plt.subplot(4, 2, i + 7)
            plt.plot([step[i] for step in joint_velocity_obs_list], label=f"Observed Joint Velocity {i+1}")
            plt.plot(
                [step[i] for step in target_joint_velocity_list], label=f"Target Joint Velocity {i+1}", linestyle="--"
            )
            plt.title(f"Observed and Target Joint Velocity {i+1}", fontsize=10, pad=10)
            plt.legend()

        plt.tight_layout()
        plt.show()
        
        if args_cli.save_offline_data and storage is not None:
            print("최종적으로 버퍼에 남아있는 데이터를 파일로 저장합니다...")
            storage.flush_to_zarr()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()