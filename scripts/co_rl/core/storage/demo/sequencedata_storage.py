import torch
import numpy as np
import os
import heapq
import zarr
from collections import deque

# PPO의 RolloutStorage 클래스가 있다고 가정
from scripts.co_rl.core.storage import RolloutStorage

class SequenceDataStorage:
    def __init__(self,
                 num_envs: int,
                 obs_dim: int,
                 action_dim: int,
                 zarr_path: str,
                 # 파라미터 이름을 에피소드 개수 기준으로 변경
                 high_reward_episodes=500,
                 low_reward_episodes=100):

        # --- 버퍼 설정 ---
        self.high_reward_buffer = []
        self.low_reward_buffer = []
        # 용량 변수 이름을 변경하여 명확화
        self.high_reward_capacity = high_reward_episodes
        self.low_reward_capacity = low_reward_episodes

        # --- 상태 유지를 위한 에피소드 버퍼 ---
        self.episode_buffers = [
            {'reward_sum': 0.0, 'transitions': []} for _ in range(num_envs)
        ]

        # --- Zarr 데이터셋 설정 ---
        self.zarr_path = zarr_path
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._init_zarr_store()

    def _init_zarr_store(self):
        # (이전과 동일)
        if os.path.exists(self.zarr_path):
            self.root = zarr.open_group(self.zarr_path, mode='a')
        else:
            self.root = zarr.open_group(self.zarr_path, mode='w')
            self.root.create_dataset('observations', shape=(0, self.obs_dim), chunks=(1024, self.obs_dim), dtype='float32', maxshape=(None, self.obs_dim))
            self.root.create_dataset('actions', shape=(0, self.action_dim), chunks=(1024, self.action_dim), dtype='float32', maxshape=(None, self.action_dim))
            self.root.create_dataset('rewards', shape=(0, 1), chunks=(1024, 1), dtype='float32', maxshape=(None, 1))
            self.root.create_dataset('dones', shape=(0, 1), chunks=(1024, 1), dtype='bool', maxshape=(None, 1))
            self.root.create_dataset('next_observations', shape=(0, self.obs_dim), chunks=(1024, self.obs_dim), dtype='float32', maxshape=(None, self.obs_dim))
            self.root.create_dataset('timeouts', shape=(0, 1), chunks=(1024, 1), dtype='bool', maxshape=(None, 1))

    def add_step_data(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, timeouts: torch.Tensor, next_obs: torch.Tensor):
        """
        play.py와 같이 한 스텝씩 들어오는 데이터를 처리합니다. (next_obs 포함)
        """
        num_envs = obs.shape[0]
        
        # CPU로 데이터 이동
        obs, actions, rewards, dones, timeouts, next_obs = map(
            lambda x: x.cpu(), [obs, actions, rewards, dones, timeouts, next_obs]
        )

        for i in range(num_envs):
            transition = {
                'observations': obs[i],
                'actions': actions[i],
                'rewards': rewards[i].unsqueeze(0),
                'dones': dones[i].unsqueeze(0),
                'timeouts': timeouts[i].unsqueeze(0),
                'next_observations': next_obs[i] # <-- next_observations 추가
            }
            
            self.episode_buffers[i]['transitions'].append(transition)
            self.episode_buffers[i]['reward_sum'] += rewards[i].item()

            if dones[i] or timeouts[i]:
                finished_episode = self.episode_buffers[i]
                self._process_finished_episode(finished_episode)
                self.episode_buffers[i] = {'reward_sum': 0.0, 'transitions': []}
    
    
    def add_rollout_data(self, rollout_storage: RolloutStorage, last_observations: torch.Tensor):
        # (이전과 동일)
        num_steps, num_envs = rollout_storage.observations.shape[:2]

        obs_all = rollout_storage.observations.cpu()
        actions_all = rollout_storage.actions.cpu()
        rewards_all = rollout_storage.rewards.cpu()
        dones_all = rollout_storage.dones.cpu()
        timeouts_all = rollout_storage.time_outs.cpu()
        last_obs_all = last_observations.cpu()
    
        for t in range(num_steps):
            for i in range(num_envs):
                obs = obs_all[t, i]
                action = actions_all[t, i]
                reward = rewards_all[t, i]
                done = dones_all[t, i]
                timeout = timeouts_all[t, i]
                next_obs = obs_all[t + 1, i] if t < num_steps - 1 else last_obs_all[i]
                
                transition = {
                    'observations': obs, 'actions': action,
                    'rewards': reward, 'dones': done,
                    'next_observations': next_obs, 'timeouts': timeout
                }
                
                self.episode_buffers[i]['transitions'].append(transition)
                self.episode_buffers[i]['reward_sum'] += reward.item()

                if done or timeout:
                    finished_episode = self.episode_buffers[i]
                    self._process_finished_episode(finished_episode)
                    self.episode_buffers[i] = {'reward_sum': 0.0, 'transitions': []}

        # 출력 메시지를 에피소드 개수 기준으로 변경
        print(f"High-rew buffer: {len(self.high_reward_buffer)}/{self.high_reward_capacity} episodes, "
              f"Low-rew buffer: {len(self.low_reward_buffer)}/{self.low_reward_capacity} episodes")

    def _process_finished_episode(self, episode_data: dict):

        episode_reward = episode_data['reward_sum']
        transitions = episode_data['transitions']
        num_transitions = len(transitions)

        if num_transitions == 0:
            return

        data_chunk = None
        
        try:
            data_chunk = {key: torch.stack([t[key] for t in transitions]) for key in transitions[0]}

        except RuntimeError as e:
            print(f"\n[Warning] 데이터 처리 중 에러 발생, 해당 에피소드를 건너뜁니다.")
            print(f"  - 에러 메시지: {e}")
            print(f"  - 에피소드 길이: {num_transitions}, 보상: {episode_reward:.2f}\n")

        if data_chunk is not None:
            # --- High-Reward 버퍼 처리 (최소 힙) ---
            if len(self.high_reward_buffer) < self.high_reward_capacity:
                heapq.heappush(self.high_reward_buffer, (episode_reward, data_chunk, num_transitions))
            elif self.high_reward_buffer and episode_reward > self.high_reward_buffer[0][0]:
                heapq.heapreplace(self.high_reward_buffer, (episode_reward, data_chunk, num_transitions))

            # --- Low-Reward 버퍼 처리 (최대 힙, 음수 보상 사용) ---
            neg_reward = -episode_reward
            if len(self.low_reward_buffer) < self.low_reward_capacity:
                heapq.heappush(self.low_reward_buffer, (neg_reward, data_chunk, num_transitions))
            elif self.low_reward_buffer and neg_reward > self.low_reward_buffer[0][0]:
                heapq.heapreplace(self.low_reward_buffer, (neg_reward, data_chunk, num_transitions))
    
    def flush_to_zarr(self):
        all_chunks = [item[1] for item in self.high_reward_buffer] + \
                     [item[1] for item in self.low_reward_buffer]
        
        if not all_chunks:
            print("Warning: No data in buffer to flush.")
            return

        num_flushed = 0
        for chunk in all_chunks:
            self.root['observations'].append(chunk['observations'].numpy())
            self.root['actions'].append(chunk['actions'].numpy())
            self.root['rewards'].append(chunk['rewards'].numpy())
            self.root['dones'].append(chunk['dones'].numpy())
            self.root['next_observations'].append(chunk['next_observations'].numpy())
            self.root['timeouts'].append(chunk['timeouts'].numpy())
            num_flushed += len(chunk['observations'])

        print(f"Successfully flushed {num_flushed} transitions to Zarr store.")
        self.clear()

    def clear(self):
        self.high_reward_buffer.clear()
        self.low_reward_buffer.clear()