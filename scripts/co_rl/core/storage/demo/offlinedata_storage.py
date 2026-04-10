import torch
import numpy as np
import os
import heapq
import zarr 
# 사용자께서 제공해주신 RolloutStorage 클래스가 있다고 가정합니다.
# from your_ppo_code import RolloutStorage 
from scripts.co_rl.core.storage import RolloutStorage

class OfflineDataStorage:
    """
    PPO의 RolloutStorage로부터 (s, a, r, s', d) 튜플을 수집하여
    오프라인 강화학습용 데이터셋으로 저장하는 클래스
    """
    def __init__(self, 
                 obs_dim : int,
                 action_dim : int,
                 zarr_path : str,
                 high_reward_capacity=900000, 
                 low_reward_capacity=100000):
        # 버퍼는 (총 보상, 데이터 딕셔너리) 튜플을 저장합니다.
        self.high_reward_buffer = []
        self.low_reward_buffer = []
        
        # 용량은 transition 개수 기준
        # num_envs * sampling_ratio =  
        self.high_reward_capacity = high_reward_capacity
        self.low_reward_capacity = low_reward_capacity
        self.high_reward_size = 0
        self.low_reward_size = 0
        
                # --- Zarr 데이터셋 설정 ---
        self.zarr_path = zarr_path
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._init_zarr_store()
    
    def _init_zarr_store(self):
        if os.path.exists(self.zarr_path):
            print(f"Zarr store found at {self.zarr_path}. Re-opening.")
            self.root = zarr.open_group(self.zarr_path, mode='a')
        else:
            print(f"Creating new Zarr store at {self.zarr_path}")
            self.root = zarr.open_group(self.zarr_path, mode='w')
            # 크기 변경이 가능한(resizable) 배열 생성
            # shape=(0, dim)으로 시작, maxshape=(None, dim)으로 무한정 추가 가능
            self.root.create_dataset('observations', shape=(0, self.obs_dim), chunks=(1024, self.obs_dim), dtype='float32', maxshape=(None, self.obs_dim))
            self.root.create_dataset('actions', shape=(0, self.action_dim), chunks=(1024, self.action_dim), dtype='float32', maxshape=(None, self.action_dim))
            self.root.create_dataset('rewards', shape=(0, 1), chunks=(1024, 1), dtype='float32', maxshape=(None, 1))
            self.root.create_dataset('dones', shape=(0, 1), chunks=(1024, 1), dtype='bool', maxshape=(None, 1))
            self.root.create_dataset('timeouts', shape=(0, 1), chunks=(1024, 1), dtype='bool', maxshape=(None, 1))
            self.root.create_dataset('next_observations', shape=(0, self.obs_dim), chunks=(1024, self.obs_dim), dtype='float32', maxshape=(None, self.obs_dim))
        

    def add_rollout_data(self, rollout_storage: RolloutStorage, last_observations: torch.Tensor, last_privileged_observations: torch.Tensor = None, sampling_ratio : float = 1/64):
        """
        한 번의 PPO 롤아웃이 끝난 후 RolloutStorage에서 데이터를 추출하여 추가합니다.

        Args:
            rollout_storage (RolloutStorage): 데이터가 가득 찬 RolloutStorage 객체.
            last_observations (torch.Tensor): 롤아웃의 마지막 스텝 이후에 환경에서 받은 observation.
                                              (num_envs, *obs_shape) 형태여야 합니다.
            last_privileged_observations (torch.Tensor, optional): last_observations에 해당하는 privileged observation.
        """
        num_envs = rollout_storage.observations.shape[1]
        device = rollout_storage.observations.device

        if sampling_ratio < 1.0:
            num_to_sample = int(num_envs * sampling_ratio)
            sampled_indices = torch.randperm(num_envs, device=device)[:num_to_sample]
        else:
            num_to_sample = num_envs
            sampled_indices = torch.arange(num_envs, device=device)

        obs_from_storage = rollout_storage.observations[:, sampled_indices]
        actions_from_storage = rollout_storage.actions[:, sampled_indices]
        rewards_from_storage = rollout_storage.rewards[:, sampled_indices]
        dones_from_storage = rollout_storage.dones[:, sampled_indices]
        time_outs_from_storage = rollout_storage.time_outs[:, sampled_indices]
        last_obs_sampled = last_observations[sampled_indices]
        
        next_obs = torch.cat([obs_from_storage[1:], last_obs_sampled.unsqueeze(0)], dim=0)
        
        obs_flat = obs_from_storage.permute(1, 0, 2).flatten(0, 1)
        actions_flat = actions_from_storage.permute(1, 0, 2).flatten(0, 1)
        rewards_flat = rewards_from_storage.permute(1, 0, 2).flatten(0, 1)
        dones_flat = dones_from_storage.permute(1, 0, 2).flatten(0, 1)
        time_outs_flat = time_outs_from_storage.permute(1, 0, 2).flatten(0, 1)
        next_obs_flat = next_obs.permute(1, 0, 2).flatten(0, 1)

        # rollout_reward는 샘플링된 데이터의 총합으로 계산
        rollout_reward = rewards_from_storage.sum().item()
        
        num_transitions = len(obs_flat)
        data_chunk = {
            'observations': obs_flat.cpu(), 'actions': actions_flat.cpu(),
            'rewards': rewards_flat.cpu(), 'dones': dones_flat.cpu(),
            'time_outs' : time_outs_flat.cpu(),'next_observations': next_obs_flat.cpu()
        }

        # 2. High-Reward 버퍼 처리 (최소 힙)
        if self.high_reward_size < self.high_reward_capacity:
            heapq.heappush(self.high_reward_buffer, (rollout_reward, data_chunk, num_transitions))
            self.high_reward_size += num_transitions
        elif rollout_reward > self.high_reward_buffer[0][0]: # 현재 보상이 힙의 최소값보다 크면
            # 가장 작은 요소를 제거하고 새 요소를 추가 (더 효율적인 heapreplace 사용)
            removed_reward, removed_chunk, removed_size = heapq.heapreplace(
                self.high_reward_buffer, (rollout_reward, data_chunk, num_transitions)
            )
            self.high_reward_size += (num_transitions - removed_size)

        # 3. Low-Reward 버퍼 처리 (최대 힙)
        neg_reward = -rollout_reward
        if self.low_reward_size < self.low_reward_capacity:
            heapq.heappush(self.low_reward_buffer, (neg_reward, data_chunk, num_transitions))
            self.low_reward_size += num_transitions
        elif neg_reward > self.low_reward_buffer[0][0]: # 현재 음수 보상이 힙의 최소값(원래 보상의 최대값)보다 크면
            removed_neg_reward, removed_chunk, removed_size = heapq.heapreplace(
                self.low_reward_buffer, (neg_reward, data_chunk, num_transitions)
            )
            self.low_reward_size += (num_transitions - removed_size)
            
        print(f"High-rew buffer: {self.high_reward_size}/{self.high_reward_capacity}, Low-rew buffer: {self.low_reward_size}/{self.low_reward_capacity}")

    def flush_to_zarr(self):
        """
        메모리 버퍼에 쌓인 데이터를 Zarr 데이터셋에 '청크 단위'로 추가하고 버퍼를 비웁니다.
        """
        all_chunks = [item[1] for item in self.high_reward_buffer] + \
                     [item[1] for item in self.low_reward_buffer]
        
        if not all_chunks:
            print("Warning: No data in buffer to flush.")
            return

        num_flushed = 0
        # torch.cat 없이, 각 청크를 순회하며 Zarr에 바로 추가합니다.
        for chunk in all_chunks:
            for key, data_tensor in chunk.items():
                self.root[key].append(data_tensor.numpy())
            
            num_flushed += len(chunk['observations'])

        print(f"Successfully flushed {num_flushed} transitions to Zarr store chunk by chunk.")
        self.clear()

    def clear(self):
        self.high_reward_buffer.clear()
        self.low_reward_buffer.clear()
        self.high_reward_size = 0
        self.low_reward_size = 0
    
