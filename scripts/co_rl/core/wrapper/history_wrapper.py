import torch

class HistoryWrapper:
    def __init__(self, env, history_len, device="cpu"):
        self.env = env
        self.history_len = history_len
        self.device = device
        
        # 원본 환경의 정보 가져오기
        self.num_envs = env.num_envs
        self.num_obs = env.num_obs
        
        # (num_envs, history_len, num_obs)
        self.obs_history = torch.zeros(
            (self.num_envs, self.history_len, self.num_obs), 
            device=self.device, 
            dtype=torch.float
        )
        
        self.num_obs = self.num_obs * self.history_len 

    def step(self, action):
        obs, rewards, dones, infos = self.env.step(action)
        
        self.obs_history[:, :-1, :] = self.obs_history[:, 1:, :].clone()
        self.obs_history[:, -1, :] = obs.to(self.device)
        
        reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            reset_obs = obs[reset_env_ids] 
            
            # (N_reset, Obs_Dim) -> (N_reset, History_Len, Obs_Dim)으로 확장하여 덮어쓰기
            self.obs_history[reset_env_ids] = reset_obs.unsqueeze(1).expand(-1, self.history_len, -1).clone()
            
        obs_history = self.obs_history.view(self.num_envs, -1)
        return obs_history, rewards, dones, infos

    def reset(self):
        obs = self.env.reset() # (Num_Envs, Obs_Dim)
    
        self.obs_history = obs.unsqueeze(1).expand(-1, self.history_len, -1).clone()
        
        return self.obs_history.view(self.num_envs, -1)

    def get_observations(self):
        # Runner에서 초기 obs 가져올 때 호출됨
        obs = self.env.get_observations() # (Num_Envs, Obs_Dim) 일 수 있음
        # wrapper 입장에서는 history 버퍼를 리턴해야 함
        return self.obs_history.view(self.num_envs, -1)

    # ... 기타 속성 접근을 위한 getattr ...
    def __getattr__(self, name):
        return getattr(self.env, name)