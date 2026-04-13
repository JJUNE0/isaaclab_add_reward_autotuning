# CLAUDE.md — Flamingo Reward Autotuning 프로젝트 가이드

> **이 문서는 Claude Code가 코드베이스를 이해하고 수정하기 위한 가이드입니다.**

## 프로젝트 개요

- **목적**: 휠-바이페드 로봇(Flamingo)의 blind locomotion을 위한 RL 프레임워크
- **핵심 기술**: Context-Aware Discriminator + Confidence-Aware Curriculum (t-test LCB) + RMA (Teacher-Student)
- **기술 스택**: Isaac Sim 4.0.0, IsaacLab 1.1.0, Python 3.10, PyTorch, RSL-RL 기반 커스텀 co_rl

## 프로젝트 구조

```
├── lab/flamingo/
│   ├── isaaclab/isaaclab/
│   │   ├── envs/                          # ManagerBasedMOORLEnv
│   │   └── managers/
│   │       ├── domain_managers/           # Confidence-Aware Curriculum (DomainManager)
│   │       └── moo_managers/              # MOORewardManager (Discriminator 기반 보상)
│   ├── tasks/moo_based/locomotion/velocity/
│   │   ├── mdp/                           # observations, rewards, events, commands, terminations
│   │   ├── sensors/                       # LiftMask 커스텀 센서
│   │   ├── flamingo_env/                  # 2-wheel-2-leg 로봇 환경 (Flamingo rev03)
│   │   ├── flamingo_4w4l_env/             # 4-wheel-4-leg 로봇 환경
│   │   └── flamingo_light_env/            # 경량 로봇 환경
│   └── assets/flamingo/                   # URDF/USD 로봇 정의
├── scripts/co_rl/
│   ├── core/
│   │   ├── algorithms/                    # PPO, MOOPPO, Distillation 등
│   │   ├── modules/                       # ActorCritic, Discriminator, RMATeacher/Student
│   │   ├── runners/                       # OnPolicyRunner, MOO_OnPolicyRunner
│   │   ├── storage/                       # RolloutStorage, MOORolloutStorage
│   │   ├── wrapper/                       # CoRlVecEnvWrapper, ONNX exporter
│   │   └── utils/                         # DiffNormalizer, EMA, StateHandler
│   ├── train.py                           # 학습 진입점
│   └── play.py                            # 추론/배포 진입점
```

## 핵심 아키텍처 용어 매핑 (논문 ↔ 코드)

| 논문 용어                        | 코드 용어/파일                                  |
|----------------------------------|------------------------------------------------|
| Context-Aware Discriminator      | `DiscriminatorMLP` (discriminator.py)          |
| Confidence-Aware Curriculum      | `DomainManager` (domain_manager.py)            |
| Harmonic Potential               | `reward_pot` in MOORewardManager.compute()     |
| Feature Matching                 | `reward_fm` in MOORewardManager.compute()      |
| Privileged Encoder (SN)          | `Encoder` (teacher_student.py)                 |
| Adaptation Module (Dilated TCN)  | `AdaptationModule` (teacher_student.py)        |
| MOO Reward Terms                 | `MOORewardTermCfg` + feature_functions.py      |

---

## 🐛 수정이 필요한 버그 목록

### BUG-01 [CRITICAL] `base_lin_vel_z_link` 가 y축 속도를 반환

**파일**: `scripts/co_rl/core/modules/../../tasks/moo_based/locomotion/velocity/mdp/observations.py`

**현재 코드** (약 line 35):
```python
def base_lin_vel_z_link(env, asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b[:, 1].unsqueeze(-1)  # BUG: 인덱스 1 = Y축
```

**수정**:
```python
def base_lin_vel_z_link(env, asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b[:, 2].unsqueeze(-1)  # FIX: 인덱스 2 = Z축
```

**영향**: Critic/Student 학습에 잘못된 속도 정보가 전달됨

---

### BUG-02 [CRITICAL] LSTM Actuator에서 inference가 2회 실행

**파일**: `lab/flamingo/tasks/moo_based/locomotion/velocity/actuators/actuator_net.py`

**현재 코드** (`ActuatorNetLSTM.compute()` 메서드 내부):
```python
with torch.inference_mode():
    torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.network(
        self.sea_input, (self.sea_hidden_state, self.sea_cell_state)
    )

# run network inference  ← 이 아래 블록 전체 삭제!
with torch.inference_mode():
    torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.network(
        self.sea_input, (self.sea_hidden_state, self.sea_cell_state)
    )
```

**수정**: 두 번째 `with torch.inference_mode():` 블록 전체를 삭제. hidden state가 2번 업데이트되어 시간 축이 뒤틀림.

---

### BUG-03 [CRITICAL] `DiscriminatorGRU`에 `extract_features` 미정의

**파일**: `scripts/co_rl/core/modules/discriminator.py`

**문제**: `MOORewardManager.compute()`에서 `motion` 모드일 때 `self._discriminator.extract_features(seq, obs)`를 호출하지만, `DiscriminatorGRU`에는 이 메서드가 없어 `AttributeError` 발생.

**수정**: `DiscriminatorGRU` 클래스에 다음 메서드 추가:
```python
def extract_features(self, x_seq: torch.Tensor, obs: torch.Tensor = None) -> torch.Tensor:
    """GRU의 마지막 hidden state를 feature로 반환"""
    out, _ = self.gru(x_seq)  # (N, T, H)
    return out[:, -1]          # (N, H)
```

또한 `DiscriminatorGRU.forward()`의 시그니처도 `DiscriminatorBase`와 맞춰야 함:
```python
def forward(self, x_seq: torch.Tensor, obs: torch.Tensor = None) -> torch.Tensor:
    # obs는 현재 GRU에서 사용하지 않지만, 인터페이스 통일을 위해 받아둠
    out, _ = self.gru(x_seq)
    h_last = out[:, -1]
    return self.head(h_last)
```

---

### BUG-04 [CRITICAL] Distillation RolloutStorage 오타

**파일**: `scripts/co_rl/core/storage/distillation_rollout_storage.py`

**현재 코드** (`reccurent_mini_batch_generator` 내부):
```python
last_was_done[0] = Tr   # NameError: 'Tr' is not defined
```

**수정**:
```python
last_was_done[0] = True
```

---

### BUG-05 [HIGH] `RMAStudent.act()` — Actor가 2번 실행됨

**파일**: `scripts/co_rl/core/modules/teacher_student.py`

**현재 흐름**:
1. `forward()` → `self.actor(policy_in)` → `action_mean` 반환
2. `act()` → `self.update_distribution(action_mean)` 호출
3. `update_distribution(obs)` (부모 메서드) → `self.actor(obs)` 다시 실행

즉, actor가 2번 실행되고, 두 번째 실행에서는 이미 action인 값을 입력으로 사용함.

**수정**: `RMAStudent`에 `update_distribution` 오버라이드 추가:
```python
def update_distribution(self, action_mean):
    """action_mean을 직접 분포의 mean으로 사용 (actor를 다시 실행하지 않음)"""
    self.distribution = torch.distributions.Normal(action_mean, action_mean * 0.0 + self.std)
```

---

### BUG-06 [HIGH] Volume-Preserving Update 수식 오류

**파일**: `lab/flamingo/isaaclab/isaaclab/managers/domain_managers/domain_manager.py`

**현재 코드**:
```python
self.rho = math.exp(cfg.kappa / max(self.param_dim, 1)) - 1.0
```

**논문 Eq.10**: ρ = exp(κ / √d_param) - 1

**수정**:
```python
self.rho = math.exp(cfg.kappa / math.sqrt(max(self.param_dim, 1))) - 1.0
```

---

### BUG-07 [HIGH] `error_foot_clearance`에서 scale 이중 적용

**파일**: `lab/flamingo/tasks/moo_based/locomotion/velocity/flamingo_env/rough_env/stand_drive/feature_functions.py` (및 동일한 다른 복사본들)

**현재 코드**:
```python
error = torch.clamp(target_foot_z - current_foot_z, min=0.0) * scale  # 1차 scale
...
return apply_kernel(error, kernel, scale, temperature)                  # 2차 scale 적용!
```

**수정**: 첫 번째 `* scale`을 제거:
```python
error = torch.clamp(target_foot_z - current_foot_z, min=0.0)  # scale 제거
...
return apply_kernel(error, kernel, scale, temperature)
```

---

### BUG-08 [MEDIUM] Hybrid Reward 가중치 하드코딩

**파일**: `lab/flamingo/isaaclab/isaaclab/managers/moo_managers/moo_reward_manager.py`

**현재 코드** (`compute()` 내부):
```python
w_pot = 0.0    # Potential 보상이 사용되지 않음
w_disc = 0.2
w_fm = 0.8
```

**수정 방향**: 
1. `ADDCfg`에 `w_pot`, `w_disc`, `w_fm` 필드 추가
2. `MOOPPO.__init__`에서 `self.moo_manager`에 가중치 전달
3. `MOORewardManager.__init__`에서 인스턴스 변수로 저장
4. `compute()`에서 `self.w_pot`, `self.w_disc`, `self.w_fm` 사용

또한 `w_pot=0.0`일 때 `reward_pot` 계산 자체를 스킵하는 분기 추가:
```python
if self.w_pot > 0:
    U = 0.5 * (self._delta_buf ** 2).mean(dim=1)
    reward_pot = torch.exp(-self.alpha * U)
else:
    reward_pot = torch.zeros(self.num_envs, device=self.device)
```

---

### BUG-09 [MEDIUM] `_capture_max_params` 하드코딩된 term 이름

**파일**: `lab/flamingo/isaaclab/isaaclab/managers/domain_managers/domain_manager.py`

**현재 코드**:
```python
target_terms = ["randomize_mass", "randomize_com", "randomize_gains", 
                "randomize_joints", "randomize_friction", "randomize_base_mass"]
```

**수정**: 자동 탐지 방식으로 변경:
```python
def _capture_max_params(self):
    if isinstance(self.cfg, dict):
        cfg_items = self.cfg.items()
    else:
        cfg_items = self.cfg.__dict__.items()
    
    for term_name, term_cfg in cfg_items:
        if isinstance(term_cfg, EventTermCfg) and hasattr(term_cfg, "params") and term_cfg.params:
            self.max_params[term_name] = deepcopy(term_cfg.params)
```

---

### BUG-10 [LOW] feature_functions.py 6중 복사

**관련 파일**:
- `flamingo_env/flat_env/stand_drive/feature_functions.py`
- `flamingo_env/flat_env/recovery/feature_functions.py`
- `flamingo_env/rough_env/stand_drive/feature_functions.py`
- `flamingo_4w4l_env/flat_env/stand_drive/feature_functions.py`
- `flamingo_4w4l_env/rough_env/stand_drive/feature_functions.py`
- `flamingo_light_env/flat_env/stand_drive/feature_functions.py`

**수정 방향**:
1. 공통 함수들을 `lab/flamingo/tasks/moo_based/locomotion/velocity/mdp/feature_functions_common.py`로 추출
2. 각 환경별 feature_functions.py에서 `from ...mdp.feature_functions_common import *`로 import
3. 환경 고유 함수만 각 파일에 유지

공통으로 추출할 함수 목록:
- `apply_kernel`, `error_track_lin_vel_xy`, `error_track_ang_vel_z`, `error_base_height`
- `error_flat_euler_rp`, `error_joint_deviation_huber`, `error_joint_deviation`
- `action_rate`, `action_rate_huber`, `action_rate_l2`, `joint_acc_l2`, `dof_acc`
- `ActionRatePenalty`, `TorqueRatePenalty`

---

## 📋 Phase별 수정 계획

> **Claude Code에게 한 Phase씩 지시하세요.** 각 Phase는 독립적으로 동작하며, 이전 Phase의 완료를 전제하지 않습니다 (Phase 5 제외).
> 
> **지시 예시**: "CLAUDE.md의 Phase 1을 수행해줘. 수정 후 각 파일의 diff를 보여줘."

### Phase 1: 원라인 Critical 버그 (BUG-01, 02, 04)

**난이도**: ⭐ | **예상 소요**: 5분 | **의존성**: 없음

단순 수정 3건. 각각 1줄만 바꾸면 되고, 서로 독립적.

| 버그 | 파일 | 수정 내용 |
|------|------|-----------|
| BUG-01 | `mdp/observations.py` | `[:, 1]` → `[:, 2]` (Z축 인덱스) |
| BUG-02 | `actuators/actuator_net.py` | 두 번째 `with torch.inference_mode():` 블록 전체 삭제 |
| BUG-04 | `storage/distillation_rollout_storage.py` | `Tr` → `True` |

**검증**: 각 파일이 문법 에러 없이 import 되는지 확인 (`python -c "import ..."`)

---

### Phase 2: 모듈 인터페이스 버그 (BUG-03, 05)

**난이도**: ⭐⭐ | **예상 소요**: 15분 | **의존성**: 없음

Discriminator와 Student 모듈의 인터페이스 불일치 수정.

| 버그 | 파일 | 수정 내용 |
|------|------|-----------|
| BUG-03 | `modules/discriminator.py` | `DiscriminatorGRU`에 `extract_features()` 메서드 추가 + `forward()` 시그니처 통일 |
| BUG-05 | `modules/teacher_student.py` | `RMAStudent`에 `update_distribution()` 오버라이드 추가 (actor 2회 실행 방지) |

**검증**: 
- BUG-03: `DiscriminatorGRU`가 `extract_features(dummy_seq, dummy_obs)` 호출 시 정상 반환하는지 확인
- BUG-05: `RMAStudent.act()` 호출 시 `self.actor`가 1회만 실행되는지 확인 (forward hook으로 카운트)

---

### Phase 3: 수식/로직 버그 (BUG-06, 07)

**난이도**: ⭐⭐ | **예상 소요**: 10분 | **의존성**: 없음

수학 수식 오류와 scale 이중 적용 수정.

| 버그 | 파일 | 수정 내용 |
|------|------|-----------|
| BUG-06 | `domain_managers/domain_manager.py` | `cfg.kappa / max(...)` → `cfg.kappa / math.sqrt(max(...))` |
| BUG-07 | `feature_functions.py` × 6개 파일 | `error_foot_clearance`에서 첫 번째 `* scale` 제거 |

**⚠️ BUG-07 주의**: `feature_functions.py`가 6개 파일에 복사되어 있음. **모든 파일**에 동일하게 적용할 것:
1. `flamingo_env/flat_env/stand_drive/feature_functions.py`
2. `flamingo_env/flat_env/recovery/feature_functions.py`
3. `flamingo_env/rough_env/stand_drive/feature_functions.py`
4. `flamingo_4w4l_env/flat_env/stand_drive/feature_functions.py`
5. `flamingo_4w4l_env/rough_env/stand_drive/feature_functions.py`
6. `flamingo_light_env/flat_env/stand_drive/feature_functions.py`

**검증**: `grep -rn "error_foot_clearance" lab/flamingo/tasks/` 으로 모든 파일에서 `* scale`이 제거되었는지 확인

---

### Phase 4: 설계 개선 — Config 리팩토링 (BUG-08, 09)

**난이도**: ⭐⭐⭐ | **예상 소요**: 30분 | **의존성**: 없음

하드코딩된 값을 Config로 빼는 작업. 여러 파일에 걸쳐 수정 필요.

| 버그 | 수정 범위 |
|------|-----------|
| BUG-08 | `ADDCfg`에 `w_pot/w_disc/w_fm` 추가 → `MOOPPO`에서 전달 → `MOORewardManager`에서 사용 |
| BUG-09 | `DomainManager._capture_max_params()`를 자동 탐지 방식으로 변경 |

**검증**:
- BUG-08: 기존 하드코딩 값(`w_pot=0.0, w_disc=0.2, w_fm=0.8`)을 Config 기본값으로 설정하여 **기존 동작이 변하지 않는지** 확인
- BUG-09: 새로운 randomize term을 추가해도 자동으로 캡처되는지 확인

---

### Phase 5: 코드 구조 리팩토링 (BUG-10)

**난이도**: ⭐⭐⭐⭐ | **예상 소요**: 45분 | **의존성**: Phase 3 (BUG-07) 완료 후 수행 권장

`feature_functions.py` 6중 복사를 공통 모듈로 통합.

**작업 순서**:
1. `lab/flamingo/tasks/moo_based/locomotion/velocity/mdp/feature_functions_common.py` 생성
2. 공통 함수 추출: `apply_kernel`, `error_track_lin_vel_xy`, `error_track_ang_vel_z`, `error_base_height`, `error_flat_euler_rp`, `error_joint_deviation_huber`, `error_joint_deviation`, `action_rate`, `action_rate_huber`, `action_rate_l2`, `joint_acc_l2`, `dof_acc`, `ActionRatePenalty`, `TorqueRatePenalty`
3. 각 환경별 `feature_functions.py`에서 `from ...mdp.feature_functions_common import *` 추가
4. 중복 함수 삭제, 환경 고유 함수만 유지
5. 모든 환경에서 import 정상 동작하는지 확인

**검증**: 
- `python -c "from lab.flamingo.tasks.moo_based.locomotion.velocity.flamingo_env.flat_env.stand_drive.feature_functions import *"` (모든 환경에 대해)
- `diff`로 공통 함수가 정확히 동일한지 확인 후 추출

---

### Phase 6: 로봇 configuration 맞추기.

sim2sim framework /cosim-analyze 코드 확인.

모든 코드 확인하지 말고 flamingo_p_v3 관련된 코드만 확인.

목표 환경 : /cosim-analyze/envs/flamingo_p_v3/flamingo_p_v3.py

참고 config : /cosim-analyze/config/emv_table.yml


## 🚀 ONNX 배포 가이드

### 배포 파이프라인 개요

RMAStudent 배포 시 2개의 ONNX 모델이 필요:

```
[센서 데이터] → [전처리] → [History Buffer 관리]
                              ↓
              adaptation_encoder.onnx → z_hat (잠재 벡터)
                              ↓
              [obs + z_hat 결합] → policy.onnx → actions → [로봇 제어]
```

### ONNX Export 현황

현재 `empirical_normalization = False`로 설정되어 있으므로, normalizer는 `nn.Identity()`입니다. 따라서 normalizer 차원 불일치 문제는 발생하지 않으며, **별도의 normalizer export가 필요 없습니다.** 기존 export 코드 그대로 사용 가능합니다.

> ⚠️ 만약 향후 `empirical_normalization=True`로 전환할 경우, `_OnnxPolicyExporter`가 normalizer를 포함하는데 policy.onnx의 입력 차원은 `num_obs + latent_dim`이고 normalizer는 `num_obs`용이라 차원 불일치가 발생합니다. 이때는 Student 전용 exporter를 별도로 작성해야 합니다.

### 실제 로봇 배포 시 입력/출력 스펙

#### 1. `adaptation_encoder.onnx`

| 항목 | 값 |
|------|-----|
| 입력 이름 | `history_in` |
| 입력 shape | `(1, history_len, frame_dim)` = `(1, 50, frame_dim)` |
| 출력 이름 | `z_hat` |
| 출력 shape | `(1, latent_dim)` = `(1, 16)` 또는 `(1, 64)` |

`frame_dim` = `single_obs_dim + exterio_dim` (예: stack_policy의 1프레임 + none_stack_policy)

#### 2. `policy.onnx`

| 항목 | 값 |
|------|-----|
| 입력 이름 | `policy_input` (또는 `obs`) |
| 입력 shape | `(1, num_obs + latent_dim)` |
| 출력 이름 | `actions` |
| 출력 shape | `(1, num_actions)` |

`num_actions` = hip(2) + shoulder_leg(4) + wheel(2) = 8 (Flamingo 2w2l 기준)

#### 3. `obs_normalizer` — 현재 불필요

`empirical_normalization = False`이므로 normalizer export가 필요 없습니다. obs를 구성할 때 `ObservationsCfg`에 정의된 **scale 값만** 적용하면 됩니다.

### 로봇 측 추론 코드 (Python 의사코드)

```python
import onnxruntime as ort
import numpy as np
from collections import deque

# --- 초기화 ---
encoder_session = ort.InferenceSession("adaptation_encoder.onnx")
policy_session = ort.InferenceSession("policy.onnx")

# 차원 정보 (학습 환경의 ObservationsCfg에서 확인)
HISTORY_LEN = 50
SINGLE_OBS_DIM = ...   # stack_policy 1프레임 차원 (joint_pos + joint_vel + ang_vel + gravity + actions)
EXTERIO_DIM = ...       # none_stack_policy 차원 (velocity_commands 등)
FRAME_DIM = SINGLE_OBS_DIM + EXTERIO_DIM
NUM_OBS = ...           # 전체 policy obs 차원 (stacked)
LATENT_DIM = 16         # 또는 64 (Config에 따라)

# History Buffer 초기화
history_buffer = np.zeros((1, HISTORY_LEN, FRAME_DIM), dtype=np.float32)

# --- 매 제어 루프 (20Hz = sim_dt * decimation) ---
def inference_step(raw_sensor_data: dict) -> np.ndarray:
    """
    raw_sensor_data 구성:
      - joint_pos: (num_joints,) — 관절 위치 [rad]
      - joint_vel: (num_joints,) — 관절 속도 [rad/s]  
      - imu_ang_vel: (3,) — IMU 각속도 [rad/s]
      - imu_projected_gravity: (3,) — 투영된 중력 벡터
      - last_actions: (num_actions,) — 이전 스텝 액션
      - velocity_commands: (3,) — (vx_cmd, vy_cmd, wz_cmd)
    """
    
    # 1. Observation 벡터 구성 (scale만 적용, 정규화 없음)
    stack_obs_single = np.concatenate([
        raw_sensor_data["joint_pos"],                          # joint_pos
        raw_sensor_data["joint_vel"] * 0.15,                   # joint_vel (scale 적용!)
        raw_sensor_data["imu_ang_vel"] * 0.25,                 # base_ang_vel (scale!)
        raw_sensor_data["imu_projected_gravity"],              # projected_gravity
        raw_sensor_data["last_actions"],                       # last_action
    ])
    
    none_stack_obs = np.concatenate([
        raw_sensor_data["velocity_commands"] * np.array([2.0, 0.0, 0.25]),  # scaled commands
    ])
    
    # 2. 전체 obs 벡터 구성 (stacking 포함)
    full_obs = build_stacked_obs(stack_obs_single, none_stack_obs)  # (1, num_obs)
    
    # 3. History Buffer 업데이트 (FIFO)
    current_frame = np.concatenate([stack_obs_single, none_stack_obs])
    history_buffer[0, :-1, :] = history_buffer[0, 1:, :]  # shift left
    history_buffer[0, -1, :] = current_frame               # 최신 프레임 추가
    
    # 4. Adaptation Encoder 실행 → z_hat
    z_hat = encoder_session.run(
        ["z_hat"], 
        {"history_in": history_buffer}
    )[0]  # (1, latent_dim)
    
    # 5. Policy 실행 → actions
    policy_input = np.concatenate([full_obs, z_hat], axis=-1)  # (1, num_obs + latent_dim)
    actions = policy_session.run(
        ["actions"],
        {"policy_input": policy_input}
    )[0]  # (1, num_actions)
    
    return actions[0]  # (num_actions,)
```

### 차원 확인 체크리스트

배포 전 반드시 다음을 확인하세요:

```python
# 학습 로그에서 확인할 값들:
print(f"num_single_obs (stack_policy 1프레임): {env.num_single_obs}")
print(f"num_obs (전체 policy obs): {env.num_obs}")
print(f"num_actions: {env.num_actions}")
print(f"frame_dim: {actor_critic.frame_dim}")
print(f"history_len: {actor_critic.history_len}")
print(f"latent_dim: {actor_critic.encoder.net[-1].out_features}")
print(f"actor input dim: {actor_critic.actor[0].in_features}")
# → actor input dim == num_obs + latent_dim 이어야 함
```

### Observation 순서 (반드시 학습 환경과 일치해야 함)

**stack_policy** (매 프레임, noise 포함 학습):
1. `hip_shoulder_joint_pos` — (num_hip + num_shoulder,)
2. `leg_joint_pos` — (num_leg,) × gear_ratio
3. `hip_shoulder_joint_vel` — scale=0.15
4. `leg_joint_vel` — scale=0.15, × gear_ratio  
5. `wheel_joint_vel` — scale=0.15
6. `base_ang_vel` — scale=0.25
7. `base_projected_gravity`
8. `last_action` — (num_actions,)

**none_stack_policy** (현재 시점만):
1. `velocity_commands` — (vx×2.0, vy×0.0, wz×0.25)
2. (환경별 추가 항목)

---

## 빌드 & 실행 커맨드

```bash
# 학습 (Flat, Teacher)
python scripts/co_rl/train.py --task Isaac-Velocity-Flat-Flamingo-v3-moo-ppo --algo mooppo --num_envs 4096

# 학습 (Rough, Teacher)  
python scripts/co_rl/train.py --task Isaac-Velocity-Rough-Flamingo-v3-moo-ppo --algo mooppo --num_envs 4096

# 학습 (Rough, RMA Teacher)
python scripts/co_rl/train.py --task Isaac-Velocity-Rough-Flamingo-v3-moo-Teacher-ppo --algo mooppo --num_envs 4096

# 추론 (Play)
python scripts/co_rl/play.py --task Isaac-Velocity-Rough-Flamingo-Play-v3-moo-ppo --algo mooppo --num_envs 100

# ONNX Export는 play.py 실행 시 자동으로 exported/ 폴더에 생성됨
```

---

## 수정 작업 시 주의사항

1. **feature_functions.py 수정 시**: 동일 함수가 6개 파일에 복사되어 있으므로, 하나를 수정하면 나머지 5개도 동일하게 수정하거나, BUG-10의 리팩토링을 먼저 수행
2. **Config 수정 시**: `@configclass` 데코레이터가 사용됨. `dataclass`의 `MISSING` sentinel이 포함된 필드는 반드시 초기화 필요
3. **ONNX export 수정 시**: `opset_version=11`을 유지. Isaac Lab의 PyTorch 버전과 호환성 확인
4. **DomainManager 수정 시**: `EventManager`를 상속하므로 `_prepare_terms` 오버라이드가 매우 민감함. `EventTermCfg` 타입 체크를 반드시 유지
