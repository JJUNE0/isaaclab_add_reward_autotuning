# IsaacLab Reward Autotuning: Flamingo Project

이 프로젝트는 NVIDIA Isaac Lab을 기반으로 로봇의 보상 함수 자동 튜닝 및 강화 학습을 수행하기 위한 프레임워크입니다.

## 🛠️ Installation

본 프로젝트는 Isaac Lab 환경에서 동작합니다. 먼저 Isaac Lab이 설치되어 있어야 합니다.

1. **Isaac Lab 설치**: [Isaac Lab 공식 설치 가이드](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)를 따라 `env_isaaclab` 콘다 환경을 구축하세요.
2. **프로젝트 복제 및 패키지 설치**:
   ```bash
   git clone <YOUR_GITHUB_REPO_URL>
   cd isaaclab_add_reward_autotuning
   pip install -e .
   # 또는 uv 사용 시
   uv pip install -e .
   ```
3. **환경 설정 (PYTHONPATH)**:
   로컬 IsaacLab 익스텐션을 인식하기 위해 경로를 추가합니다.
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)/lab/flamingo/isaaclab
   ```

## 🚀 Usage (Execution Commands)

`launch.json`에 설정된 주요 실행 명령어들입니다. 상황에 맞춰 터미널에서 실행하세요.

### 1. Multi-Objective Optimization (MOO) - Rough Terrain
*   **Train**:
    ```bash
    python scripts/co_rl/train.py --task Isaac-Velocity-Rough-Flamingo-v3-moo-ppo --num_envs 4096 --headless --algo mooppo --num_policy_stacks 50 --max_iteration 10000 --logger wandb --log_project_name moo_rough_asymmetry_tcn
    ```
*   **Play**:
    ```bash
    python scripts/co_rl/play.py --task Isaac-Velocity-Rough-Flamingo-Play-v3-moo-ppo --num_envs 16 --algo mooppo --num_policy_stacks 50
    ```

### 2. Teacher-Student Learning (RMA)
*   **Teacher Train**:
    ```bash
    python scripts/co_rl/train.py --task Isaac-Velocity-Rough-Flamingo-v3-moo-Teacher-ppo --num_envs 4096 --headless --algo mooppo --num_policy_stacks 2 --num_critic_stacks 2 --max_iteration 10000 --logger wandb --log_project_name flamingo_rev_03_2_3_moo_stair
    ```
*   **Student Train**:
    ```bash
    python scripts/co_rl/train.py --task Isaac-Velocity-Rough-Flamingo-v3-moo-Student-ppo --num_envs 4096 --headless --algo mooppo --num_policy_stacks 2 --num_critic_stacks 2 --max_iteration 10000 --logger wandb --log_project_name student_moo_rough_base-velocity
    ```
*   **Student Play**:
    ```bash
    python scripts/co_rl/play.py --task Isaac-Velocity-Rough-Flamingo-Play-v3-moo-Student-ppo --num_envs 16 --algo mooppo --num_policy_stacks 2 --num_critic_stacks 2
    ```

## 📂 Project Structure
- `lab/flamingo`: Isaac Lab 익스텐션 (로봇 자산, 태스크 정의)
- `scripts/co_rl`: CO-RL 알고리즘 및 학습 실행 스크립트
- `pyproject.toml`: 프로젝트 패키지 설정 및 의존성 관리

## 📝 Features
- Multi-Objective Optimization (MOO) 기반 보상 자동 튜닝
- Teacher-Student 학습 구조 (RMA) 지원
- 다양한 로봇 설정 (Flamingo, A1, Humanoid 등) 지원
