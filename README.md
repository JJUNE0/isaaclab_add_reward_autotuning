# IsaacLab Reward Autotuning: Flamingo Project

This project is a framework for **Reward Function Autotuning** and **Reinforcement Learning** (RL) based on NVIDIA Isaac Lab. It supports various task configurations such as `manager_based`, `moo_based`, and `constraint_based`, and includes tools for Constrained Reinforcement Learning (CO-RL).

## 🛠️ Installation

This project requires an existing installation of Isaac Lab.

1. **Install Isaac Lab**: Follow the [Isaac Lab Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) to set up your `env_isaaclab` conda environment.
2. **Clone the Repository & Install Package**:
   ```bash
   git clone https://github.com/JJUNE0/isaaclab_add_reward_autotuning.git
   cd isaaclab_add_reward_autotuning
   pip install -e .
   # Or using uv
   uv pip install -e .
   ```
3. **Environment Setup (PYTHONPATH)**:
   Add the local IsaacLab extensions to your python path:
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)/lab/flamingo/isaaclab
   ```

## 🚀 Usage (Execution Commands)

Below are the primary execution commands based on the `launch.json` configurations.

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
- `lab/flamingo`: Isaac Lab extension (Robot assets, sensor configs, and task definitions)
- `scripts/co_rl`: CO-RL algorithm implementations and execution scripts
- `pyproject.toml`: Project package configuration and dependency management

## 📝 Key Features
- **Reward Autotuning**: Based on Multi-Objective Optimization (MOO)
- **RMA Support**: Robust Teacher-Student learning architecture
- **Robot Assets**: Configurations for Flamingo, A1, Humanoid, and more
- **Task Management**: Supports Manager-based, MOO-based, and Constraint-based envs
