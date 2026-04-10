# 🦩 IsaacLab Reward Autotuning: Flamingo Project

![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.0.0-silver.svg)
![IsaacLab](https://img.shields.io/badge/IsaacLab-1.1.0-green.svg)
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📖 Overview

**IsaacLab Reward Autotuning** is an advanced Reinforcement Learning (RL) framework built on NVIDIA Isaac Lab. This project primarily focuses on the robust control of **Flamingo (a wheeled-biped robot)** by solving the notoriously difficult problem of manual reward engineering. 

By integrating **Multi-Objective Optimization (MOO)** and **Adversarial Differential Discriminator (ADD)** concepts, this framework automatically tunes reward weights to resolve objective conflicts (e.g., Task completion vs. Style/Smoothness). This fundamentally suppresses high-frequency jittering and enables highly dynamic, yet stable locomotion in rough terrains without relying on exhaustive hand-crafted reward tuning.

## ✨ Key Features

- 🎯 **Multi-Objective Optimization (MOO) based Autotuning**: Dynamically balances conflicting objectives (task performance vs. energy efficiency/smoothness) finding the Pareto optimal policy.
- 📉 **Adversarial Differential Regularization (ADD)**: Evaluates high-order dynamics (velocity, acceleration, jerk) to drastically reduce mechanical jittering and ensure natural, physical realism.
- 🧠 **Robust Teacher-Student Architecture (RMA)**: Implements Rapid Motor Adaptation. The Teacher policy learns with privileged environmental information, which is then distilled into the Student policy for real-world deployment.
- 🏗️ **Extensible Task Management**: Seamlessly supports `manager_based`, `moo_based`, and `constraint_based` RL environments.

---

## 🛠️ Installation

This project requires a working installation of **NVIDIA Isaac Lab**.

### 1. Prerequisite: Isaac Lab Setup
Ensure you have successfully set up the `env_isaaclab` conda environment. If not, please refer to the [Isaac Lab Installation Guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

### 2. Clone & Install
Clone this repository and install the custom environments and tuning tools:

```bash
git clone https://github.com/JJUNE0/isaaclab_add_reward_autotuning.git
cd isaaclab_add_reward_autotuning

# Using standard pip
pip install -e .

# Or using uv for faster resolution
uv pip install -e .
```

### 3. Environment Variable Setup
To ensure the custom Flamingo environment is registered properly, append the project path to your PYTHONPATH:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/lab/flamingo/isaaclab
```

## 🚀 Quick Start
Below are the primary execution commands for training and evaluating the Flamingo robot. All scripts leverage wandb for logging.

### 1. Multi-Objective Optimization (MOO) - Rough Terrain
**Train the Policy:**
```bash
python scripts/co_rl/train.py \
    --task Isaac-Velocity-Rough-Flamingo-v3-moo-ppo \
    --num_envs 4096 \
    --headless \
    --algo mooppo \
    --num_policy_stacks 50 \
    --max_iteration 10000 \
    --logger wandb \
    --log_project_name moo_rough_asymmetry_tcn
```

**Play/Evaluate:**
```bash
python scripts/co_rl/play.py \
    --task Isaac-Velocity-Rough-Flamingo-Play-v3-moo-ppo \
    --num_envs 16 \
    --algo mooppo \
    --num_policy_stacks 50
```

### 2. Teacher-Student Learning (RMA Pipeline)
**Step A: Train the Teacher Policy (Privileged Info)**
```bash
python scripts/co_rl/train.py \
    --task Isaac-Velocity-Rough-Flamingo-v3-moo-Teacher-ppo \
    --num_envs 4096 \
    --headless \
    --algo mooppo \
    --num_policy_stacks 2 \
    --num_critic_stacks 2 \
    --max_iteration 10000 \
    --logger wandb \
    --log_project_name flamingo_teacher_moo_stair
```

**Step B: Train the Student Policy (Distillation)**
```bash
python scripts/co_rl/train.py \
    --task Isaac-Velocity-Rough-Flamingo-v3-moo-Student-ppo \
    --num_envs 4096 \
    --headless \
    --algo mooppo \
    --num_policy_stacks 2 \
    --num_critic_stacks 2 \
    --max_iteration 10000 \
    --logger wandb \
    --log_project_name flamingo_student_moo_base
```

**Step C: Evaluate the Student Policy**
```bash
python scripts/co_rl/play.py \
    --task Isaac-Velocity-Rough-Flamingo-Play-v3-moo-Student-ppo \
    --num_envs 16 \
    --algo mooppo \
    --num_policy_stacks 2 \
    --num_critic_stacks 2
```

## 📂 Repository Structure
```plaintext
isaaclab_add_reward_autotuning/
├── config/             # Configuration files for RL algorithms and logging
├── lab/
│   └── flamingo/       # Core Isaac Lab extensions
│       ├── assets/     # USD files and robot URDFs for Flamingo, A1, etc.
│       ├── sensors/    # Sensor configurations (IMU, RayCasters)
│       └── tasks/      # Task definitions (Manager-based, MOO-based, etc.)
├── scripts/
│   └── co_rl/          # Execution scripts for Constrained/MOO RL
│       ├── train.py    # Main training loop
│       └── play.py     # Inference and visualization script
└── pyproject.toml      # Python package dependencies and metadata
```

## 🤝 Acknowledgments
Developed and maintained for research regarding robust locomotion, reinforcement learning, and generative control models.
