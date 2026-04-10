# visualize_zarr.py

import argparse
import zarr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os


# ==============================================================================
# ▼▼▼ 1. 사용자 설정: Observation 데이터의 각 항목이 차지하는 인덱스 ▼▼▼
OBS_MAP = {
    "joint_pos": slice(0, 6),
    "joint_vel": slice(6, 14),
    "base_angvel": slice(14, 17),
    "base_projected_gravity": slice(17, 20),
    "target_joint_pos": slice(20, 26),
    "target_wheel_vel": slice(26, 28),
    "linear_velocity_command": slice(140, 144)
}


PLOT_CONFIG = [
    {
        'title': 'Joint Position vs Target ',
        'lines': [
            {'key': 'joint_pos', 'label': 'Actual', 'style': '-'},
            {'key': 'target_joint_pos', 'label': 'Target', 'style': '--'},
        ]
    },
    {
        'title': 'Joint Velocity vs Target ',
        'lines': [
            {'key': 'joint_vel', 'label': 'Actual', 'style': '-'},
            
            {'key': 'target_wheel_vel', 'label': 'Target Wheel', 'style': '--'},
        ]
    },

    {
        'title': 'Base Angular Velocity',
        'lines': [{'key': 'base_angvel', 'label': 'Actual', 'style': '-'}]
    },
    {
        'title': 'Base Projected Gravity',
        'lines': [{'key': 'base_projected_gravity', 'label': 'Actual', 'style': '-'}]
    },
    {
        'title': 'Linear Velocity Command',
        'lines': [{'key': 'linear_velocity_command', 'label': 'Command', 'style': '-'}]
    },
    {
        'title': 'Reward',
        'lines': [{'key': 'reward', 'label': 'Reward', 'style': '-'}]
    }
]
# ==============================================================================
num_pos_dims = OBS_MAP['joint_pos'].stop - OBS_MAP['joint_pos'].start
for i in range(num_pos_dims):
    PLOT_CONFIG.append({
        'title': f'Joint Position Dim {i}',
        'lines': [
            {'key': 'joint_pos', 'dim': i, 'label': 'Actual', 'style': '-'},
            {'key': 'target_joint_pos', 'dim': i, 'label': 'Target', 'style': '--'},
        ]
    })

# --- Wheel Velocity 플롯 (차원별로 생성) ---
# joint_vel의 마지막 2개 차원이 wheel_vel이라고 가정합니다.
num_wheel_dims = OBS_MAP['target_wheel_vel'].stop - OBS_MAP['target_wheel_vel'].start
num_joint_vel_dims = OBS_MAP['joint_vel'].stop - OBS_MAP['joint_vel'].start
for i in range(num_wheel_dims):
    # joint_vel에서 바퀴에 해당하는 차원을 계산 (예: 8개 중 마지막 2개)
    joint_vel_wheel_dim = num_joint_vel_dims - num_wheel_dims + i
    PLOT_CONFIG.append({
        'title': f'Wheel Velocity Dim {i}',
        'lines': [
            {'key': 'joint_vel', 'dim': joint_vel_wheel_dim, 'label': 'Actual', 'style': '-'},
            {'key': 'target_wheel_vel', 'dim': i, 'label': 'Target', 'style': '--'},
        ]
    })

# --- 나머지 단일 플롯들 추가 ---
single_plots = ["base_angvel", "base_projected_gravity", "linear_velocity_command"]
for key in single_plots:
    if key in OBS_MAP:
        PLOT_CONFIG.append({
            'title': key.replace('_', ' ').title(),
            'lines': [{'key': key, 'dim': None, 'label': 'Value', 'style': '-'}]
        })



def find_episode_boundaries(dones, timeouts):
    # (이전과 동일)
    boundaries = []
    episode_end_flags = np.logical_or(dones, timeouts)
    end_indices = np.where(episode_end_flags)[0]
    start_idx = 0
    for end_idx in end_indices:
        boundaries.append((start_idx, end_idx + 1))
        start_idx = end_idx + 1
    if start_idx < len(dones):
        boundaries.append((start_idx, len(dones)))
    return boundaries


def create_episode_figure(episode_data, episode_num, total_episodes):
    """하나의 에피소드에 대한 Matplotlib Figure 객체를 생성하고 반환합니다."""
    # (plot_episode 함수와 거의 동일하나, plt.show() 대신 fig 객체를 반환)
    obs_data = episode_data['observations']
    rew_data = episode_data['rewards']
    num_timesteps = len(obs_data)
    timesteps = np.arange(num_timesteps)

    num_subplots = len(PLOT_CONFIG)
    fig, axes = plt.subplots(
        num_subplots, 1, figsize=(14, 3 * num_subplots), sharex=True
    )
    if num_subplots == 1:
        axes = [axes]
    
    fig.suptitle(f"Episode {episode_num + 1} / {total_episodes} (Timesteps: {num_timesteps})", fontsize=16)

    # ... (이전 plot_episode의 그리기 로직과 동일) ...
    for i, plot_info in enumerate(PLOT_CONFIG):
        ax = axes[i]
        ax.set_title(plot_info['title'])
        ax.set_ylabel("Value")
        ax.grid(True)
        for line_info in plot_info['lines']:
            key = line_info['key']
            dim_index = line_info.get('dim')
            
            if key == 'reward':
                if rew_data    is not None:
                    data_to_plot = rew_data
                else:
                    print(f"경고: episode_data에 'rewards' 키가 없습니다. Reward 플롯을 생략합니다.")
                    continue
            else: # reward가 아닌 다른 모든 키는 OBS_MAP을 사용합니다.
                if key in OBS_MAP:
                    obs_slice = OBS_MAP[key]
                    data_to_plot = obs_data[:, obs_slice]
                else:
                    print(f"경고: OBS_MAP에 '{key}' 키가 없습니다. 이 라인은 생략됩니다.")
                    continue
            
            if dim_index is None:
                for d in range(data_to_plot.shape[1]):
                    ax.plot(timesteps, data_to_plot[:, d], linestyle=line_info['style'], label=f"{line_info['label']}_{d}")
            else:
                data_to_plot = data_to_plot[:, dim_index]
                ax.plot(timesteps, data_to_plot, linestyle=line_info['style'], label=line_info['label'])
        ax.legend()

    axes[-1].set_xlabel("Time Step")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    return fig


def main(args):
    """메인 함수: Zarr 파일을 로드하고 선택된 에피소드로 PDF 리포트를 생성합니다."""
    print(f"Zarr 파일 로드 중: {args.zarr_path}")
    store = zarr.open(args.zarr_path, 'r')
    observations = store['observations'][:]
    rewards = store['rewards'][:] 
    dones = store['dones'][:, 0]
    timeouts = store['timeouts'][:, 0]

    episode_boundaries = find_episode_boundaries(dones, timeouts)
    total_episodes = len(episode_boundaries)
    print(f"총 {total_episodes}개의 에피소드를 찾았습니다.")

    if not episode_boundaries:
        print("데이터가 없습니다.")
        return
        
    # --- ▼▼▼ 시각화할 에피소드 인덱스 결정 로직 ▼▼▼ ---
    target_episode_indices = []
    if args.episodes:
        # --episodes 인자로 특정 에피소드 번호를 직접 지정한 경우
        for ep_num in args.episodes:
            if 1 <= ep_num <= total_episodes:
                target_episode_indices.append(ep_num - 1) # 0-based index로 변환
            else:
                print(f"[Warning] 에피소드 번호 {ep_num}는 유효한 범위(1~{total_episodes})를 벗어납니다. 무시합니다.")
    elif args.interval:
        # --interval 인자로 일정한 간격의 에피소드를 지정한 경우
        target_episode_indices = list(range(args.interval - 1, total_episodes, args.interval))
    else:
        # 인자가 없으면 모든 에피소드를 대상으로 함 (이전과 동일)
        target_episode_indices = list(range(total_episodes))
    
    if not target_episode_indices:
        print("시각화하도록 선택된 에피소드가 없습니다.")
        return
        
    print(f"시각화 대상 에피소드 (번호): {[i + 1 for i in target_episode_indices]}")
    # --- ▲▲▲ 로직 변경 끝 ▲▲▲ ---

    os.makedirs(args.output_dir, exist_ok=True)
    pdf_path = os.path.join(args.output_dir, args.pdf_filename)
    
    print(f"\nPDF 리포트 생성 중: {pdf_path}")
    
    with PdfPages(pdf_path) as pdf:
        # 선택된 에피소드들만 순회
        for episode_idx in target_episode_indices:
            start, end = episode_boundaries[episode_idx]
            episode_data = {'observations': observations[start:end],
                            'rewards': rewards[start:end],}
            
            print(f"  - Episode {episode_idx + 1} 그래프 생성...")
            
            fig = create_episode_figure(episode_data, episode_idx, total_episodes)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\n리포트 생성이 완료되었습니다. '{args.output_dir}' 폴더를 확인하세요.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zarr 데이터로 PDF 리포트를 생성합니다.")
    parser.add_argument("--zarr_path", type=str, required=True, help="분석할 Zarr 파일의 경로")
    parser.add_argument("--output_dir", type=str, default="reports", help="PDF 리포트가 저장될 폴더")
    parser.add_argument("--pdf_filename", type=str, default="report.pdf", help="저장될 PDF 파일의 이름")
    
    # --- ▼▼▼ 인자(Argument) 수정 ▼▼▼ ---
    # 기존 batch_size 대신 episodes와 interval 인자 추가
    parser.add_argument(
        "--episodes", type=int, nargs='+', default=None, help="시각화할 특정 에피소드 번호를 스페이스로 구분하여 입력 (예: --episodes 50 100 150)"
    )
    parser.add_argument(
        "--interval", type=int, default=None, help="일정한 간격으로 에피소드를 샘플링하여 시각화 (예: --interval 50 이면 50, 100, 150...번째 에피소드)"
    )
    
    args = parser.parse_args()
    main(args)