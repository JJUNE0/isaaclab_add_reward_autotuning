from dataclasses import MISSING
from isaaclab.utils import configclass
from scripts.co_rl.core.wrapper import CoRlPpoActorCriticCfg
from .co_rl_cfg import FlamingoMOOPPORunnerCfg

# ---------------------------------------------------------
# [RMA Teacher] 설정 클래스
# ---------------------------------------------------------
@configclass
class RMATeacherActorCriticCfg(CoRlPpoActorCriticCfg):
    """Configuration for RMA Teacher Policy."""
    class_name: str = "RMATeacher" # eval()에서 이 클래스를 로드함
    latent_dim: int = 64           # 환경 인코더가 압축할 차원

    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"
    
# ---------------------------------------------------------
# [RMA Student] 설정 클래스
# ---------------------------------------------------------
@configclass
class RMAStudentActorCriticCfg(RMATeacherActorCriticCfg):
    """Configuration for RMA Student Policy."""
    class_name: str = "RMAStudent"
    teacher_checkpoint_path: str = MISSING
    history_len: int = 50         # 관측 이력 길이
    
@configclass
class FlamingoRMATeacherRunnerCfg(FlamingoMOOPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.max_iterations = 5000
        self.experiment_name = "Flamingo_RMA_Teacher"
        
        # [핵심] Teacher 모드 활성화
        self.use_rma_teacher = True
        self.use_rma_student = False

        # [정책 교체] RMATeacherActorCriticCfg 사용
        self.policy = RMATeacherActorCriticCfg()
        
@configclass
class FlamingoRMAStudentRunnerCfg(FlamingoMOOPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.max_iterations = 5000
        self.experiment_name = "Flamingo_RMA_Student"
        
        self.use_rma_teacher = False
        self.use_rma_student = True
        
        self.policy = RMAStudentActorCriticCfg(
            teacher_checkpoint_path="logs/co_rl/Flamingo_RMA_Teacher/mooppo/2026-01-26_11-42-03/model_8100.pt"
        )