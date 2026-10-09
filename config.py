"""모든 설정값은 여기 한 곳에서만 바꾼다. 나머지 스크립트는 이 파일을 읽기만 한다.
다른 파일은 `import config as cfg`로 이 파일을 불러와 `cfg.N`처럼 값을 꺼내 쓴다."""
from pathlib import Path   # 경로를 문자열 대신 객체로 다루는 표준 라이브러리 클래스 (/ 연산자로 경로를 이어 붙일 수 있음)

ROOT = Path(__file__).parent                    # 이 파일(config.py)이 있는 폴더 = 프로젝트 루트. 어디서 실행해도 같은 경로가 됨
DIV2K_DIR = ROOT / "data" / "DIV2K_train_HR"   # HR 학습 이미지 폴더 (0001.png ~ 0800.png)
OUT_DIR = ROOT / "outputs"                      # 모든 결과물(패치, 점수, 그룹, 모델, 그림)을 저장하는 폴더

PATCH_SIZE = 96          # 패치 한 변의 픽셀 수. 반드시 4의 배수 (4배 축소했을 때 정수 크기가 되어야 함) -> README에 기록
PATCHES_PER_IMAGE = 10   # 이미지 1장에서 자르는 패치 수. 800장 x 10 = 8000개 풀
MAX_IMAGES = None        # None이면 전부 사용. 빠른 테스트는 20 같은 숫자로 (01에서 앞쪽 이미지 몇 장만 쓰게 됨)
POOL_SEED = 0            # 패치 풀을 자를 때의 시드. 풀은 고정하고, 시드마다 달라지는 건 A/B/C 선택뿐

SEEDS = [0, 1, 2]        # 탐색용 시드 3개 (03의 그룹 뽑기와 07의 학습 무작위성을 함께 결정). 새 분할 기준은 먼저 이 3개로 본다
FINAL_SEEDS = list(range(10))   # 최종 확인용 시드 10개 [0, 1, ..., 9]. 최종 후보에만 쓴다 (앞 3개는 SEEDS와 같으므로 기존 결과를 그대로 포함)
                                # 03/08/10에 'final'을 붙여 실행하면 이 목록을 쓴다. 결과가 애매하다고 이 수를 늘리지 않는다(사전에 고정)
N = 1000                 # 조건별 패치 수 (A, B, C 모두 동일)

assert PATCH_SIZE % 4 == 0, "PATCH_SIZE는 4의 배수여야 합니다"   # 조건이 거짓이면 이 메시지와 함께 즉시 멈춤 (% 는 나머지 연산)
OUT_DIR.mkdir(exist_ok=True)   # outputs 폴더를 만든다. exist_ok=True: 이미 있으면 오류 없이 넘어감

# ---- 단계 ③ 학습 설정: A, B, C 모든 조건에서 반드시 동일하게 둔다 ----
EPOCHS = 100         # 학습 횟수(고정). 평가용 사진(Set5/Set14)을 보고 정하지 않는다. 1 에폭 = 패치 N개를 한 번씩 다 보는 것
BATCH_SIZE = 16      # 한 번의 가중치 갱신에 쓰는 패치 수
LR = 1e-4            # Adam 학습률 (1e-4 = 0.0001). 한 번 갱신할 때 가중치를 얼마나 크게 움직일지
AUGMENT = True       # 학습 중 90도 회전·좌우반전을 무작위로 적용 (모든 조건 동일)
MODEL_DIR = OUT_DIR / "models"   # (옛 위치) 경로 분리 전 기준 모델 9개가 있던 폴더. 지금은 09_compare_images.py만 여기를 읽음
MODEL_DIR.mkdir(exist_ok=True)   # models 폴더를 만든다 (이미 있으면 넘어감)

# ---- 실험별 저장 경로 (기준 결과를 덮어쓰지 않기 위해 실험마다 폴더를 나눈다) ----
# 풀(patches.npy)과 점수(scores.npy)는 모든 실험이 공유하므로 outputs/ 바로 아래에 그대로 둔다.
# 실험마다 달라지는 것(그룹, 모델, 평가 결과, 로그)만 outputs/exp/<실험이름>/ 아래에 저장한다.
EXP_ROOT = OUT_DIR / "exp"       # 모든 실험 폴더의 부모 폴더
BASELINE = "baseline"            # 기준 실험(중앙값 분할)의 이름. 다른 실험의 A는 이 실험의 A 모델을 재사용한다


def exp_dir(name):
    """실험 이름 -> 그 실험의 저장 폴더 경로(Path). 폴더를 만들지는 않는다(만드는 건 03의 몫).
    입력: name = 실험 이름 문자열 (영문, 숫자, _, - 만 허용. 예: "baseline", "q25_split")
    출력: outputs/exp/<name> 경로
    이름 규칙을 두는 이유: 공백·한글·슬래시가 들어가면 경로가 꼬이거나 다른 폴더를 가리킬 수 있어서."""
    assert name and all(c.isascii() and (c.isalnum() or c in "_-") for c in name), \
        f"실험 이름은 영문/숫자/_/- 만 쓰세요: {name!r}"   # 빈 문자열이거나 허용하지 않는 글자가 있으면 멈춤 (\ 는 다음 줄로 이어 쓰기)
    return EXP_ROOT / name       # 예: outputs/exp/baseline
