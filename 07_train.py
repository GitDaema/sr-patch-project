"""단계 ③: 조건(A/B/C)과 시드, 실험 이름을 정해 SRCNN을 한 번 학습한다.
사용법:  python 07_train.py B 0 <실험이름>      (예: python 07_train.py B 0 baseline). 시드는 0~9 (3~9는 최종 확인용)
기준 실험은 조건 3개 x 시드 3개 = 9번, 새 실험은 A를 기준 모델로 재사용하므로 B, C x 시드 3개 = 6번 돌린다.
입력: outputs/patches.npy, outputs/scores.npy, outputs/exp/<실험이름>/groups_seed{시드}.npz
출력: outputs/exp/<실험이름>/models/srcnn_{조건}_seed{시드}.pt (학습된 가중치)"""
import sys, time         # sys: 명령줄 인자(sys.argv) 읽기 / time: 경과 시간 측정
import numpy as np       # 배열 계산
import torch             # PyTorch
import config as cfg     # 프로젝트 설정 (EPOCHS, LR 등)
from srcnn import SRCNN, bicubic_down_up, to_tensor, predict   # 모델과 공통 함수

assert len(sys.argv) == 4, "사용법: python 07_train.py [A|B|C] [시드] <실험이름>"   # 인자 개수 확인 (파일 이름 포함 4개)
cond, seed, exp = sys.argv[1], int(sys.argv[2]), sys.argv[3]   # sys.argv = ["07_train.py", "B", "0", "baseline"] -> 조건 "B", 시드 정수 0, 실험 이름
assert cond in ("A", "B", "C") and seed in cfg.FINAL_SEEDS, "사용법: python 07_train.py [A|B|C] [시드 0~9] <실험이름>"   # 잘못된 인자면 멈춤. 시드 3~9는 최종 확인용 (03을 final로 먼저 실행해야 그룹 파일이 있음)
assert cond != "A" or exp == cfg.BASELINE, "새 실험의 A는 기준 실험의 A 모델을 재사용합니다. A는 학습하지 마세요"   # 'A가 아니거나, 기준 실험이거나' 둘 중 하나는 참이어야 통과

exp_path = cfg.exp_dir(exp)                                 # 이 실험의 폴더 outputs/exp/<exp>
groups_path = exp_path / f"groups_seed{seed}.npz"           # 03이 만든 그룹 파일
assert groups_path.exists(), f"{groups_path} 가 없습니다. 03_make_groups.py {exp} 를 먼저 실행하세요"
model_dir = exp_path / "models"                             # 이 실험의 모델 저장 폴더
model_dir.mkdir(exist_ok=True)                              # 없으면 만듦
path = model_dir / f"srcnn_{cond}_seed{seed}.pt"            # 저장할 가중치 파일 경로
assert not path.exists(), f"{path} 가 이미 있습니다. 덮어쓰지 않도록 멈춥니다 (다시 학습하려면 파일을 직접 지우세요)"   # 학습(약 1분) 시작 전에 미리 확인

torch.manual_seed(seed)   # PyTorch 난수 고정 (CPU·GPU 모두): 배치 순서, 증강 선택에 영향. 마지막 층 외 가중치 초기값도 여기서 결정
np.random.seed(seed)      # numpy 전역 난수 고정. 현재 이 파일에서 np.random 전역 난수를 쓰는 곳은 없어 실질적 효과는 없음(예방용)
device = "cuda" if torch.cuda.is_available() else "cpu"   # GPU를 쓸 수 있으면 "cuda", 아니면 "cpu" (한 줄짜리 if-else)
print(f"실험 {exp} | 조건 {cond} | 시드 {seed} | 장치 {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

# ---- 데이터: 03에서 뽑은 패치 N개 -> (입력, 정답) 쌍 ----
patches = np.load(cfg.OUT_DIR / "patches.npy")    # (8000, 96, 96, 3) uint8
scores = np.load(cfg.OUT_DIR / "scores.npy")      # (8000,) float64 (점검 출력용)
idx = np.load(groups_path)[cond]                  # 이 실험·시드·조건의 패치 번호 (N,) 정수. npz에서 이름(A/B/C)으로 꺼냄

hr = patches[idx]                                       # 정답: 원본 패치. 번호 배열로 한꺼번에 고르기 -> (N, 96, 96, 3) uint8
lr_up = np.stack([bicubic_down_up(p) for p in hr])      # 입력: 4배 축소 후 bicubic 확대. (N, 96, 96, 3) uint8
Y, X = to_tensor(hr).to(device), to_tensor(lr_up).to(device)   # 둘 다 (N, 3, 96, 96) float 0~1로 바꿔 GPU에 올림. Y = 정답, X = 입력
N = len(Y)                                              # 학습 패치 수 (= cfg.N)


def per_patch_psnr(a, b):
    """패치마다 PSNR을 구한다. 02의 psnr_bic과 같은 공식(255 기준, mse 하한 1e-10).
    입력: a, b = (N, 3, H, W) float 텐서, 값 0~1
    출력: (N,) float64 텐서, 단위 dB"""
    mse255 = ((a.double() - b.double()) ** 2).mean(dim=(1, 2, 3)) * 255 ** 2   # double = 64비트 실수. 채널·세로·가로 축 평균 -> 패치별 mse. 0~1 단위라 255^2를 곱해 0~255 단위로
    return 10 * torch.log10(255 ** 2 / mse255.clamp_min(1e-10))              # clamp_min: 1e-10보다 작은 값은 1e-10으로 (0으로 나누기 방지)


def report(tag, pred):
    """학습 패치 기준 PSNR의 평균과 중앙값을 출력한다 (진단용, 성능 근거 아님).
    입력: tag = 출력할 이름표, pred = (N, 3, H, W) 예측 텐서. 정답은 전역 변수 Y를 사용. 출력: 없음"""
    p = per_patch_psnr(pred, Y)   # (N,) 패치별 PSNR
    print(f"{tag}: 평균 {p.mean().item():.2f} dB | 중앙값 {p.median().item():.2f} dB")   # .item(): 원소 1개짜리 텐서 -> 파이썬 숫자


report("학습 전(= bicubic 입력)", X)
print(f"   -> 02/03에서 본 이 그룹의 평균 PSNR_bic: {scores[idx].mean():.2f} dB (위 평균과 같아야 정상)")

# ---- 학습 ----
model = SRCNN().to(device)                                  # 모델 생성 후 GPU로
opt = torch.optim.Adam(model.parameters(), lr=cfg.LR)       # Adam 최적화기: 모델의 모든 가중치를 학습률 LR로 갱신
loss_fn = torch.nn.MSELoss()                                # 손실 = 출력과 정답의 평균 제곱 오차

t0 = time.time()                                            # 시작 시각 (초)
for epoch in range(1, cfg.EPOCHS + 1):                      # 1 ~ 100 에폭
    model.train()                                           # 학습 모드로 전환 (이 모델엔 드롭아웃 등이 없어 실제 차이는 없지만 관례)
    perm = torch.randperm(N, device=device)                 # 0~N-1을 무작위로 섞은 순서 -> 매 에폭 배치 구성이 달라짐
    total = 0.0                                             # 이번 에폭의 손실 합 (출력용)
    for i in range(0, N, cfg.BATCH_SIZE):                   # 0, 16, 32, ... 위치에서 배치 시작
        b = perm[i:i + cfg.BATCH_SIZE]                      # 이번 배치에 들어갈 패치 번호 (마지막 배치는 16개보다 적을 수 있음. 1000 = 16x62 + 8)
        x, y = X[b], Y[b]                                   # 입력·정답 배치 (B, 3, 96, 96)
        if cfg.AUGMENT:                                 # 입력과 정답에 같은 변환을 적용
            k = int(torch.randint(0, 4, (1,)))              # 0~3 중 하나 -> 90도 x k 회전. 배치 전체에 같은 k 적용
            x, y = torch.rot90(x, k, (2, 3)), torch.rot90(y, k, (2, 3))   # 축 2(세로), 3(가로) 평면에서 회전
            if torch.rand(1).item() < 0.5:                  # 50% 확률로
                x, y = x.flip(3), y.flip(3)                 # 가로 방향(축 3) 뒤집기 = 좌우반전
        loss = loss_fn(model(x), y)                         # 예측과 정답의 MSE
        opt.zero_grad()                                     # 이전 배치에서 쌓인 기울기를 0으로 초기화 (안 하면 누적됨)
        loss.backward()                                     # 역전파: 손실을 각 가중치로 미분한 기울기 계산
        opt.step()                                          # 기울기를 이용해 가중치 한 번 갱신
        total += loss.item() * len(b)                       # 배치 평균 손실 x 배치 크기 = 배치 손실 합
    if epoch == 1 or epoch % 10 == 0:                       # 1에폭과 10의 배수 에폭에서만 출력
        print(f"epoch {epoch:3d}/{cfg.EPOCHS}  학습 MSE {total / N:.6f}  ({time.time() - t0:.0f}초)")

# ---- 결과 ----
model.eval()                                                # 평가 모드로 전환
report("학습 후 (학습에 쓴 패치 기준)", predict(model, X))
print("   -> 단색 패치는 평균을 크게 흔들 수 있어 중앙값도 같이 봅니다. '학습 후 중앙값 > 학습 전 중앙값'이면 정상.")
torch.save(model.state_dict(), path)                        # state_dict = 층 이름 -> 가중치 텐서 사전. 구조는 저장하지 않으므로 불러올 때 SRCNN()을 먼저 만듦
print("저장:", path)
