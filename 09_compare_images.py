"""산출물 3: 복원 결과 비교 사진.
사진마다 '정답(HR) / bicubic / A / B / C' 를 같은 위치의 확대 조각으로 나란히 보여 준다.
조각 위치는 bicubic이 정답과 가장 많이 다른 곳(= 복원이 가장 어려운 곳)으로 자동 선택한다.
조각 아래 숫자는 그 조각만의 PSNR(Y 채널)이다. 모델은 SEED_SHOWN 시드의 것 하나만 보여 준다(표의 평균과는 다름).
출력: outputs/compare_images.png"""
import numpy as np                    # 배열 계산
import torch                          # PyTorch
import matplotlib                     # 그래프 라이브러리
matplotlib.use("Agg")                 # 창 없이 파일로만 저장하는 모드
import matplotlib.pyplot as plt       # 그림 그리기 함수
from PIL import Image                 # 사진 열기
from skimage.color import rgb2ycbcr   # RGB -> YCbCr
import config as cfg                  # 프로젝트 설정
from srcnn import SRCNN, bicubic_down_up, to_tensor, predict   # 모델과 공통 함수

PICK = [("Set14", "barbara"), ("Set14", "zebra"), ("Set5", "butterfly")]   # (세트, 파일 이름). 바꿔도 됨
SEED_SHOWN = 0      # 모델은 이 시드의 것을 사용
CROP = 64           # 조각 한 변 (HR 픽셀 단위)
LABELS = {"A": "A random", "B": "B detail", "C": "C flat"}   # 조건 -> 그림에 표시할 이름
device = "cuda" if torch.cuda.is_available() else "cpu"     # GPU 사용 가능 여부


def load_hr(path):                                   # 08_evaluate.py와 같은 방식
    """이미지 파일 -> (H, W, 3) uint8, H와 W를 4의 배수로 자른 정답 사진."""
    hr = np.asarray(Image.open(path).convert("RGB"))         # RGB로 읽어 numpy 배열로
    h, w = hr.shape[0] // 4 * 4, hr.shape[1] // 4 * 4        # 4의 배수로 내림
    return np.ascontiguousarray(hr[:h, :w])                  # 잘라서 연속 배열로


def luma(img):
    """(H, W, 3) uint8 RGB -> (H, W) float Y(밝기) 채널, 값 16~235."""
    return rgb2ycbcr(img)[..., 0]


def hardest_window(hr, inp, size=CROP, stride=16):
    """bicubic 오차(정답과의 제곱 차이 평균)가 가장 큰 size x size 조각의 왼쪽 위 좌표를 찾는다.
    입력: hr = 정답, inp = bicubic 결과 (둘 다 (H, W, 3) uint8), size = 조각 크기, stride = 창을 옮기는 간격
    출력: (x, y) 정수 튜플"""
    err = (luma(hr) - luma(inp)) ** 2                # 픽셀별 제곱 오차 지도 (H, W)
    best, pos = -1.0, (0, 0)                         # 지금까지의 최대 오차와 그 위치 (오차는 0 이상이라 -1로 시작하면 첫 창이 반드시 채택됨)
    for y in range(0, err.shape[0] - size + 1, stride):       # 세로로 16픽셀씩 옮기며
        for x in range(0, err.shape[1] - size + 1, stride):   # 가로로 16픽셀씩 옮기며
            e = err[y:y + size, x:x + size].mean()            # 이 창의 평균 오차
            if e > best:                                      # 최대값을 갱신하면
                best, pos = e, (x, y)                         # 기록
    return pos


models = {}                                          # 조건 -> 모델
for cond in "ABC":                                   # "A", "B", "C"
    path = cfg.MODEL_DIR / f"srcnn_{cond}_seed{SEED_SHOWN}.pt"   # 보여 줄 시드의 가중치 파일
    m = SRCNN().to(device)                           # 빈 모델 생성
    m.load_state_dict(torch.load(path, map_location=device))     # 가중치 채우기 (파일이 없으면 여기서 오류)
    models[cond] = m.eval()                          # 평가 모드로 저장

fig, axes = plt.subplots(len(PICK), 5, figsize=(13, 2.8 * len(PICK)))   # 사진 수(3)행 x 5열 (HR, bicubic, A, B, C)
for r, (set_name, name) in enumerate(PICK):          # r = 행 번호, (세트, 파일 이름)
    folder = cfg.ROOT / "data" / set_name            # 세트 폴더
    hr = load_hr(folder / f"{name}.png")             # 정답 사진
    inp = bicubic_down_up(hr)                        # bicubic 결과 = 모델 입력
    x = to_tensor(inp[None]).to(device)              # (1, 3, H, W) float 0~1
    imgs = {"HR (answer)": hr, "bicubic": inp}       # 열 이름 -> 사진 (사전은 넣은 순서를 유지하므로 이 순서대로 열이 됨)
    for cond, m in models.items():                   # 모델마다
        out = predict(m, x)[0].permute(1, 2, 0).cpu().numpy()   # 추론 결과 (H, W, 3) float 0~1
        imgs[LABELS[cond]] = np.round(out * 255).astype(np.uint8)   # 08과 같은 8비트 반올림

    x0, y0 = hardest_window(hr, inp)                 # 확대해서 보여 줄 위치
    hr_crop = hr[y0:y0 + CROP, x0:x0 + CROP]         # 정답 조각 (PSNR 계산 기준)
    for c, (label, img) in enumerate(imgs.items()):  # c = 열 번호
        crop = img[y0:y0 + CROP, x0:x0 + CROP]       # 같은 위치 조각
        ax = axes[r, c]                              # 그릴 칸
        ax.imshow(crop, interpolation="nearest")     # 확대해도 픽셀이 뭉개지지 않게
        ax.axis("off")                               # 눈금 숨김
        if c == 0:                                   # 첫 열(정답)은 제목만
            ax.set_title(f"{name} - HR (answer)", fontsize=9)
        else:                                        # 나머지 열은 조각 PSNR 표시 (가장자리 제외 없음, 08의 표와 다른 값)
            mse = ((luma(crop) - luma(hr_crop)) ** 2).mean()   # 조각의 Y 채널 mse
            ax.set_title(f"{label}  {10 * np.log10(255 ** 2 / max(mse, 1e-10)):.2f} dB", fontsize=9)
plt.tight_layout()                                   # 여백 자동 조정
plt.savefig(cfg.OUT_DIR / "compare_images.png", dpi=150)
print("저장:", cfg.OUT_DIR / "compare_images.png")
