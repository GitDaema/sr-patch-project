"""단계 ④-4: 최종 확인용 시각 비교 그림. 같은 사진의 같은 위치를 확대해 6가지 결과를 나란히 놓는다.
열: HR(정답) / bicubic / A(전체 무작위) / 기준 B(PSNR_bic로만 고름) / 구조(coh_strat B) / 텍스처(coh_strat C)

사진 고르는 규칙 (결과를 보기 전에 EXPERIMENT_LOG에 사전 등록한 규칙. 좋은 예만 골라 보여 주는 것을 막기 위함):
 - 주 세트(Set5+Set14, 19장) 중 '구조 − 기준 B'의 PSNR 차이(시드 10개 평균)가 가장 큰 3장 + 가장 작은 3장
 - 사진 안 위치: bicubic 오차가 가장 큰 64x64 조각 (09_compare_images.py와 같은 규칙)
 - 모델은 시드 0의 것 하나만 사용 (표의 10개 시드 평균과는 다를 수 있음)
 - 조각 위 숫자는 그 조각만의 PSNR(Y 채널, 가장자리 제외 없음)
사용법:  python 12_compare_visual.py      (두 실험의 results_final.csv가 있어야 함)
출력: outputs/compare_final.png"""
import csv                            # 결과 파일 읽기
import numpy as np                    # 배열 계산
import torch                          # PyTorch
import matplotlib                     # 그래프 라이브러리
matplotlib.use("Agg")                 # 창 없이 파일로만 저장
import matplotlib.pyplot as plt       # 그림 그리기
from PIL import Image                 # 사진 열기
from skimage.color import rgb2ycbcr   # RGB -> Y
import config as cfg                  # 프로젝트 설정
from srcnn import SRCNN, bicubic_down_up, to_tensor, predict   # 모델과 공통 함수

SEED_SHOWN = 0      # 그림에 쓸 모델의 시드 (사전 등록)
CROP = 64           # 조각 한 변 (HR 픽셀)
N_EACH = 3          # 이득이 큰 쪽, 작은 쪽에서 각각 몇 장
COLUMNS = [("A random", "baseline", "A"), ("base B (PSNR_bic)", "baseline", "B"),
           ("structure (coh B)", "coh_strat", "B"), ("texture (coh C)", "coh_strat", "C")]   # (열 이름, 실험, 조건)
device = "cuda" if torch.cuda.is_available() else "cpu"


def per_image_psnr(exp, method):
    """results_final.csv에서 한 그룹의 사진별 PSNR 시드 평균을 구한다 (Set5, Set14만).
    입력: exp, method / 출력: {(세트, 사진 이름): 평균 PSNR}"""
    vals = {}                                           # (세트, 사진) -> PSNR 리스트
    with open(cfg.exp_dir(exp) / "results_final.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):                     # 모든 행
            if r["method"] == method and r["set"] in ("Set5", "Set14"):   # 이 그룹, 주 세트만
                vals.setdefault((r["set"], r["image"]), []).append(float(r["psnr"]))
    return {k: np.mean(v) for k, v in vals.items()}     # 시드 평균


def load_hr(path):
    """이미지 파일 -> (H, W, 3) uint8, H·W를 4의 배수로 자름 (08과 같은 방식)"""
    hr = np.asarray(Image.open(path).convert("RGB"))
    h, w = hr.shape[0] // 4 * 4, hr.shape[1] // 4 * 4
    return np.ascontiguousarray(hr[:h, :w])


def luma(img):
    """(H, W, 3) uint8 -> (H, W) float Y 채널, 16~235"""
    return rgb2ycbcr(img)[..., 0]


def hardest_window(hr, inp, size=CROP, stride=16):
    """bicubic 오차가 가장 큰 size x size 조각의 왼쪽 위 (x, y). 09_compare_images.py와 같은 규칙"""
    err = (luma(hr) - luma(inp)) ** 2                   # 픽셀별 제곱 오차 (H, W)
    best, pos = -1.0, (0, 0)
    for y in range(0, err.shape[0] - size + 1, stride):
        for x in range(0, err.shape[1] - size + 1, stride):
            e = err[y:y + size, x:x + size].mean()      # 이 창의 평균 오차
            if e > best:
                best, pos = e, (x, y)
    return pos


# ---- 사진 고르기 (사전 등록 규칙) ----
struct, base = per_image_psnr("coh_strat", "B"), per_image_psnr("baseline", "B")   # 사진별 PSNR (시드 10개 평균)
diff = sorted(struct, key=lambda k: struct[k] - base[k])   # 차이(구조 − 기준 B)가 작은 순으로 정렬한 (세트, 사진) 목록. lambda = 이름 없는 짧은 함수
picks = diff[::-1][:N_EACH] + diff[:N_EACH]             # 큰 3장 + 작은 3장
print("고른 사진 (구조 − 기준 B, PSNR 시드 10개 평균):", [(k[1], round(struct[k] - base[k], 3)) for k in picks])

# ---- 모델 불러오기 (시드 0) ----
models = {}
for label, exp, cond in COLUMNS:
    m = SRCNN().to(device)                              # 빈 모델
    m.load_state_dict(torch.load(cfg.exp_dir(exp) / "models" / f"srcnn_{cond}_seed{SEED_SHOWN}.pt", map_location=device))   # 가중치 채우기
    models[label] = m.eval()

# ---- 그림 ----
fig, axes = plt.subplots(len(picks), 2 + len(COLUMNS), figsize=(15, 2.7 * len(picks)))   # 6행 x 6열
for r, (set_name, name) in enumerate(picks):            # 행마다 사진 하나
    files = list((cfg.ROOT / "data" / set_name).glob(f"{name}.*"))   # 확장자를 모르므로 이름으로 찾음
    hr = load_hr(files[0])                              # 정답
    inp = bicubic_down_up(hr)                           # bicubic
    x = to_tensor(inp[None]).to(device)                 # 모델 입력 (1, 3, H, W)
    imgs = {"HR (answer)": hr, "bicubic": inp}          # 열 이름 -> 사진 (넣은 순서대로 열이 됨)
    for label, m in models.items():
        out = predict(m, x)[0].permute(1, 2, 0).cpu().numpy()   # (H, W, 3) float 0~1
        imgs[label] = np.round(out * 255).astype(np.uint8)      # 08과 같은 8비트 반올림
    x0, y0 = hardest_window(hr, inp)                    # 확대할 위치
    hr_crop = hr[y0:y0 + CROP, x0:x0 + CROP]            # 정답 조각
    for c, (label, img) in enumerate(imgs.items()):
        crop = img[y0:y0 + CROP, x0:x0 + CROP]          # 같은 위치 조각
        ax = axes[r, c]
        ax.imshow(crop, interpolation="nearest")        # 픽셀이 뭉개지지 않게 확대
        ax.axis("off")
        if c == 0:                                      # 첫 열: 사진 이름과 10시드 평균 차이
            tag = "gain TOP" if r < N_EACH else "gain BOTTOM"
            ax.set_title(f"{name} [{tag}]\nstruct-baseB {struct[(set_name, name)] - base[(set_name, name)]:+.3f} dB", fontsize=8)
        else:                                           # 나머지: 조각 PSNR
            mse = ((luma(crop) - luma(hr_crop)) ** 2).mean()
            ax.set_title(f"{label}\n{10 * np.log10(255 ** 2 / max(mse, 1e-10)):.2f} dB", fontsize=8)
plt.suptitle(f"Seed {SEED_SHOWN} models, hardest 64x64 crop. Rows: top-{N_EACH} and bottom-{N_EACH} images by (structure - base B) PSNR gain", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.97])                  # 위쪽 3%는 전체 제목 자리로 비워 둠 (제목과 칸 제목이 겹치지 않게)
plt.savefig(cfg.OUT_DIR / "compare_final.png", dpi=130)
print("저장:", cfg.OUT_DIR / "compare_final.png")
