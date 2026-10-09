"""단계 ⑤-2 (탐색 분석, 사전 등록하지 않음): "구조 그룹으로 배운 모델이 정말 더 선명한가?"를 숫자와 그림으로 본다.

선명도 지표 = 결과 이미지의 경계 세기 / 정답 이미지의 경계 세기 (Y 채널 Sobel 크기의 평균, 가장자리 4px 제외)
  1.0이면 정답만큼 선명, 작을수록 흐림. bicubic은 약 0.5~0.6.
  ⚠ 경계 세기가 크다고 '정확하다'는 뜻은 아니다(노이즈나 과장된 경계도 세기를 키운다). 그래서 경계 위 MSE도 함께 본다.
영역 나누기: 원본(정답)의 경계 세기 상위 10% 픽셀 = '강한 경계', 나머지 = '나머지(면·텍스처)'
대상: 시드 0~9 모델 x 4그룹 (A, 기준 B, 구조, 텍스처), Set5+Set14(19장)와 BSD100(100장)
출력: 화면 표 + outputs/sharpness.txt, outputs/diffmap_struct_vs_baseB.png (시드 0, 두 모델 차이 지도)
사용법:  python 14_sharpness.py"""
import numpy as np                    # 배열 계산
import torch                          # PyTorch
import matplotlib                     # 그래프 라이브러리
matplotlib.use("Agg")                 # 창 없이 파일로만 저장
import matplotlib.pyplot as plt       # 그림 그리기
from PIL import Image                 # 사진 열기
from scipy import ndimage             # Sobel 필터
from skimage.color import rgb2ycbcr   # RGB -> Y
import config as cfg                  # 프로젝트 설정
from srcnn import SRCNN, bicubic_down_up, to_tensor, predict   # 모델과 공통 함수

GROUPS = [("A", "baseline", "A"), ("baseB", "baseline", "B"), ("struct", "coh_strat", "B"), ("texture", "coh_strat", "C")]   # (이름, 실험, 조건)
SHAVE = 4                             # 가장자리 제외 픽셀 (08과 같음)
EDGE_TOP = 90                         # 원본 경계 세기 상위 (100-EDGE_TOP)% 픽셀을 '강한 경계'로 [Claude 선택값]
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_hr(path):
    """이미지 파일 -> (H, W, 3) uint8, 4의 배수로 자름 (08과 같은 방식)"""
    hr = np.asarray(Image.open(path).convert("RGB"))
    h, w = hr.shape[0] // 4 * 4, hr.shape[1] // 4 * 4
    return np.ascontiguousarray(hr[:h, :w])


def grad_map(img):
    """(H, W, 3) uint8 -> (H-8, W-8) float, Y 채널 경계 세기(Sobel 가로·세로 크기). 가장자리 SHAVE 픽셀 제외"""
    y = rgb2ycbcr(img)[..., 0]                                       # Y 채널 (H, W)
    g = np.hypot(ndimage.sobel(y, axis=0), ndimage.sobel(y, axis=1))  # hypot(a, b) = sqrt(a^2 + b^2) = 경계 세기
    return g[SHAVE:-SHAVE, SHAVE:-SHAVE]


def to_uint8(model, inp):
    """모델 출력 -> (H, W, 3) uint8 (08과 같은 8비트 반올림). 입력 inp: bicubic 결과 (H, W, 3) uint8"""
    out = predict(model, to_tensor(inp[None]).to(device))[0].permute(1, 2, 0).cpu().numpy()   # (H, W, 3) float 0~1
    return np.round(out * 255).astype(np.uint8)


def load_model(exp, cond, seed):
    """저장된 가중치로 SRCNN을 만들어 평가 모드로 돌려준다"""
    m = SRCNN().to(device)
    m.load_state_dict(torch.load(cfg.exp_dir(exp) / "models" / f"srcnn_{cond}_seed{seed}.pt", map_location=device))
    return m.eval()


# ---- 사진 준비: (정답, bicubic, 정답 경계 지도, 강한 경계 마스크) ----
SETS = {"Set5+14": sorted((cfg.ROOT / "data" / "Set5").iterdir()) + sorted((cfg.ROOT / "data" / "Set14").iterdir()),
        "BSD100": sorted((cfg.ROOT / "data" / "BSD100").iterdir())}
data = {}
for k, files in SETS.items():
    data[k] = []
    for p in files:
        hr = load_hr(p)                                  # 정답
        g = grad_map(hr)                                 # 정답의 경계 세기 지도
        mask = g >= np.percentile(g, EDGE_TOP)           # 강한 경계 위치 (bool 지도)
        data[k].append((hr, bicubic_down_up(hr), g, mask))

# ---- 그룹 x 시드마다 측정 ----
res = {}                                                 # (그룹 이름, 세트) -> 시드별 [전체 선명도, 경계 위 선명도, 나머지 선명도, 경계 위 MSE]
for name, exp, cond in GROUPS:
    for s in cfg.FINAL_SEEDS:                            # 시드 0~9
        m = load_model(exp, cond, s)
        for k in SETS:
            vals = []                                    # 사진별 4개 값
            for hr, inp, g, mask in data[k]:
                o = to_uint8(m, inp)                     # 복원 결과
                go = grad_map(o)                         # 결과의 경계 세기 지도
                yo = rgb2ycbcr(o)[..., 0][SHAVE:-SHAVE, SHAVE:-SHAVE]   # 결과 Y
                yh = rgb2ycbcr(hr)[..., 0][SHAVE:-SHAVE, SHAVE:-SHAVE]  # 정답 Y
                vals.append([go.mean() / g.mean(),                      # 전체 선명도
                             go[mask].mean() / g[mask].mean(),          # 강한 경계 위 선명도
                             go[~mask].mean() / g[~mask].mean(),        # 나머지 영역 선명도 (~ = 참/거짓 뒤집기)
                             ((yo - yh) ** 2)[mask].mean()])            # 강한 경계 위 MSE
            res.setdefault((name, k), []).append(np.mean(vals, axis=0))   # 사진 평균 -> 이 시드의 값

lines = []
for k in SETS:
    b = np.mean([grad_map(inp).mean() / g.mean() for hr, inp, g, mask in data[k]])   # bicubic의 전체 선명도
    lines.append(f"[{k}] 선명도 = 결과 경계 세기 / 정답 경계 세기 (1.0 = 정답만큼). bicubic 전체 {b:.4f}. 시드 10개 평균 ± 표준편차")
    lines.append(f"   {'그룹':8s}{'전체':>16s}{'강한 경계 위':>18s}{'나머지':>18s}{'경계 위 MSE(Y)':>20s}")
    for name, _, _ in GROUPS:
        v = np.array(res[(name, k)])                     # (10, 4)
        mu, sd = v.mean(axis=0), v.std(axis=0, ddof=1)
        lines.append(f"   {name:8s}" + "".join(f"{mu[j]:>10.4f}±{sd[j]:.4f}" for j in range(3)) + f"{mu[3]:>12.2f}±{sd[3]:.2f}")
print("\n".join(lines))
(cfg.OUT_DIR / "sharpness.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---- 차이 지도 그림 (시드 0): 두 모델의 결과가 어디서 다른가, 어느 쪽이 정답에 가까운가 ----
st, bb = load_model("coh_strat", "B", 0), load_model("baseline", "B", 0)
names = [("Set5", "butterfly"), ("Set14", "ppt3"), ("Set5", "bird")]   # 이득 상위(butterfly, ppt3)와 하위(bird) 예시
fig, axes = plt.subplots(len(names), 4, figsize=(14, 3.6 * len(names)))
for r, (sn, n) in enumerate(names):
    hr = load_hr(next((cfg.ROOT / "data" / sn).glob(n + ".*")))
    inp = bicubic_down_up(hr)
    a, b = to_uint8(st, inp).astype(float), to_uint8(bb, inp).astype(float)   # 구조 모델, 기준 B 모델 결과
    gain = np.abs(b - hr).mean(2) - np.abs(a - hr).mean(2)   # 양수 = 구조 모델이 정답에 더 가까움
    v = np.percentile(np.abs(gain), 99)                  # 색 범위 (극단값 1% 무시)
    panels = [(hr, None, n), (grad_map(hr), "gray", "HR edge strength"),
              (np.clip(np.abs(a - b).mean(2) * 10, 0, 255), "hot", "|struct - baseB| x10"),
              (gain, "bwr_r", "blue = struct closer to HR / red = baseB closer")]
    for c, (img, cmap, title) in enumerate(panels):
        ax = axes[r, c]
        if c == 3:
            ax.imshow(img, cmap=cmap, vmin=-v, vmax=v)   # 0을 흰색 가운데로
        else:
            ax.imshow(img if cmap is None else img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
plt.tight_layout()
plt.savefig(cfg.OUT_DIR / "diffmap_struct_vs_baseB.png", dpi=90)
print("저장:", cfg.OUT_DIR / "sharpness.txt", "/", cfg.OUT_DIR / "diffmap_struct_vs_baseB.png")
