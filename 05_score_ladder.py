"""진단용: 점수 구간(백분위)별로 실제 패치가 어떻게 생겼는지 보여 준다.
목적: 어느 점수부터 '밋밋한 풍경'이 아니라 '사실상 단색(쓸모없는 패치)'인지 눈으로 정한다.
입력: outputs/patches.npy, outputs/scores.npy
출력: 화면에 점수 구간별 개수, outputs/score_ladder.png"""
import numpy as np                 # 배열 계산
import matplotlib                  # 그래프 라이브러리
matplotlib.use("Agg")              # 창 없이 파일로만 저장하는 모드
import matplotlib.pyplot as plt    # 그림 그리기 함수
import config as cfg               # 프로젝트 설정

patches = np.load(cfg.OUT_DIR / "patches.npy")   # (8000, 96, 96, 3) uint8
scores = np.load(cfg.OUT_DIR / "scores.npy")     # (8000,) float64

# 1) 숫자로 보기: 점수가 높은 쪽 꼬리에 패치가 몇 개나 있는가
print("점수 구간별 패치 수 (해당 dB 이상):")
for t in [40, 50, 60, 70, 80, 100]:              # 기준 점수 t를 하나씩
    n = int((scores >= t).sum())                 # t 이상인 패치 수 (bool 배열의 합 = 참의 개수)
    print(f"  {t:3d} dB 이상: {n:5d}개 ({n / len(scores) * 100:.2f}%)")

# 2) 눈으로 보기: 점수 순위(백분위)별로 그 근처 패치 8개를 무작위로 보여 줌
rng = np.random.default_rng(0)                   # 그림이 매번 같게 나오도록 시드 고정 (03의 시드와는 무관)
order = np.argsort(scores)                       # 점수 낮은 순
levels = [50, 75, 90, 95, 98, 99, 99.5]          # 백분위 (위로 갈수록 밋밋)
fig, axes = plt.subplots(len(levels), 8, figsize=(14, 1.9 * len(levels)))   # 백분위 개수(7)행 x 8열
for r, q in enumerate(levels):                   # r = 행 번호, q = 백분위
    center = int(len(order) * q / 100)           # 그 백분위에 해당하는 순위 (예: 50% -> 4000번째)
    lo, hi = max(center - 40, 0), min(center + 40, len(order))   # 그 순위 주변 최대 80개. max/min으로 배열 범위를 벗어나지 않게
    picks = rng.choice(order[lo:hi], 8, replace=False)           # 그 구간의 패치 번호 중 8개를 중복 없이 무작위로
    for c, i in enumerate(picks):                # c = 열 번호, i = 패치 번호
        ax = axes[r, c]                          # 이번에 그릴 칸
        ax.imshow(patches[i])                    # 패치 표시
        ax.axis("off")                           # 눈금 숨김
        ax.set_title(f"{scores[i]:.1f} dB", fontsize=8)   # 점수 표시
    axes[r, 0].text(-0.12, 0.5, f"{q}%", transform=axes[r, 0].transAxes,   # 행 왼쪽 바깥에 백분위 라벨
                    ha="right", va="center", fontsize=11, weight="bold")
plt.tight_layout()                               # 여백 자동 조정
plt.subplots_adjust(left=0.07)                   # 왼쪽 라벨이 잘리지 않게 왼쪽 여백을 따로 확보
plt.savefig(cfg.OUT_DIR / "score_ladder.png", dpi=130)
print("저장:", cfg.OUT_DIR / "score_ladder.png")
