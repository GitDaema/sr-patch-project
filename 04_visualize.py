"""단계 ②-3: 점수가 가장 낮은 패치 8개(위)와 가장 높은 패치 8개(아래)를 한 장에 붙인다.
입력: outputs/patches.npy, outputs/scores.npy
출력: outputs/lowest_vs_highest.png"""
import numpy as np                 # 배열 계산
import matplotlib                  # 그래프 라이브러리
matplotlib.use("Agg")              # 창 없이 파일로만 저장하는 모드 (pyplot보다 먼저)
import matplotlib.pyplot as plt    # 그림 그리기 함수
import config as cfg               # 프로젝트 설정

patches = np.load(cfg.OUT_DIR / "patches.npy")   # (8000, 96, 96, 3) uint8
scores = np.load(cfg.OUT_DIR / "scores.npy")     # (8000,) float64

is_constant = scores > 100          # 148.13 dB = 완전 단색 패치. (8000,) bool 배열 (참/거짓)
print("단색 패치 수:", is_constant.sum(), "/", len(scores))   # bool 배열의 sum = 참의 개수
order = np.argsort(scores)          # 점수 낮은 순으로 정렬했을 때의 패치 번호들 (값이 아니라 위치를 돌려줌)
lowest = order[:8]                  # 앞에서 8개 = 점수가 가장 낮은(디테일 많은) 8개
highest = [i for i in order[::-1] if not is_constant[i]][:8]   # order[::-1] = 뒤집어서 높은 순. 완전 단색은 건너뛰고 앞에서 8개

fig, axes = plt.subplots(2, 8, figsize=(16, 4.6))   # 2행 8열 칸으로 된 그림. axes[행, 열]로 각 칸에 접근
for row, idx_list, name in [(0, lowest, "LOWEST score (most detail)"),   # (행 번호, 패치 번호 목록, 행 제목)을 차례로
                            (1, highest, "HIGHEST score (flattest)")]:
    for col, i in enumerate(idx_list):               # enumerate: (순서 번호, 값)을 함께 꺼냄 -> 열 번호 col, 패치 번호 i
        ax = axes[row, col]                          # 이번에 그릴 칸
        ax.imshow(patches[i])                        # 패치 이미지 표시 (uint8 RGB)
        ax.set_title(f"{scores[i]:.1f} dB", fontsize=9)   # 칸 위에 점수 표시
        ax.axis("off")                               # 눈금·테두리 숨김
    axes[row, 0].text(0, -0.28, name, transform=axes[row, 0].transAxes, fontsize=11, weight="bold")   # 행의 첫 칸 아래에 행 제목 (transAxes: 칸 기준 0~1 좌표)
plt.tight_layout()                                   # 여백 자동 조정
plt.savefig(cfg.OUT_DIR / "lowest_vs_highest.png", dpi=150)
print("저장:", cfg.OUT_DIR / "lowest_vs_highest.png")
