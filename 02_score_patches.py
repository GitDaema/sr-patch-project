"""단계 ②-1: 논문 식 (1)의 PSNR_bic 점수를 모든 패치에 계산한다.
점수가 낮을수록 4배 축소/확대로 잃는 정보가 많다 = 디테일이 많은 패치.
입력: outputs/patches.npy (8000, 96, 96, 3) uint8
출력: outputs/scores.npy  (8000,) float64, 단위 dB. patches.npy와 같은 순서(i번 점수 = i번 패치)
      outputs/score_hist.png 점수 분포 히스토그램"""
import numpy as np                 # 배열 계산
from PIL import Image              # bicubic 축소·확대에 사용
import matplotlib                  # 그래프 라이브러리
matplotlib.use("Agg")              # 화면 창 없이 파일로만 그림을 저장하는 모드 (pyplot을 불러오기 전에 지정해야 함)
import matplotlib.pyplot as plt    # 그래프 그리기 함수 모음
import config as cfg               # 프로젝트 설정

def psnr_bic(patch):
    """패치 하나의 PSNR_bic 점수를 구한다.
    입력: patch = PIL RGB 이미지 (96x96)
    출력: float, 단위 dB. 원본과 '4배 축소 후 4배 확대한 것' 사이의 PSNR (RGB 3채널 전체, 최대값 255 기준)
    낮을수록 축소로 잃는 정보가 많음 = 디테일이 많음. 완전 단색이면 mse가 0이라 하한 1e-10이 적용되어 약 148.13 dB"""
    s = patch.size[0]                                    # 패치 한 변 길이 (96). 정사각형이라 가로만 봄
    lr = patch.resize((s // 4, s // 4), Image.BICUBIC)   # PIL의 BICUBIC 축소는 안티에일리어싱이 포함됨. // 는 정수 나눗셈 (96 -> 24)
    up = lr.resize((s, s), Image.BICUBIC)                # (OpenCV INTER_CUBIC 축소는 포함 안 됨 -> 점수가 달라질 수 있음). 24 -> 96으로 다시 확대
    a = np.asarray(patch, dtype=np.float64)              # 원본을 실수 배열로 (96, 96, 3) float64, 값 0~255. uint8 그대로 빼면 음수가 깨지므로 실수로 변환
    b = np.asarray(up, dtype=np.float64)                 # 축소·확대 결과도 같은 형태
    mse = ((a - b) ** 2).mean()                          # 평균 제곱 오차: 모든 픽셀·채널의 (차이)^2 평균
    return 10 * np.log10(255 ** 2 / max(mse, 1e-10))     # PSNR 공식. mse가 0이면 0으로 나누게 되므로 1e-10을 하한으로 둠

patches = np.load(cfg.OUT_DIR / "patches.npy")           # (8000, 96, 96, 3) uint8
scores = np.array([psnr_bic(Image.fromarray(p)) for p in patches])   # 패치마다 PIL 이미지로 바꿔 점수 계산 -> (8000,) float64
np.save(cfg.OUT_DIR / "scores.npy", scores)              # 03, 04, 05, 07이 이 파일을 읽음

thr = np.median(scores)                                  # 중앙값 (여기서는 출력·그림용. 실제 분할은 03에서 다시 계산)
print(f"패치 {len(scores)}개 | 최소 {scores.min():.2f} dB | 중앙값 {thr:.2f} dB | 최대 {scores.max():.2f} dB")
print("(참고: 논문은 OST 데이터에서 중앙값이 23 dB. 이 실험 데이터의 중앙값은 위 값이고, 달라도 정상)")

plt.figure(figsize=(6, 3.5))                             # 새 그림. 크기 6x3.5 인치
plt.hist(scores, bins=60)                                # 점수 히스토그램 (구간 60개)
plt.axvline(thr, color="red", label=f"median = {thr:.2f} dB")   # 중앙값 위치에 빨간 세로선
plt.xlabel("PSNR_bic (dB)"); plt.ylabel("number of patches"); plt.legend()   # 축 이름과 범례 (; 는 한 줄에 여러 문장)
plt.tight_layout()                                       # 글자가 잘리지 않게 여백 자동 조정
plt.savefig(cfg.OUT_DIR / "score_hist.png", dpi=150)     # 파일로 저장 (dpi = 해상도)
print("저장:", cfg.OUT_DIR / "score_hist.png")
