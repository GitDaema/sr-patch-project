"""단계 ②-1b: 패치마다 '구조성(coherence)' 점수를 계산한다. (후보 ⑤ 실험용, 02의 PSNR_bic는 그대로 둔다)

직관: 픽셀마다 '밝기가 바뀌는 방향'을 화살표로 본다. 화살표가 한두 방향으로 정렬되어 있으면 구조(선·윤곽·글자·줄무늬),
      제각각이면 텍스처(잔디·털·자갈·노이즈)다.
계산:
 1) Y(밝기) 채널에서 Sobel로 가로·세로 밝기 변화(gx, gy)를 구한다.
 2) 구조 텐서: gx*gx, gy*gy, gx*gy를 가우시안(SIGMA)으로 주변과 평균한다 -> 픽셀마다 '주변 화살표들의 분포' 요약.
 3) 픽셀별 coherence = (λ1 − λ2) / (λ1 + λ2). λ1, λ2 = 구조 텐서의 두 고유값(가장 강한 방향 / 그에 수직인 방향의 변화량).
    한 방향으로만 바뀌면 λ2 ≈ 0 -> 1에 가까움(구조). 모든 방향으로 고르게 바뀌면 λ1 ≈ λ2 -> 0에 가까움(텍스처).
 4) 패치 점수 = 픽셀별 coherence를 'bicubic 오차(손실)'로 가중 평균한 값.
    이유: 우리가 알고 싶은 것은 "잃어버린 정보가 구조인가 텍스처인가"이므로, 손실이 난 곳의 구조성만 센다.
    가장자리 BORDER 픽셀은 Sobel·가우시안의 경계 처리 영향을 줄이려고 제외한다.
 값 범위 0~1, 클수록 구조적. 손실이 전혀 없는 패치(완전 단색)는 정의할 수 없어 nan.
입력: outputs/patches.npy (8000, 96, 96, 3) uint8
출력: outputs/scores_coherence.npy (8000,) float64 (patches.npy와 같은 순서)
      outputs/coherence_examples.png  B 풀 안에서 PSNR_bic가 비슷한 패치들 중 구조성 최고 8개 / 최저 8개"""
import numpy as np                    # 배열 계산
from scipy import ndimage             # Sobel 필터, 가우시안 필터
from scipy.stats import spearmanr     # 순위 상관 (sanity 점검용)
from skimage.color import rgb2ycbcr   # RGB -> Y 채널 (08 평가와 같은 함수)
import matplotlib                     # 그래프 라이브러리
matplotlib.use("Agg")                 # 창 없이 파일로만 저장하는 모드
import matplotlib.pyplot as plt       # 그림 그리기 함수
import config as cfg                  # 프로젝트 설정
from srcnn import bicubic_down_up     # 4배 축소 후 확대 (02·07·08과 같은 연산)

SIGMA = 2.0    # 구조 텐서를 평균할 가우시안 폭(픽셀). 클수록 넓은 범위의 화살표를 함께 봄 [Claude 선택값]
BORDER = 4     # 점수 계산에서 제외할 가장자리 픽셀 수 (08의 SHAVE와 같은 값) [Claude 선택값]


def coherence_score(patch):
    """패치 하나의 손실 가중 구조성 점수.
    입력: patch = (96, 96, 3) uint8, 값 0~255
    출력: float, 0~1 (클수록 구조적). 손실 합이 0이면 nan"""
    y = rgb2ycbcr(patch)[..., 0]                         # 원본 Y 채널 (96, 96) float64, 값 16~235
    y_up = rgb2ycbcr(bicubic_down_up(patch))[..., 0]     # 축소·확대한 것의 Y 채널, 같은 모양
    err = (y - y_up) ** 2                                # 픽셀별 손실 지도 (96, 96). 가중치로 사용
    gx = ndimage.sobel(y, axis=1)                        # 가로 방향(열이 바뀌는 방향) 밝기 변화 (96, 96)
    gy = ndimage.sobel(y, axis=0)                        # 세로 방향(행이 바뀌는 방향) 밝기 변화 (96, 96)
    jxx = ndimage.gaussian_filter(gx * gx, SIGMA)        # 구조 텐서 성분 1: 주변의 가로 변화 세기 평균
    jyy = ndimage.gaussian_filter(gy * gy, SIGMA)        # 구조 텐서 성분 2: 주변의 세로 변화 세기 평균
    jxy = ndimage.gaussian_filter(gx * gy, SIGMA)        # 구조 텐서 성분 3: 가로·세로 변화가 함께 움직이는 정도
    diff = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)      # λ1 − λ2 (2x2 대칭행렬 고유값의 차를 직접 계산하는 공식)
    coh = diff / (jxx + jyy + 1e-12)                     # (λ1 − λ2) / (λ1 + λ2). 1e-12는 0으로 나누기 방지 -> (96, 96), 0~1
    c = coh[BORDER:-BORDER, BORDER:-BORDER]              # 가장자리 제외 (88, 88)
    w = err[BORDER:-BORDER, BORDER:-BORDER]              # 가중치도 같은 범위만
    if w.sum() < 1e-12:                                  # 손실이 사실상 없으면(완전 단색 등)
        return np.nan                                    # 구조성을 정의할 수 없음
    return float((c * w).sum() / w.sum())                # 손실이 큰 픽셀일수록 크게 반영한 가중 평균


if __name__ == "__main__":   # 이 파일을 직접 실행할 때만 아래가 돈다 (다른 파일이 import할 때는 함수만 가져감)
    patches = np.load(cfg.OUT_DIR / "patches.npy")       # (8000, 96, 96, 3) uint8
    psnr_bic = np.load(cfg.OUT_DIR / "scores.npy")       # (8000,) float64, 02의 PSNR_bic
    coh = np.array([coherence_score(p) for p in patches])   # (8000,) float64, 패치별 구조성
    np.save(cfg.OUT_DIR / "scores_coherence.npy", coh)   # 03의 분할 기준이 읽음

    ok = ~np.isnan(coh)                                  # nan이 아닌 패치 (bool 배열)
    b_pool = psnr_bic < np.median(psnr_bic)              # 기준 실험의 B 풀(디테일 많음) 여부
    print(f"구조성 점수: 계산 {ok.sum()}개, nan {(~ok).sum()}개 | 최소 {np.nanmin(coh):.3f} | 중앙값 {np.nanmedian(coh):.3f} | 최대 {np.nanmax(coh):.3f}")
    print(f"PSNR_bic와 순위 상관: 전체 {spearmanr(coh[ok], psnr_bic[ok])[0]:+.3f} | B 풀 안 {spearmanr(coh[b_pool], psnr_bic[b_pool])[0]:+.3f}")
    print("  (0에 가까울수록 PSNR_bic와 다른 정보. B 풀 안 상관이 작아야 '손실 크기는 같고 구조성만 다른' 비교가 가능)")

    # ---- 그림: B 풀 중 PSNR_bic가 가운데 구간(B 풀의 40~60% 순위)인 패치들에서 구조성 최고 8개 / 최저 8개 ----
    b_idx = np.where(b_pool)[0]                          # B 풀 패치 번호들
    lo, hi = np.percentile(psnr_bic[b_idx], [40, 60])    # B 풀 안 PSNR_bic 40%, 60% 지점
    mid = b_idx[(psnr_bic[b_idx] >= lo) & (psnr_bic[b_idx] <= hi)]   # 손실 크기가 비슷한 패치들만
    order = mid[np.argsort(coh[mid])]                    # 구조성 낮은 순으로 정렬한 패치 번호
    rows = [("HIGH coherence (structure)", order[::-1][:8]), ("LOW coherence (texture)", order[:8])]   # 위: 구조성 최고, 아래: 최저
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.8))    # 2행 8열
    for r, (name, idx_list) in enumerate(rows):          # 행마다
        for c, i in enumerate(idx_list):                 # 칸마다
            ax = axes[r, c]                              # 그릴 칸
            ax.imshow(patches[i])                        # 패치 표시
            ax.set_title(f"coh {coh[i]:.2f} | {psnr_bic[i]:.1f}dB", fontsize=8)   # 구조성과 PSNR_bic 표시
            ax.axis("off")                               # 눈금 숨김
        axes[r, 0].text(0, -0.25, name, transform=axes[r, 0].transAxes, fontsize=11, weight="bold")   # 행 제목
    plt.suptitle(f"B pool, PSNR_bic {lo:.1f}~{hi:.1f} dB (similar loss), sorted by coherence", fontsize=11)   # 전체 제목
    plt.tight_layout()                                   # 여백 자동 조정
    plt.savefig(cfg.OUT_DIR / "coherence_examples.png", dpi=130)
    print("저장:", cfg.OUT_DIR / "scores_coherence.npy", "/", cfg.OUT_DIR / "coherence_examples.png")
