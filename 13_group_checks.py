"""단계 ⑤: 최종 주장의 신뢰성 점검 (학습·평가를 다시 하지 않고, 이미 있는 그룹 파일·점수·학습 로그만 읽는다).

점검 1. 섞인 변수: 비교하는 그룹들이 '구조성' 말고 다른 성질도 다른가?
        (손실 크기 PSNR_bic, 원본 이미지 다양성, 대비=픽셀 표준편차, 평균 밝기) — 시드 0~9 평균
점검 2. 구조성 설정값 민감도: 구조성 계산의 SIGMA를 1, 4로 바꿔도 같은 패치가 '구조 쪽'으로 분류되는가? (계산만)
점검 3. 학습 패치 개선폭: 07 로그의 '학습 전 → 학습 후' 중앙값 PSNR 차이. 손실 크기가 같은 그룹끼리 '얼마나 배웠나'를 비교

사용법:  python 13_group_checks.py
입력: outputs/patches.npy, scores.npy, scores_coherence.npy, outputs/exp/{baseline,coh_strat}/groups_seed*.npz, train_*.log"""
import re                     # 정규식: 로그에서 숫자 뽑기
import importlib              # 이름이 숫자로 시작하는 파일(02b_...)을 모듈로 불러오기 위해 (import 문으로는 불가)
import numpy as np            # 배열 계산
from scipy.stats import spearmanr   # 순위 상관
import config as cfg          # 프로젝트 설정
from patch_info import n_images     # 서로 다른 원본 이미지 수

GROUPS = [("A (전체 무작위)", "baseline", "A"), ("기준 B (B풀 무작위)", "baseline", "B"),
          ("구조 (coh B)", "coh_strat", "B"), ("텍스처 (coh C)", "coh_strat", "C")]   # (표시 이름, 실험, 조건)

patches = np.load(cfg.OUT_DIR / "patches.npy")          # (8000, 96, 96, 3) uint8
psnr_bic = np.load(cfg.OUT_DIR / "scores.npy")          # (8000,) PSNR_bic
coh = np.load(cfg.OUT_DIR / "scores_coherence.npy")     # (8000,) 구조성 (SIGMA=2)
flat = patches.reshape(len(patches), -1).astype(np.float32)   # (8000, 96*96*3) 패치마다 한 줄로 펼침 (통계 계산용)
contrast = flat.std(axis=1)                             # (8000,) 패치별 픽셀 표준편차 = 대비
bright = flat.mean(axis=1)                              # (8000,) 패치별 평균 밝기 (0~255)

# ---- 점검 1: 섞인 변수 ----
print("== 점검 1: 그룹별 성질 (시드 0~9 평균) ==")
print(f"{'그룹':22s}{'PSNR_bic':>10s}{'구조성':>8s}{'원본이미지수':>12s}{'대비':>8s}{'밝기':>8s}")
for name, exp, cond in GROUPS:                          # 그룹마다
    vals = []                                           # 시드별 [PSNR_bic, 구조성, 이미지 수, 대비, 밝기]
    for s in cfg.FINAL_SEEDS:                           # 시드 0~9
        idx = np.load(cfg.exp_dir(exp) / f"groups_seed{s}.npz")[cond]   # 이 그룹의 패치 번호 (N,)
        vals.append([psnr_bic[idx].mean(), np.nanmean(coh[idx]), n_images(idx), contrast[idx].mean(), bright[idx].mean()])   # nanmean: 단색 패치의 nan은 빼고 평균
    v = np.mean(vals, axis=0)                           # 시드 평균
    print(f"{name:22s}{v[0]:10.2f}{v[1]:8.3f}{v[2]:12.0f}{v[3]:8.1f}{v[4]:8.1f}")

# ---- 점검 2: 구조성 SIGMA 민감도 ----
print("\n== 점검 2: 구조성 SIGMA 민감도 (B 풀 4000개, 계산만) ==")
mod = importlib.import_module("02b_score_structure")    # 02b 파일을 모듈로 불러옴 (if __name__ == "__main__" 아래는 실행되지 않음)
pool = np.where(psnr_bic < np.median(psnr_bic))[0]      # baseline B 풀
order = pool[np.argsort(psnr_bic[pool])]                # PSNR_bic 순 (splits.py와 같은 10구간 방식)
bins = np.array_split(order, 10)                        # 10구간


def struct_side(c):
    """구조성 배열 c (8000,)로, B 풀의 각 패치가 구간 안에서 '구조 쪽(상위 절반)'인지를 돌려준다. 출력: {패치 번호: True/False}"""
    side = {}
    for b in bins:                                      # 구간마다
        s = b[np.argsort(c[b])]                         # 구조성 낮은 순
        for k, i in enumerate(s):                       # 정렬된 순서 k, 패치 번호 i
            side[i] = k >= len(s) // 2                  # 뒤 절반이면 구조 쪽
    return side


base_side = struct_side(coh)                            # SIGMA=2 (실제 실험에 쓴 값) 분류
for sig in [1.0, 4.0]:                                  # 비교할 SIGMA 값들
    mod.SIGMA = sig                                     # 모듈의 설정값을 바꿔서
    c2 = np.full(len(patches), np.nan)                  # 새 구조성 (B 풀만 계산)
    c2[pool] = [mod.coherence_score(patches[i]) for i in pool]
    side2 = struct_side(c2)                             # 새 분류
    agree = np.mean([base_side[i] == side2[i] for i in pool])   # 같은 쪽으로 분류된 비율
    print(f"  SIGMA {sig}: SIGMA 2와 순위 상관 {spearmanr(coh[pool], c2[pool])[0]:+.3f} | 구조/텍스처 분류 일치 {agree * 100:.1f}%")
print("  (일치율이 높으면 결과가 SIGMA 선택에 크게 좌우되지 않는다는 근거. 50%면 무작위 수준)")

# ---- 점검 3: 학습 패치 개선폭 ----
print("\n== 점검 3: 학습 패치 기준 개선폭 (07 로그: 학습 후 중앙값 − 학습 전 중앙값, dB) ==")
for name, exp, cond in GROUPS:                          # 그룹마다
    gains = []                                          # 시드별 개선폭
    for s in cfg.FINAL_SEEDS:
        log = cfg.exp_dir(exp) / f"train_{cond}{s}.log"  # 학습 로그
        if not log.exists():                            # 없으면 건너뜀
            continue
        text = log.read_bytes().decode("utf-8", errors="ignore")   # 일부 로그는 다른 인코딩(cp949)이라, 글자가 깨져도 숫자만 읽으면 되므로 무시하고 읽음
        nums = [float(x) for x in re.findall(r"(\d+\.\d+) dB", text)]   # 'xx.xx dB' 숫자들, 순서: 학습전 평균, 학습전 중앙값, PSNR_bic 평균, 학습후 평균, 학습후 중앙값
        if len(nums) >= 4:
            gains.append(nums[4] - nums[1])             # 학습 후 중앙값 − 학습 전 중앙값. 평균은 거의 단색인 패치(148 dB)에 크게 흔들려서 중앙값을 씀 (07의 안내와 같은 이유)
    g = np.array(gains)
    print(f"  {name:22s} 시드 {len(g)}개 | 개선 평균 {g.mean():+.3f} dB (최소 {g.min():+.3f}, 최대 {g.max():+.3f})" if len(g) else f"  {name}: 로그 없음")
print("  (학습에 쓴 패치 기준이라 성능 근거가 아니라 '얼마나 배울 거리가 있었나'의 진단값. A는 학습 전 점수가 B·C보다 훨씬 높아(밋밋한 패치 포함) B·C와 직접 비교하지 않음)")
