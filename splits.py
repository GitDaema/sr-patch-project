"""분할 기준 모음. 03_make_groups.py가 '실험 이름'으로 여기서 기준 함수를 골라 쓴다.

왜 따로 모았나: 기준을 03 안에서 직접 고치면 이전 실험의 기준 코드가 사라져, 그 실험을 다시 만들거나 검증할 수 없다.
               기준마다 함수를 하나씩 두면 모든 실험을 언제든 똑같이 재현·검증할 수 있다.

새 기준을 추가하는 방법:
 1) 아래 형식대로 함수를 하나 만든다.  def split_이름(scores): ... return {...}
 2) 맨 아래 SPLITS 사전에 "실험이름": 함수 를 한 줄 추가한다.
 3) python 03_make_groups.py 실험이름

기준 함수의 약속(모든 기준 함수가 지켜야 하는 형식):
 입력: scores = (8000,) float64, PSNR_bic (모든 기준이 참고용으로 받음)
 출력: 사전(dict)
   "desc"  : 기준을 설명하는 한 줄 문자열 (split_info.txt에 기록)
   "labels": {"B": B의 의미, "C": C의 의미} 문자열
   "info"  : 준비 단계에서 출력할 줄들의 리스트 (풀 크기 등)
   "draw"  : 함수. draw(rng) -> (B, C). 각각 (N,) 정수 패치 번호, 중복 없음.
             03이 A를 먼저 뽑은 '같은 rng'를 넘겨주므로, draw 안에서 rng를 쓰는 순서도 결과를 결정한다.
   "check" : 함수. check(A, B, C) -> 출력할 줄들의 리스트. 이 기준에 맞는 sanity 점검 (시드마다 호출)"""
import numpy as np     # 배열 계산
import config as cfg   # 프로젝트 설정 (N, OUT_DIR)
from patch_info import n_images   # 그룹이 몇 장의 서로 다른 원본 이미지에서 왔는지 세는 함수


# ======================= 기준 1: baseline (PSNR_bic 중앙값 분할) =======================
def split_median(scores):
    """기준 실험. B = PSNR_bic 중앙값 미만(디테일 많음), C = 중앙값 이상(밋밋함). 각 풀 안에서 균등 무작위.
    입력·출력: 위 '기준 함수의 약속' 참고"""
    thr = np.median(scores)                       # 중앙값은 전체 풀에서 한 번만 구한다 (시드마다 바꾸지 않음)
    detail_pool = np.where(scores < thr)[0]       # B가 뽑을 풀. np.where(조건)[0] = 조건이 참인 위치(패치 번호)들의 배열
    flat_pool = np.where(scores >= thr)[0]        # C가 뽑을 풀. 중앙값과 같은 점수는 이쪽에 들어감
    assert cfg.N <= min(len(detail_pool), len(flat_pool)), "N이 풀보다 큽니다"   # 중복 없이 N개를 뽑으려면 풀이 N 이상이어야 함

    def draw(rng):
        """이 시드의 rng로 B, C를 뽑는다. (경로 분리 전 03과 똑같은 호출 순서 -> 기준 그룹 파일과 동일한 결과)"""
        B = rng.choice(detail_pool, cfg.N, replace=False)   # 디테일 풀 안에서 N개
        C = rng.choice(flat_pool, cfg.N, replace=False)     # 밋밋 풀 안에서 N개
        return B, C

    def check(A, B, C):
        """이 기준의 sanity: 평균 PSNR_bic가 B < A < C 순서여야 정상"""
        ok = scores[B].mean() < scores[A].mean() < scores[C].mean()   # 파이썬은 a < b < c 를 한 번에 쓸 수 있음
        return [f"         B < A < C (평균 PSNR_bic): {ok}"]

    return {
        "desc": "PSNR_bic 중앙값 분할: B = 중앙값 미만, C = 중앙값 이상 (풀 안에서 균등 무작위)",
        "labels": {"B": "디테일 많음(PSNR_bic 하위 50%)", "C": "밋밋함(PSNR_bic 상위 50%)"},
        "info": [f"중앙값 {thr:.2f} dB | B 풀 {len(detail_pool)}개 | C 풀 {len(flat_pool)}개 | N = {cfg.N}"],
        "draw": draw,
        "check": check,
    }


# ============ 기준 2: coh_strat (후보 ⑤: 손실 크기는 맞추고 구조성만 다르게) ============
COH_BINS = 10   # B 풀을 PSNR_bic 순위로 나눌 구간 수. 구간마다 같은 개수를 뽑아 두 그룹의 PSNR_bic 분포를 맞춘다 [Claude 선택값]


def split_coh_strat(scores):
    """후보 ⑤. 기준 실험의 B 풀(PSNR_bic 중앙값 미만, 4000개) 안에서만 뽑는다.
    B 풀을 PSNR_bic 순위로 COH_BINS개 구간(각 약 400개)으로 나누고, 구간마다 구조성(coherence) 중앙값으로 반을 나눈다.
      B = 각 구간의 구조성 상위 절반에서 N/COH_BINS개씩 -> 구조적인 손실(선·윤곽·줄무늬)
      C = 각 구간의 구조성 하위 절반에서 N/COH_BINS개씩 -> 텍스처 손실(잔디·잎·자갈)
    두 그룹이 PSNR_bic 구간마다 같은 개수이므로 '손실 크기'는 거의 같고 '구조성'만 다르다 (변수 하나만 바꾸기).
    주의: 이 실험의 C는 baseline의 C(밋밋함)와 뜻이 다르다. 입력·출력: 위 '기준 함수의 약속' 참고"""
    coh = np.load(cfg.OUT_DIR / "scores_coherence.npy")   # (8000,) float64, 0~1. 02b_score_structure.py가 만든 구조성 점수
    pool = np.where(scores < np.median(scores))[0]        # 기준 실험의 B 풀 (4000개)
    assert not np.isnan(coh[pool]).any(), "B 풀에 구조성 nan이 있습니다"   # 손실이 있는 패치들이라 nan이 없어야 정상
    assert cfg.N % COH_BINS == 0, "N이 구간 수로 나누어떨어져야 합니다"    # 구간마다 같은 개수를 뽑기 위해
    per_bin = cfg.N // COH_BINS                           # 구간마다 뽑을 개수 (1000 // 10 = 100)

    order = pool[np.argsort(scores[pool])]                # B 풀을 PSNR_bic 낮은 순으로 정렬한 패치 번호
    bins = np.array_split(order, COH_BINS)                # 순위대로 COH_BINS개 덩어리로 자름 (각 약 400개). 리스트의 원소 = 패치 번호 배열
    hi_halves, lo_halves = [], []                         # 구간별 구조성 상위 절반 / 하위 절반
    for b in bins:                                        # 구간마다
        s = b[np.argsort(coh[b])]                         # 이 구간 패치를 구조성 낮은 순으로 정렬
        half = len(s) // 2                                # 절반 위치
        lo_halves.append(s[:half])                        # 앞 절반 = 구조성 낮음 (텍스처)
        hi_halves.append(s[half:])                        # 뒤 절반 = 구조성 높음 (구조)
        assert per_bin <= half, "구간의 절반이 뽑을 개수보다 작습니다"   # 중복 없이 뽑을 수 있는지

    def draw(rng):
        """구간마다 per_bin개씩 뽑아 이어 붙인다. B(구조) 전부 -> C(텍스처) 전부 순서로 rng를 쓴다."""
        B = np.concatenate([rng.choice(h, per_bin, replace=False) for h in hi_halves])   # (N,) 구조 그룹
        C = np.concatenate([rng.choice(l, per_bin, replace=False) for l in lo_halves])   # (N,) 텍스처 그룹
        return B, C

    def check(A, B, C):
        """이 기준의 sanity:
        ① B와 C의 PSNR_bic 평균·중앙값이 거의 같아야 한다 (손실 크기를 맞췄는가)
        ② 구조성은 B가 C보다 확실히 높아야 한다 (바꾸려던 변수가 실제로 바뀌었는가)
        ③ 원본 이미지 다양성 (기록용. 차이가 크면 '다양성'이 섞인 변수라는 뜻)"""
        return [
            f"         PSNR_bic 평균  B {scores[B].mean():.2f} | C {scores[C].mean():.2f}   중앙값 B {np.median(scores[B]):.2f} | C {np.median(scores[C]):.2f}  (거의 같아야 정상)",
            f"         구조성 평균  A {np.nanmean(coh[A]):.3f} | B {coh[B].mean():.3f} | C {coh[C].mean():.3f}  (B > C 여야 정상)",
            f"         서로 다른 원본 이미지 수  A {n_images(A)} | B {n_images(B)} | C {n_images(C)}",
        ]

    return {
        "desc": f"후보⑤ coh_strat: baseline B 풀(PSNR_bic<중앙값) 안에서 PSNR_bic {COH_BINS}구간별로 구조성 상위 절반=B, 하위 절반=C, 구간마다 {per_bin}개씩",
        "labels": {"B": "구조적 손실(구조성 높음)", "C": "텍스처 손실(구조성 낮음)"},
        "info": [f"B 풀 {len(pool)}개 -> {COH_BINS}구간 (구간당 약 {len(pool) // COH_BINS}개, 절반씩 구조/텍스처) | 구간당 {per_bin}개씩 뽑음 | N = {cfg.N}"],
        "draw": draw,
        "check": check,
    }


# ======================= 실험 이름 -> 기준 함수 =======================
SPLITS = {
    "baseline": split_median,      # 기준 실험
    "coh_strat": split_coh_strat,  # 후보 ⑤
}
