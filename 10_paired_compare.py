"""단계 ④-2: 이미지별 짝 비교. 08이 만든 results.csv(사진별 값)를 읽어, 사진마다 두 방법의 차이를 구하고
"몇 장에서 이겼는가"를 센다. 학습·평가를 다시 하지 않고 기존 결과 파일만 읽는다.

왜 필요한가:
 - 08 요약표의 ± 는 '학습 시드에 의한 변동'만 반영한다. '어떤 테스트 사진을 골랐는가'에 의한 변동은 들어 있지 않다.
 - 평균 차이가 작아도 대부분의 사진에서 같은 방향이면 의미가 있고, 절반 정도면 사실상 차이가 없다.

계산 방식:
 1) 방법마다, 사진마다 시드 평균을 먼저 구한다. (A의 시드 0과 B의 시드 0은 서로 관련이 없으므로 시드끼리 짝짓지 않는다)
 2) 사진마다 차이 = X − Y 를 구한다 (예: B − A).
 3) '좋은 방향'으로 이긴 사진 수를 센다. PSNR·SSIM은 클수록, LPIPS는 작을수록 좋다. 차이가 정확히 0이면 동점.
 4) 부호 검정 p값(참고용): "이길 확률이 반반"이라는 가정에서 이 정도로 한쪽에 몰릴 확률. 동점은 빼고 계산.
 이 방식의 '평균 차이'는 08 요약표의 (X 평균 − Y 평균)과 같은 값이 된다 (모든 사진에 같은 수의 시드가 있으므로 평균 순서를 바꿔도 같음).

추가로 '시드 단위' 비교도 출력한다 (사진 변동이 아니라 학습 변동을 보는 것):
 - 시드마다 사진 평균을 구해, 시드별 값 n개끼리 X 평균 − Y 평균과 95% 신뢰구간(Welch t 구간)을 계산한다.
 - A의 시드 s와 B의 시드 s는 짝이 아니므로(서로 다른 패치·초기값), 짝 없는 두 집단 비교(Welch)를 쓴다.
 - 신뢰구간이 0을 포함하지 않으면 "새 시드로 다시 학습해도 같은 방향일 가능성이 높다"는 근거가 된다. 시드 3개면 구간이 매우 넓다.

사용법:  python 10_paired_compare.py <실험이름>          results.csv (탐색, 시드 3개)
         python 10_paired_compare.py <실험이름> final    results_final.csv (최종 확인, 시드 10개)
입력: outputs/exp/<실험이름>/results.csv 또는 results_final.csv
출력: 화면 요약 + outputs/exp/<실험이름>/paired_diff.csv 또는 paired_diff_final.csv
      (사진별 차이. 결과 파일에서 매번 똑같이 다시 계산되므로 덮어써도 정보 손실 없음)"""
import sys                 # 명령줄 인자(sys.argv) 읽기
import csv                 # CSV 파일 읽기·쓰기
from math import comb      # comb(n, k) = n개 중 k개를 고르는 경우의 수 (부호 검정 계산용)
import numpy as np         # 배열 계산
from scipy import stats    # t 분포 (신뢰구간 계산용). scikit-image 설치 시 함께 설치되어 있음
import config as cfg       # 프로젝트 설정 (exp_dir)

PAIRS = [("B", "A"), ("C", "A"), ("B", "C")]           # 비교할 쌍 (X, Y) -> 사진마다 X − Y. 실험 질문의 핵심은 B vs A
METRICS = [("psnr", +1), ("ssim", +1), ("lpips", -1)]  # (results.csv 열 이름, 좋은 방향). +1 = 클수록 좋음, -1 = 작을수록 좋음
SETS = {"Set5": ["Set5"], "Set14": ["Set14"], "전체": ["Set5", "Set14"]}   # 표 이름 -> 포함할 세트들 (08의 요약표와 같은 구성)
# final 모드의 BSD100(독립 확인 세트)은 아래에서 결과 파일에 있을 때만 SETS에 추가한다

# ---- 실험 이름과 입력 파일 ----
USAGE = "사용법: python 10_paired_compare.py <실험이름> [final]"   # 잘못 실행했을 때 보여 줄 안내
assert len(sys.argv) in (2, 3), USAGE                  # 인자는 실험 이름 1개, 또는 실험 이름 + final
assert len(sys.argv) == 2 or sys.argv[2] == "final", USAGE   # 세 번째 인자가 있다면 반드시 "final"
exp = sys.argv[1]                                      # 실험 이름
final = len(sys.argv) == 3                             # True면 최종 확인 결과(results_final.csv)를 읽음
suffix = "_final" if final else ""                     # 파일 이름 뒤에 붙일 글자
exp_path = cfg.exp_dir(exp)                            # outputs/exp/<exp>
csv_path = exp_path / f"results{suffix}.csv"           # 08이 만든 사진별 결과
assert csv_path.exists(), f"{csv_path} 가 없습니다. 08_evaluate.py {exp}{' final' if final else ''} 를 먼저 실행하세요"

with open(csv_path, newline="", encoding="utf-8") as f:   # 읽기 모드로 열기
    rows = list(csv.DictReader(f))                        # 각 행을 {"method": "B", "seed": "0", "set": "Set5", "image": "baby", "psnr": "...", ...} 사전으로. 값은 전부 문자열
if any(r["set"] == "BSD100" for r in rows):               # 결과 파일에 BSD100 행이 있으면 (final 모드)
    SETS["BSD100 (독립 확인)"] = ["BSD100"]               # 독립 확인 세트를 따로 비교 ('전체'와 섞지 않음)


def image_means(method):
    """한 방법의 사진별 시드 평균을 구한다.
    입력: method = "A", "B", "C" 중 하나
    출력: (means, n_seed)
          means  = {(세트, 사진 이름): (3,) float 배열 [psnr, ssim, lpips]}  사진마다 시드 평균
          n_seed = 사용한 시드 수 (정수)"""
    per = {}                                           # (세트, 사진) -> 시드별 [psnr, ssim, lpips] 리스트들
    seeds = set()                                      # 등장한 시드 번호 집합 (중복 자동 제거)
    for r in rows:                                     # results.csv의 모든 행
        if r["method"] != method:                      # 다른 방법의 행은
            continue                                   # 건너뜀
        key = (r["set"], r["image"])                   # 사진을 구분하는 열쇠 (세트 이름까지 포함해야 이름이 겹쳐도 안전)
        per.setdefault(key, []).append([float(r[m]) for m, _ in METRICS])   # setdefault: 열쇠가 없으면 빈 리스트를 만들고 그 리스트를 돌려줌. 문자열 -> 실수
        seeds.add(r["seed"])                           # 시드 번호 기록
    counts = {len(v) for v in per.values()}            # 사진마다 시드 개수의 집합. 모두 같으면 원소가 1개
    assert len(counts) <= 1, f"{method}: 사진마다 시드 수가 다릅니다 {counts}"   # 일부 시드만 평가된 사진이 있으면 평균이 왜곡되므로 멈춤
    return {k: np.mean(v, axis=0) for k, v in per.items()}, len(seeds)        # axis=0: 시드 방향으로 평균 -> (3,)


def sign_test_p(wins, losses):
    """양측 부호 검정 p값 (동점 제외).
    입력: wins = X가 이긴 사진 수, losses = X가 진 사진 수
    출력: p값 (0~1). "이길 확률이 0.5"라는 가정에서, 지금처럼 한쪽으로 치우친(또는 더 치우친) 결과가 나올 확률 x 2(양쪽)
    예: 19장 중 15승 4패 -> 약 0.019, 10승 9패 -> 1.0"""
    n = wins + losses                                  # 동점을 뺀 사진 수
    if n == 0:                                         # 전부 동점이면 판단 불가
        return 1.0
    k = min(wins, losses)                              # 적은 쪽의 수
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n   # 이항분포에서 0~k번 나올 확률의 합 (한쪽 꼬리)
    return min(1.0, 2 * tail)                          # 양쪽 꼬리이므로 2배. 1을 넘으면 1로


def seed_means(method, set_names):
    """한 방법의 시드별 사진 평균을 구한다 (08 요약표가 내부에서 쓰는 값과 같음).
    입력: method = "A" 등, set_names = 포함할 세트 이름 리스트
    출력: (시드 수, 3) float 배열. 각 행 = 그 시드 모델의 [psnr, ssim, lpips] 사진 평균. 행 순서 = 시드 번호 순"""
    per = {}                                           # 시드 번호(정수) -> 사진별 [psnr, ssim, lpips] 리스트들
    for r in rows:                                     # 모든 행 중
        if r["method"] == method and r["set"] in set_names:   # 이 방법이고 포함할 세트의 사진이면
            per.setdefault(int(r["seed"]), []).append([float(r[m]) for m, _ in METRICS])   # 시드별로 모음
    return np.array([np.mean(per[s], axis=0) for s in sorted(per)])   # 시드 번호 순으로, 사진 방향 평균 -> (시드 수, 3)


def welch_ci(a, b, level=0.95):
    """짝이 없는 두 집단의 평균 차이와 신뢰구간 (Welch t 구간. 두 집단의 분산이 달라도 됨).
    입력: a, b = (n,) float 배열 (각각 시드별 값), level = 신뢰수준
    출력: (차이, 하한, 상한). 차이 = a 평균 − b 평균. 어느 쪽이든 시드가 2개 미만이면 하한·상한은 nan"""
    d = a.mean() - b.mean()                            # 평균 차이
    if len(a) < 2 or len(b) < 2:                       # 분산을 구하려면 2개 이상 필요
        return d, np.nan, np.nan                       # nan = '값 없음'
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)   # 각 평균의 분산 추정 (표본분산 / n)
    se = np.sqrt(va + vb)                              # 차이의 표준오차
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))   # Welch–Satterthwaite 자유도 (정수가 아닐 수 있음)
    t = stats.t.ppf(0.5 + level / 2, df)               # t 분포의 양측 임계값 (95%면 위쪽 2.5% 지점)
    return d, d - t * se, d + t * se                   # 차이 ± t x 표준오차


# ---- 방법별 사진 평균 준비 ----
present = sorted({r["method"] for r in rows} - {"bicubic"})   # results.csv에 있는 모델 방법들 (bicubic 제외)
means = {m: image_means(m) for m in present}                  # 방법 -> (사진별 평균 사전, 시드 수)
print(f"실험 {exp} | 방법: {present} | 시드 수: " + ", ".join(f"{m} {means[m][1]}개" for m in present))
expected = len(cfg.FINAL_SEEDS) if final else len(cfg.SEEDS)   # 이 모드에서 있어야 할 시드 수 (final 10개, 탐색 3개)
for m in present:                                              # 방법마다
    if means[m][1] != expected:                                # 시드 수가 모자라거나 다르면
        print(f"  주의: {m}의 시드가 {means[m][1]}개입니다 (이 모드의 기준 {expected}개). 학습·평가가 덜 끝난 상태일 수 있습니다")

out_rows = []                                          # paired_diff.csv에 쓸 행들
for x, y in PAIRS:                                     # 쌍마다
    if x not in means or y not in means:               # 둘 중 하나라도 결과가 없으면
        print(f"\n({x} vs {y}: 결과가 없어 건너뜀)")
        continue
    mx, my = means[x][0], means[y][0]                  # 두 방법의 사진별 평균 사전
    assert set(mx) == set(my), f"{x}와 {y}의 사진 목록이 다릅니다"   # 같은 사진끼리만 짝지을 수 있음
    # ---- (1) 시드 단위: 학습 변동을 반영한 신뢰구간 ----
    print(f"\n===== {x} vs {y} | 시드 단위 (시드별 사진 평균 {x} {means[x][1]}개 vs {y} {means[y][1]}개, 95% 신뢰구간) =====")
    for label, set_names in SETS.items():              # Set5, Set14, 전체
        sx, sy = seed_means(x, set_names), seed_means(y, set_names)   # 각 (시드 수, 3)
        print(f"[{label}]")
        for j, (m, good) in enumerate(METRICS):        # 지표마다
            d, lo, hi = welch_ci(sx[:, j], sy[:, j])   # 차이와 신뢰구간
            if np.isnan(lo):                           # 구간을 못 구한 경우
                verdict = "구간 계산 불가 (시드 2개 미만)"
            elif lo > 0 or hi < 0:                     # 구간 전체가 0의 한쪽에 있으면
                verdict = f"{x} 우세" if d * good > 0 else f"{x} 열세"   # 좋은 방향 기준으로 판정
            else:                                      # 구간이 0을 포함하면
                verdict = "0 포함 -> 판단 보류"
            print(f"  {m:6s} 차이 {d:+.4f} | 95% CI [{lo:+.4f}, {hi:+.4f}] | {verdict}")

    # ---- (2) 사진 단위: 테스트 사진 변동 (기존 짝 비교) ----
    print(f"\n----- {x} vs {y} | 사진 단위 (사진마다 시드 평균을 먼저 구한 뒤 {x} − {y}) -----")
    for label, set_names in SETS.items():              # Set5, Set14, 전체
        keys = sorted(k for k in mx if k[0] in set_names)          # 이 표에 들어갈 (세트, 사진) 목록, 이름순
        diff = np.array([mx[k] - my[k] for k in keys])            # (사진 수, 3) 사진별 차이 [dPSNR, dSSIM, dLPIPS]
        print(f"[{label}] {len(keys)}장")
        for j, (m, good) in enumerate(METRICS):                   # 지표마다 (j = 열 번호)
            d = diff[:, j] * good                                 # 좋은 방향을 + 로 통일 (LPIPS는 부호를 뒤집음)
            win = int((d > 0).sum())                              # X가 더 좋은 사진 수
            lose = int((d < 0).sum())                             # X가 더 나쁜 사진 수
            tie = len(d) - win - lose                             # 차이가 정확히 0인 사진 수
            print(f"  {m:6s} 평균 차이 {diff[:, j].mean():+.4f} | {x} 우세 {win:2d}/{len(d)}장, 열세 {lose:2d}, 동점 {tie}"
                  f" | 부호검정 p = {sign_test_p(win, lose):.3f}")   # :+.4f = 부호를 항상 붙여 소수 4자리
        if label == "전체":                                       # 사진별 상세는 '전체'에서만 출력·저장 (Set5/Set14는 그 부분집합)
            print(f"  사진별 {x} − {y}  (각 칸 옆 + 는 {x}가 더 좋음, - 는 더 나쁨)")
            for k, dv in zip(keys, diff):                         # zip: 두 목록을 같은 순서로 하나씩 짝지어 꺼냄
                marks = ["+" if v * g > 0 else ("-" if v * g < 0 else "=") for v, (_, g) in zip(dv, METRICS)]   # 지표별 승패 표시
                print(f"    {k[0]:6s} {k[1]:12s} PSNR {dv[0]:+.3f} {marks[0]}  SSIM {dv[1]:+.4f} {marks[1]}  LPIPS {dv[2]:+.4f} {marks[2]}")
                out_rows.append([f"{x}-{y}", k[0], k[1], *dv])    # *dv: 배열 값 3개를 풀어서 이어 붙임

out_csv = exp_path / f"paired_diff{suffix}.csv"                                    # 사진별 차이 저장 경로
with open(out_csv, "w", newline="", encoding="utf-8") as f:                         # 쓰기 모드 (결과 파일로부터 다시 계산되는 파생 파일)
    w = csv.writer(f)                                                             # CSV 쓰기 도구
    w.writerow(["pair", "set", "image", "d_psnr", "d_ssim", "d_lpips"])          # 머리글. 값은 X − Y 그대로(부호 뒤집지 않음)
    w.writerows(out_rows)                                                         # 결과 행 전부
print("\n저장:", out_csv)
print("해석 주의: ① 이 비교는 '테스트 사진 변동'을 보는 것이고, 08 표의 ± 는 '학습 시드 변동'이다. 둘은 다른 불확실성이다.\n"
      "          ② p값은 사진들을 서로 독립인 표본으로 가정한 참고값이다. 쌍 3개 x 지표 3개 x 세트 3개를 함께 보면 우연히 작은 p가 나올 수 있다(다중 비교).\n"
      "          ③ 사진마다 시드 평균을 쓰므로, 시드 간 흔들림이 큰 사진의 승패는 불안정할 수 있다.")
