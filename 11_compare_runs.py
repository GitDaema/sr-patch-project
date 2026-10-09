"""단계 ④-3: 서로 다른 실험의 그룹끼리 비교한다. (10은 같은 실험 안의 A/B/C만 비교할 수 있어서 따로 만듦)
예: 구조 그룹(coh_strat의 B) vs PSNR_bic로만 고른 그룹(baseline의 B). 두 그룹은 손실 크기(PSNR_bic 분포)가 같다.

사용법:  python 11_compare_runs.py <실험1>:<조건> <실험2>:<조건> [final]
         예) python 11_compare_runs.py coh_strat:B baseline:B final      -> 사진마다 (coh_strat B) − (baseline B)
입력: 각 실험의 results.csv (final이면 results_final.csv)
출력: 화면 요약 (시드 단위 95% 신뢰구간 + 사진 단위 승패·부호 검정). 계산 방식은 10_paired_compare.py와 같다:
 - 시드 단위: 시드별 사진 평균 n개끼리 Welch t 구간 (두 실험의 같은 번호 시드는 짝이 아니므로 짝 없는 비교)
 - 사진 단위: 사진마다 시드 평균을 먼저 구한 뒤 X − Y, 좋은 방향으로 이긴 사진 수"""
import sys                 # 명령줄 인자 읽기
import csv                 # CSV 읽기
from math import comb      # 부호 검정용 경우의 수
import numpy as np         # 배열 계산
from scipy import stats    # t 분포 (신뢰구간)
import config as cfg       # 프로젝트 설정 (exp_dir)

METRICS = [("psnr", +1), ("ssim", +1), ("lpips", -1)]   # (열 이름, 좋은 방향): +1 클수록 좋음, -1 작을수록 좋음

# ---- 인자 읽기 ----
USAGE = "사용법: python 11_compare_runs.py <실험1>:<조건> <실험2>:<조건> [final]"
args = sys.argv[1:]                                     # 파일 이름을 뺀 인자들
final = bool(args) and args[-1] == "final"              # 마지막 인자가 final이면 최종 확인 결과를 읽음
if final:
    args = args[:-1]                                    # final을 빼고 나머지 = 비교할 두 그룹
assert len(args) == 2 and all(":" in a for a in args), USAGE   # 정확히 두 개, 각각 '실험:조건' 형식
specs = [tuple(a.split(":")) for a in args]             # [("coh_strat", "B"), ("baseline", "B")]
fname = "results_final.csv" if final else "results.csv"   # 읽을 결과 파일 이름


def load(exp, method):
    """한 실험·조건의 결과를 (시드, 사진, 지표) 배열로 읽는다.
    입력: exp = 실험 이름, method = "A"/"B"/"C"
    출력: (vals, seeds, images)
          vals   = (시드 수, 사진 수, 3) float 배열 [psnr, ssim, lpips]
          seeds  = 시드 번호 리스트(정수, 오름차순), images = (세트, 사진 이름) 리스트(정렬)"""
    path = cfg.exp_dir(exp) / fname                     # 결과 파일 경로
    assert path.exists(), f"{path} 가 없습니다. 08_evaluate.py {exp}{' final' if final else ''} 를 먼저 실행하세요"
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["method"] == method]   # 이 조건의 행만
    assert rows, f"{exp}에 조건 {method}의 결과가 없습니다"
    seeds = sorted({int(r["seed"]) for r in rows})      # 등장한 시드들
    images = sorted({(r["set"], r["image"]) for r in rows})   # 등장한 사진들
    vals = np.full((len(seeds), len(images), 3), np.nan)   # nan으로 채운 빈 배열 (빠진 값을 찾기 위해)
    for r in rows:                                      # 행마다 제자리에 값 넣기
        vals[seeds.index(int(r["seed"])), images.index((r["set"], r["image"]))] = [float(r[m]) for m, _ in METRICS]
    assert not np.isnan(vals).any(), f"{exp}:{method} 에 빠진 (시드, 사진) 조합이 있습니다"   # 모든 시드가 모든 사진을 평가했는지
    return vals, seeds, images


def welch_ci(a, b, level=0.95):
    """짝 없는 두 집단의 평균 차이와 Welch t 신뢰구간. 입력 a, b: (n,) / 출력: (차이, 하한, 상한)"""
    d = a.mean() - b.mean()                             # 평균 차이
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)   # 각 평균의 분산 추정
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))   # Welch–Satterthwaite 자유도
    t = stats.t.ppf(0.5 + level / 2, df)                # 양측 임계값
    se = np.sqrt(va + vb)                               # 차이의 표준오차
    return d, d - t * se, d + t * se


def sign_test_p(wins, losses):
    """양측 부호 검정 p값 (동점 제외). 10_paired_compare.py와 같은 함수"""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


(X, sx, ix), (Y, sy, iy) = load(*specs[0]), load(*specs[1])   # *specs[0]: 튜플을 풀어서 (exp, method) 두 인자로 넘김
assert ix == iy, "두 그룹이 평가한 사진 목록이 다릅니다"         # 같은 사진끼리만 비교 가능
nx, ny = f"{specs[0][0]}:{specs[0][1]}", f"{specs[1][0]}:{specs[1][1]}"   # 출력용 이름
print(f"{nx} (시드 {len(sx)}개) − {ny} (시드 {len(sy)}개) | 파일 {fname}")
if len(sx) != len(sy):
    print("  주의: 두 그룹의 시드 수가 다릅니다")

sets_present = sorted({s for s, _ in ix})               # 결과에 있는 세트 이름들
groups = {"Set5": ["Set5"], "Set14": ["Set14"], "전체(Set5+Set14)": ["Set5", "Set14"]}   # 표 이름 -> 세트들 (10과 같은 구성)
if "BSD100" in sets_present:
    groups["BSD100 (독립 확인)"] = ["BSD100"]          # final이면 독립 세트도 따로

for label, names in groups.items():                     # 표마다
    cols = [k for k, (s, _) in enumerate(ix) if s in names]   # 이 표에 들어갈 사진의 열 번호
    xs, ys = X[:, cols].mean(axis=1), Y[:, cols].mean(axis=1)   # 시드별 사진 평균 (시드 수, 3)
    xi, yi = X[:, cols].mean(axis=0), Y[:, cols].mean(axis=0)   # 사진별 시드 평균 (사진 수, 3)
    print(f"\n[{label}] {len(cols)}장")
    for j, (m, good) in enumerate(METRICS):             # 지표마다
        d, lo, hi = welch_ci(xs[:, j], ys[:, j])        # 시드 단위 차이와 구간
        verdict = ("X 우세" if d * good > 0 else "X 열세") if (lo > 0 or hi < 0) else "0 포함 -> 판단 보류"   # X = 첫 번째 그룹
        dd = (xi[:, j] - yi[:, j]) * good               # 사진별 차이, 좋은 방향을 +로
        win, lose = int((dd > 0).sum()), int((dd < 0).sum())   # 이긴 사진 수, 진 사진 수
        print(f"  {m:6s} X 평균 {xs[:, j].mean():.4f} | Y 평균 {ys[:, j].mean():.4f} | 차이 {d:+.4f} "
              f"95% CI [{lo:+.4f}, {hi:+.4f}] {verdict} | 사진 {win}/{len(cols)} 우세, p = {sign_test_p(win, lose):.3f}")
print(f"\n(X = {nx}, Y = {ny})")
