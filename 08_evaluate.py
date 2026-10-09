"""단계 ④: bicubic 확대와, 학습된 모델(조건 A/B/C x 시드)을 Set5/Set14에서 평가한다.

평가 방식 (README에 적을 것):
 - HR 사진을 RGB로 읽고(흑백도 RGB로 변환), 가로·세로를 4의 배수로 자른다(남는 가장자리는 버림).
 - LR = PIL bicubic 4배 축소. 모델 입력 = 그것을 bicubic으로 다시 4배 확대 (학습 때와 같은 방식).
 - 모델 출력은 8비트(0~255 정수)로 반올림한 뒤 측정한다.
 - PSNR, SSIM: Y(밝기) 채널에서 계산하고 가장자리 4픽셀은 제외. SSIM은 Wang et al. 2004 방식(가우시안 창 11, sigma 1.5).
 - LPIPS: RGB 전체 사진, alex 네트워크 (처음 실행 시 AlexNet 가중치를 내려받으므로 시간이 걸릴 수 있음).
 - PSNR, SSIM은 높을수록 좋고, LPIPS는 낮을수록 좋다.
사용법:  python 08_evaluate.py <실험이름>          탐색용 시드 3개(cfg.SEEDS)의 모델만 평가 -> results.csv
         python 08_evaluate.py <실험이름> final    최종 확인용 시드 10개(cfg.FINAL_SEEDS)의 모델을 평가 -> results_final.csv
                                                   final 모드는 Set5, Set14에 더해 독립 테스트 세트 BSD100(100장)도 평가한다
 - B, C 모델은 outputs/exp/<실험이름>/models/ 에서, A 모델은 항상 기준 실험(outputs/exp/baseline/models/)에서 불러온다.
 - 두 모드의 결과 파일을 나눈 이유: A는 기준 실험에서 시드 10개까지 늘어나므로, 탐색 실험(시드 3개)과 섞이면 A만 시드 수가 달라진다.
결과: 사진별 값은 outputs/exp/<실험이름>/results.csv 또는 results_final.csv, 요약표는 화면에 출력."""
import sys                                             # 명령줄 인자(sys.argv) 읽기
import csv                                             # CSV(쉼표로 구분된 표) 파일 쓰기
import numpy as np                                     # 배열 계산
import torch                                           # PyTorch
import lpips                                           # LPIPS 지각 거리 라이브러리 (사람 눈 기준 '다르게 보이는 정도')
from PIL import Image                                  # 사진 열기
from skimage.color import rgb2ycbcr                    # RGB -> YCbCr 색 변환 (Y = 밝기)
from skimage.metrics import structural_similarity      # SSIM 계산 함수
import config as cfg                                   # 프로젝트 설정
from srcnn import SRCNN, bicubic_down_up, to_tensor, predict   # 모델과 공통 함수

USAGE = "사용법: python 08_evaluate.py <실험이름> [final]"   # 잘못 실행했을 때 보여 줄 안내
assert len(sys.argv) in (2, 3), USAGE                      # 인자는 실험 이름 1개, 또는 실험 이름 + final
assert len(sys.argv) == 2 or sys.argv[2] == "final", USAGE # 세 번째 인자가 있다면 반드시 "final"
exp = sys.argv[1]                                          # 실험 이름
final = len(sys.argv) == 3                                 # True면 최종 확인 모드
seeds = cfg.FINAL_SEEDS if final else cfg.SEEDS            # 평가할 시드 목록 (이 목록에 있는 시드의 모델만 불러옴)
exp_path = cfg.exp_dir(exp)                                # 이 실험의 폴더
base_path = cfg.exp_dir(cfg.BASELINE)                      # 기준 실험 폴더 (A 모델을 여기서 가져옴)
assert exp_path.exists(), f"{exp_path} 가 없습니다. 03_make_groups.py {exp} 부터 실행하세요"
csv_path = exp_path / ("results_final.csv" if final else "results.csv")   # 이 실험·모드의 평가 결과 파일
assert not csv_path.exists(), f"{csv_path} 가 이미 있습니다. 덮어쓰지 않도록 멈춥니다 (다시 평가하려면 파일을 직접 지우세요)"

device = "cuda" if torch.cuda.is_available() else "cpu"   # GPU가 있으면 GPU 사용
SHAVE = 4                                                  # PSNR·SSIM 계산 때 잘라낼 가장자리 픽셀 수 (경계 효과 제외, SR 논문 관례)
SETS = {"Set5": cfg.ROOT / "data" / "Set5", "Set14": cfg.ROOT / "data" / "Set14"}   # 세트 이름 -> 폴더 경로
if final:                                                  # 최종 확인 모드에서만 독립 테스트 세트 BSD100(100장)을 추가
    SETS["BSD100"] = cfg.ROOT / "data" / "BSD100"          # 탐색 단계에서 보지 않아야 '처음 보는 사진'으로서 의미가 있으므로 final에서만 사용
lp = lpips.LPIPS(net="alex").to(device).eval()             # AlexNet 기반 LPIPS 측정기를 만들어 GPU에 올리고 평가 모드로


def load_hr(path):
    """사진 파일을 읽어 정답(HR) 배열로 만든다.
    입력: path = 이미지 파일 경로 / 출력: (H, W, 3) uint8, H와 W는 4의 배수로 잘린 크기"""
    hr = np.asarray(Image.open(path).convert("RGB"))        # bridge.png처럼 흑백(L)인 사진도 RGB로
    h, w = hr.shape[0] // 4 * 4, hr.shape[1] // 4 * 4        # 4의 배수로 자르기 (comic, flowers, ppt3, zebra 등). 예: 361 // 4 * 4 = 360
    return np.ascontiguousarray(hr[:h, :w])                  # 위·왼쪽부터 h x w만 남김. 슬라이싱 결과를 메모리상 연속 배열로 복사 (일부 라이브러리가 요구)


def luma(img):
    """uint8 RGB -> Y(밝기) 채널. 값 범위 16~235 (SR 논문에서 흔히 쓰는 방식).
    입력: (H, W, 3) uint8 / 출력: (H, W) float64"""
    return rgb2ycbcr(img)[..., 0]   # (H, W, 3) YCbCr 결과에서 마지막 축의 0번(Y)만. ... 는 '앞의 축은 전부'라는 뜻


def measure(sr, hr):
    """복원 결과 하나를 정답과 비교해 세 지표를 구한다.
    입력: sr = 복원 사진, hr = 정답 사진. 둘 다 (H, W, 3) uint8, 같은 크기
    출력: (PSNR dB, SSIM 0~1, LPIPS) 실수 3개 튜플"""
    ys = luma(sr)[SHAVE:-SHAVE, SHAVE:-SHAVE]   # 복원의 Y 채널에서 상하좌우 4픽셀씩 잘라냄 (-4 = 끝에서 4번째 앞까지)
    yh = luma(hr)[SHAVE:-SHAVE, SHAVE:-SHAVE]   # 정답도 똑같이
    mse = ((ys - yh) ** 2).mean()               # 평균 제곱 오차
    psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-10))   # PSNR (최대값 255 기준, 0으로 나누기 방지)
    ssim = structural_similarity(ys, yh, data_range=255, win_size=11,                       # data_range: 값의 범위 폭. win_size: 창 크기
                                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False)   # 가우시안 가중 창 + 표본이 아닌 모분산 = Wang 2004 방식
    with torch.no_grad():                                    # LPIPS는 입력 범위가 -1~1
        a = to_tensor(sr[None]).to(device) * 2 - 1           # sr[None]: 맨 앞에 축 추가 (1, H, W, 3) -> 텐서 (1, 3, H, W) 0~1 -> -1~1
        b = to_tensor(hr[None]).to(device) * 2 - 1           # 정답도 같은 변환
        lp_val = lp(a, b).item()                             # LPIPS 거리 (낮을수록 비슷). 텐서 -> 숫자
    return float(psnr), float(ssim), lp_val                  # numpy 실수를 파이썬 실수로 바꿔 반환


# ---- 학습된 모델 불러오기 (있는 것만) ----
models = {}                                                  # (조건, 시드) -> 모델 사전
for cond in "ABC":                                           # 문자열을 반복하면 글자 하나씩: "A", "B", "C"
    for seed in seeds:                                       # 탐색 모드면 0~2, final 모드면 0~9
        src = base_path if cond == "A" else exp_path         # A는 항상 기준 실험의 모델, B·C는 이 실험의 모델
        path = src / "models" / f"srcnn_{cond}_seed{seed}.pt"   # 07이 저장한 가중치 파일 경로
        if path.exists():                                    # 파일이 있을 때만 (아직 학습 안 한 조합은 건너뜀)
            m = SRCNN().to(device)                           # 빈 모델(같은 구조)을 만들고
            m.load_state_dict(torch.load(path, map_location=device))   # 저장된 가중치를 채워 넣음. map_location: 불러올 장치 지정
            models[(cond, seed)] = m.eval()                  # 평가 모드로 바꿔 사전에 저장
print(f"실험 {exp} | 불러온 모델:", [f"{c}{s}" for c, s in sorted(models)] or "없음 (bicubic만 평가)")   # 리스트가 비어 있으면 or 뒤 문자열 출력
if exp != cfg.BASELINE:                                      # 새 실험이면 A의 출처를 분명히 밝힘
    print(f"  (A 모델은 기준 실험 '{cfg.BASELINE}'의 것을 재사용)")

# ---- 사진마다 모든 방법을 평가 ----
rows = []   # (방법, 시드, 세트, 사진, PSNR, SSIM, LPIPS)    bicubic의 시드는 -1로 표시
for set_name, folder in SETS.items():                        # 세트별로
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))   # 폴더 안 이미지 파일만 이름순으로
    assert files, f"{folder} 에 사진이 없습니다. 06_download_benchmarks.py를 먼저 실행하세요"
    for path in files:                                       # 사진 한 장씩
        hr = load_hr(path)                                   # 정답 (H, W, 3) uint8
        inp = bicubic_down_up(hr)                            # 4배 축소 후 bicubic 확대
        outputs = {("bicubic", -1): inp}                     # 방법별 복원 결과 사전. bicubic 결과 자체가 기준선
        x = to_tensor(inp[None]).to(device)                  # 모델 입력 (1, 3, H, W) float 0~1
        for (cond, seed), m in models.items():               # 불러온 모델마다
            out = predict(m, x)[0].permute(1, 2, 0).cpu().numpy()   # 추론 -> 첫 장 (3,H,W) -> (H,W,3)로 축 재배치 -> CPU numpy. float 0~1
            outputs[(cond, seed)] = np.round(out * 255).astype(np.uint8)   # 0~255로 늘려 반올림 후 8비트 정수로 (실제 저장되는 이미지와 같은 조건)
        for (method, seed), sr in outputs.items():           # 모든 방법의 결과를 측정
            rows.append((method, seed, set_name, path.stem, *measure(sr, hr)))   # *measure(...): 튜플 3개 값을 풀어서 이어 붙임
    print(f"[{set_name}] {len(files)}장 평가 완료")

with open(csv_path, "w", newline="", encoding="utf-8") as f:   # 쓰기 모드("w")로 열기 (위에서 기존 파일이 없음을 확인함). newline="": 윈도우에서 빈 줄 생기는 것 방지
    w = csv.writer(f)                                                             # CSV 쓰기 도구
    w.writerow(["method", "seed", "set", "image", "psnr", "ssim", "lpips"])       # 머리글 한 줄
    w.writerows(rows)                                                             # 결과 행 전부
print("저장:", csv_path)


# ---- 요약표: 시드별로 사진 평균을 먼저 구하고, 그 값들의 평균 ± 표준편차 ----
def print_table(title, set_names):
    """세트별 요약표를 화면에 출력한다.
    입력: title = 표 제목, set_names = 포함할 세트 이름 리스트 (예: ["Set5", "Set14"]). 전역 변수 rows를 읽음
    출력: 없음. 방법마다 '시드별 사진 평균'의 평균 ± 표본 표준편차(ddof=1)"""
    n_img = len({(r[2], r[3]) for r in rows if r[2] in set_names})   # (세트, 사진) 쌍의 집합 크기 = 사진 수
    print(f"\n== {title} ({n_img}장) ==  PSNR·SSIM은 높을수록, LPIPS는 낮을수록 좋음")
    print(f"{'방법':10s}{'PSNR(dB)':>18s}{'SSIM':>20s}{'LPIPS':>20s}")   # :10s = 10칸 왼쪽 정렬, :>18s = 18칸 오른쪽 정렬
    for method in ["bicubic", "A", "B", "C"]:                # 표의 행 순서
        per_seed = []                                        # 시드별 (PSNR, SSIM, LPIPS) 사진 평균을 모을 리스트
        for key in sorted({(r[0], r[1]) for r in rows if r[0] == method}):   # 이 방법의 (방법, 시드) 조합들
            vals = np.array([r[4:7] for r in rows if (r[0], r[1]) == key and r[2] in set_names], dtype=float)   # 해당 시드·세트의 (사진 수, 3) 지표 배열
            per_seed.append(vals.mean(axis=0))               # 이 시드의 사진 평균
        if not per_seed:                                     # 이 방법의 결과가 없으면 (모델 미학습)
            continue                                         # 행을 건너뜀
        per_seed = np.array(per_seed)                        # (시드 수, 3)
        mean = per_seed.mean(axis=0)                         # 시드 평균 (3,)
        std = per_seed.std(axis=0, ddof=1) if len(per_seed) > 1 else None   # 표본 표준편차 (n-1로 나눔). 시드 1개면(bicubic) 없음
        cells = [f"{mean[i]:.{2 if i == 0 else 4}f}" + (f" ± {std[i]:.{2 if i == 0 else 4}f}" if std is not None else "")   # PSNR은 소수 2자리, 나머지는 4자리
                 for i in range(3)]
        label = method if method == "bicubic" else f"{method} (시드 {len(per_seed)}개)"
        print(f"{label:10s}{cells[0]:>18s}{cells[1]:>20s}{cells[2]:>20s}")


print_table("Set5", ["Set5"])
print_table("Set14", ["Set14"])
print_table("전체", ["Set5", "Set14"])   # 19장 사진 단위 평균 (Set14 비중이 큼). '전체'는 항상 Set5+Set14 (BSD100과 섞지 않음)
if "BSD100" in SETS:                     # final 모드면 독립 테스트 세트 표를 따로 출력
    print_table("BSD100 (독립 확인)", ["BSD100"])
