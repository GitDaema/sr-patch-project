"""단계 ②-2: 실험 이름에 해당하는 분할 기준(splits.py)으로, 시드별 A/B/C 그룹(각 N개)을 뽑아 실험 폴더에 저장한다.
A = 항상 전체 8000개에서 균등 무작위 (모든 실험 공통). B, C = 실험마다 다른 기준 (splits.py 참고)
사용법:  python 03_make_groups.py <실험이름>          탐색용 시드 3개 (cfg.SEEDS)
         python 03_make_groups.py <실험이름> final    최종 확인용 시드 10개 (cfg.FINAL_SEEDS). 이미 있는 시드는 검증만, 없는 시드만 새로 저장
         실험이름은 splits.py의 SPLITS에 등록된 이름이어야 한다 (예: baseline, coh_strat)
입력: outputs/scores.npy (8000,) float64 PSNR_bic + 기준에 따라 추가 점수 파일
출력: outputs/exp/<실험이름>/groups_seed{시드}.npz  각 파일 안에 A, B, C 세 배열, 각각 (N,) 정수 = 패치 번호(patches.npy의 행 번호)
      outputs/exp/<실험이름>/split_info.txt         분할 기준 설명과 sanity 점검 출력 (실행할 때마다 뒤에 덧붙임)
덮어쓰기 규칙: 그룹 파일이 이미 있으면 절대 다시 쓰지 않는다. 대신 지금 코드로 다시 뽑은 결과가 기존 파일과 같은지 확인하고,
              다르면 멈춘다(= 그 실험의 기준 코드가 처음 만들 때와 달라졌다는 뜻).
A를 뽑는 줄과 그 순서는 바꾸지 않는다(기준 A 모델 재사용의 전제)."""
import sys             # 명령줄 인자(sys.argv) 읽기
import numpy as np     # 배열 계산
import config as cfg   # 프로젝트 설정 (N, SEEDS, FINAL_SEEDS, 경로, exp_dir)
from splits import SPLITS   # 실험 이름 -> 분할 기준 함수 사전

# ---- 실험 이름, 모드, 저장 폴더 ----
USAGE = "사용법: python 03_make_groups.py <실험이름> [final]"   # 잘못 실행했을 때 보여 줄 안내
assert len(sys.argv) in (2, 3), USAGE                       # 인자는 실험 이름 1개, 또는 실험 이름 + final
assert len(sys.argv) == 2 or sys.argv[2] == "final", USAGE  # 세 번째 인자가 있다면 반드시 "final"
exp = sys.argv[1]                           # 실험 이름 문자열
assert exp in SPLITS, f"splits.py의 SPLITS에 없는 실험 이름입니다: {exp} (등록된 이름: {list(SPLITS)})"   # 기준이 등록된 실험만 허용
final = len(sys.argv) == 3                  # True면 최종 확인 모드 (시드 10개)
seeds = cfg.FINAL_SEEDS if final else cfg.SEEDS   # 이번 실행에서 다룰 시드 목록
out_dir = cfg.exp_dir(exp)                  # 이 실험의 저장 폴더 outputs/exp/<exp>
base_dir = cfg.exp_dir(cfg.BASELINE)        # 기준 실험 폴더 (A 동일성 점검용)
out_dir.mkdir(parents=True, exist_ok=True)  # 폴더 생성. parents=True: 부모 폴더(outputs/exp)가 없으면 함께 만듦

log_lines = []                              # split_info.txt에 저장할 출력 줄들을 모아 둘 리스트


def log(msg):
    """메시지를 화면에 출력하고, 나중에 split_info.txt에 쓰도록 log_lines에도 모아 둔다.
    입력: msg = 문자열 / 출력: 없음"""
    print(msg)                  # 화면 출력
    log_lines.append(msg)       # 파일 저장용으로 보관


scores = np.load(cfg.OUT_DIR / "scores.npy")  # (8000,) float64, 패치별 PSNR_bic (모든 실험이 공유)
split = SPLITS[exp](scores)                   # 이 실험의 기준 함수를 실행해 준비 (풀 만들기 등). 결과는 사전 (splits.py 맨 위 설명 참고)

# ---- sanity 점검 1: 기준 설명과 풀 정보 ----
log(f"=== 실행: 실험 {exp} | 모드 {'final (시드 10개)' if final else '탐색 (시드 3개)'} | 시드 {seeds}")
log(split["desc"])                                          # 기준 한 줄 설명
log(f"B = {split['labels']['B']} | C = {split['labels']['C']}")   # 이 실험에서 B, C의 뜻
for line in split["info"]:                                  # 기준이 알려 주는 풀 정보
    log(line)

to_save = {}                                                # 새로 저장할 시드 -> (A, B, C). 점검을 모두 통과한 뒤에 한꺼번에 저장
for seed in seeds:                                          # 이번 모드의 시드 각각에 대해
    rng = np.random.default_rng(seed)                       # 이 시드 전용 난수 생성기
    # 주의: 같은 rng에서 A를 먼저 뽑고 그다음 B, C를 뽑는다. A는 B·C 기준과 무관하게 전체에서 뽑으므로,
    #       기준을 바꿔도 시드가 같으면 A의 인덱스는 항상 같다 (기준 A 모델 재사용의 근거)
    A = rng.choice(len(scores), cfg.N, replace=False)       # 0~7999 중 N개를 중복 없이(replace=False) 무작위로 -> (N,) 정수
    B, C = split["draw"](rng)                               # 이 실험의 기준으로 B, C를 뽑음 (같은 rng를 이어서 사용)

    # ---- sanity 점검 2: 개수와 중복 ----
    assert len(B) == len(C) == cfg.N, "B, C가 N개가 아닙니다"                          # 크기 확인
    assert len(np.unique(B)) == cfg.N and len(np.unique(C)) == cfg.N, "B 또는 C 안에 중복이 있습니다"   # 한 그룹 안 중복 금지
    # ---- sanity 점검 3: 평균 PSNR_bic (모든 기준 공통 참고값) ----
    log(f"seed {seed}: 평균 PSNR_bic  A {scores[A].mean():.2f} | B {scores[B].mean():.2f} | C {scores[C].mean():.2f}")   # scores[A]: A의 패치 번호들로 점수만 골라냄
    # ---- sanity 점검 4: 겹침 (같은 패치 번호가 몇 개 공유되는가) ----
    log(f"         겹침  A-B {np.intersect1d(A, B).size}개 | A-C {np.intersect1d(A, C).size}개 | B-C {np.intersect1d(B, C).size}개")   # intersect1d: 두 배열의 공통 원소
    # ---- sanity 점검 5: 이 기준에 맞는 점검 (splits.py에서 정의) ----
    for line in split["check"](A, B, C):
        log(line)
    # ---- sanity 점검 6: A가 기준 실험의 A와 똑같은가 (다른 실험에서만) ----
    if exp != cfg.BASELINE:                                 # 기준 실험 자신은 비교할 대상이 없으므로 건너뜀
        base_file = base_dir / f"groups_seed{seed}.npz"     # 기준 실험의 같은 시드 그룹 파일
        assert base_file.exists(), f"{base_file} 가 없습니다. 먼저 python 03_make_groups.py {cfg.BASELINE}{' final' if final else ''} 를 실행하세요"
        same = np.array_equal(A, np.load(base_file)["A"])   # 순서까지 포함해 원소가 모두 같으면 True
        log(f"         기준 A와 동일: {same}")
        assert same, "A가 기준 실험과 다릅니다. A를 뽑는 줄이나 그 앞의 난수 호출을 바꾸지 않았는지 확인하세요"   # 다르면 A 모델 재사용이 성립하지 않으므로 저장 전에 멈춤
    # ---- sanity 점검 7: 이미 있는 시드면 기존 파일과 같은가 (덮어쓰지 않고 검증만) ----
    path = out_dir / f"groups_seed{seed}.npz"               # 이 시드의 그룹 파일 경로
    if path.exists():                                       # 이미 있으면
        old = np.load(path)                                 # 기존 파일을 읽어서
        same_old = all(np.array_equal(old[k], v) for k, v in zip("ABC", (A, B, C)))   # A, B, C 세 배열이 모두 같은지 (zip: "A"와 A, "B"와 B, ... 를 짝지음)
        log(f"         기존 파일과 동일: {same_old} (이미 있으므로 저장하지 않음)")
        assert same_old, f"{path} 가 지금 코드로 뽑은 결과와 다릅니다. 기준 코드가 바뀌었다면 새 실험 이름으로 등록하세요"
    else:                                                   # 없으면
        to_save[seed] = (A, B, C)                           # 저장 대기

for seed, (A, B, C) in to_save.items():                     # 모든 점검을 통과했을 때만, 없던 시드만 저장
    np.savez(out_dir / f"groups_seed{seed}.npz", A=A, B=B, C=C)   # 세 배열을 이름 붙여 한 파일에 저장 (07이 [cond]로 꺼냄)
log(f"새로 저장한 시드: {sorted(to_save) or '없음'}")
with open(out_dir / "split_info.txt", "a", encoding="utf-8") as f:   # "a" = 이어 쓰기 모드 (이전 실행 기록을 지우지 않음)
    f.write("\n".join(log_lines) + "\n\n")                           # 이번 실행의 출력 줄을 이어 붙이고 빈 줄로 구분
print("저장 폴더:", out_dir)
