# 코드 지도 (인수인계 메모)

이 프로젝트를 직접 분석하고 변형하기 위한 메모입니다. 공개용 요약(README.md)·상세 보고서(REPORT.md)와는 별개입니다.
작성 기준일: 2026-10-07. 최종 갱신: 2026-10-08 (시드 10개·BSD100·02b/10\~14번 스크립트 반영, 공개 저장소 기준으로 정리).
코드 작성과 실험 실행·분석은 Claude(AI)가 했고, 아래 "누가 정했나" 절과 EXPERIMENT_LOG.md에 결정 주체를 구분해 두었습니다.

## 1. 데이터가 흐르는 길

    DIV2K 학습 이미지 800장
      -> [01]  패치 8000개 (96x96)            outputs/patches.npy (211MB, 공개 저장소에는 없음. 01로 다시 만듦)
      -> [02]  패치별 PSNR_bic 점수           outputs/scores.npy
      -> [02b] 패치별 구조성 점수             outputs/scores_coherence.npy (coh_strat 실험용)
      -> [03]  시드별 A/B/C 그룹 (각 N=1000)  outputs/exp/<실험>/groups_seed{시드}.npz + split_info.txt   (패치 번호 배열)
      -> [07]  조건x시드별 SRCNN 학습          outputs/exp/<실험>/models/srcnn_{A,B,C}_seed{시드}.pt (+ tee로 남긴 train_{조건}{시드}.log)
      -> [08]  평가                            탐색: Set5/Set14 -> results.csv / final: Set5/Set14 + BSD100 -> results_final.csv
      -> [10]\~[14] 분석                        짝 비교·신뢰구간(10, 11), 시각 비교(12), 그룹 점검(13), 경계 세기(14)
    시드: 탐색은 0\~2(cfg.SEEDS), 최종 확인은 0\~9(cfg.FINAL_SEEDS, 03/08/10에 final 인자).
    풀·점수(01, 02, 02b)는 모든 실험이 공유하고, 03 이후만 실험별 폴더(outputs/exp/<실험>/)에 저장된다.
    기준 실험 이름은 baseline. 새 실험의 A는 baseline의 A 모델을 재사용한다(03이 A 동일성을 점검).
    outputs/ 바로 아래의 groups_seed*.npz, results.csv, train_*.log는 경로 분리 전 원본이다. outputs/exp/baseline/의 파일과
    바이트 단위로 같아 공개 저장소에서는 뺐다(.gitignore). outputs/models/는 09가 읽으므로 공개 저장소에 포함한다.
    outputs/exp/baseline/train_{A,B,C}{0,1,2}.log는 cp949 인코딩이다(13이 읽는 원본). 같은 내용의 UTF-8 사본 train_*.utf8.log가 옆에 있다.
    시각화: [04] 최저/최고 8개 그림, [05] 점수 구간별 패치 모습, [09] 복원 결과 비교 사진(경로 분리 전 기준 모델 사용)

## 2. 파일 한 줄 요약

| 파일 | 하는 일 |
|---|---|
| config.py | 모든 설정값(패치 크기, N, 시드, 학습 설정). 바꾸고 싶은 숫자는 대부분 여기 |
| 00_download_div2k.py | DIV2K 학습용 HR 이미지 다운로드 |
| 01_make_patches.py | 이미지마다 무작위 패치 10개를 잘라 한 파일로 저장 |
| 02_score_patches.py | 식 (1) PSNR_bic 계산. 낮을수록 디테일 많음 |
| 02b_score_structure.py | 손실 가중 구조성(coherence) 점수 -> outputs/scores_coherence.npy, coherence_examples.png (coh_strat 실험용) |
| splits.py | 분할 기준 모음. 실험 이름 -> 기준 함수 (SPLITS 사전). 새 기준은 여기에 함수로 추가 |
| patch_info.py | 패치 출처(원본 이미지 번호) 도우미. n_images(): 서로 다른 원본 이미지 수 |
| 03_make_groups.py | splits.py에서 실험 이름의 기준을 골라 A(전체)/B/C에서 N개씩 뽑기 + sanity 점검: `python 03_make_groups.py <실험> [final]` |
| 04_visualize.py | 점수 최저 8개 / 최고 8개 그림 |
| 05_score_ladder.py | 점수 순위별로 패치가 어떻게 생겼는지 보는 진단 그림 |
| 06_download_benchmarks.py | Set5, Set14, BSD100 HR 이미지 다운로드 (BSD100은 최종 확인 전용 독립 테스트 세트, 08의 final 모드에서만 평가) |
| srcnn.py | SRCNN 모델, bicubic 축소->확대 함수, 텐서 변환, 분할 추론 |
| 07_train.py | (조건, 시드) 하나를 학습: `python 07_train.py B 0 <실험>` (새 실험에서 A는 거부) |
| 08_evaluate.py | bicubic과 모델들을 Set5/Set14(final 모드는 BSD100 포함)에서 PSNR·SSIM·LPIPS로 평가: `python 08_evaluate.py <실험> [final]` |
| 09_compare_images.py | HR/bicubic/A/B/C 확대 조각 비교 사진 |
| 10_paired_compare.py | 시드 단위 비교(Welch 95% 신뢰구간) + 이미지별 짝 비교(이긴 장수, 부호 검정 p): `python 10_paired_compare.py <실험> [final]` |
| 11_compare_runs.py | 서로 다른 실험의 그룹끼리 비교 (예: coh_strat:B vs baseline:B): `python 11_compare_runs.py 실험1:조건 실험2:조건 [final]` |
| 12_compare_visual.py | 최종 시각 비교 그림 (사전 등록 규칙으로 사진 선택) -> outputs/compare_final.png |
| 13_group_checks.py | 신뢰성 점검: 그룹별 섞인 변수, 구조성 SIGMA 민감도, 학습 패치 개선폭 |
| 14_sharpness.py | (탐색) 선명도 분석: 결과/정답 경계 세기 비율 (전체, 강한 경계 위, 나머지), 경계 위 MSE, 차이 지도 그림 |

## 3. 다른 시도를 하려면 어디를 건드리나 (한 번에 변수 하나만)

| 바꾸고 싶은 것 | 위치 | 같이 확인할 것 |
|---|---|---|
| 패치 점수의 정의 | 02의 `psnr_bic` | 03은 "낮을수록 디테일"을 가정함. 점수 의미가 반대가 되면 B/C 풀 방향을 뒤집어야 함 |
| 풀을 나누는 기준 | splits.py에 함수 추가 + SPLITS에 이름 등록 | 기존 기준 함수는 고치지 않음(재현·검증용). 03의 A 뽑기 줄은 바꾸지 않음 |
| 패치 크기, 이미지당 패치 수, N | config.py | 패치 크기는 4의 배수. 바꾸면 01부터 다시 실행 |
| 모델 구조 | srcnn.py의 `SRCNN` | 평가(08)는 같은 클래스를 불러 쓰므로 함께 바뀜 |
| 손실, 학습 설정 | 07의 `loss_fn`, config.py의 EPOCHS/LR/BATCH_SIZE/AUGMENT | 3조건에서 반드시 동일하게 |
| 평가 방식 | 08의 `measure` | 문헌 수치와 비교하려면 방식(Y채널, 4px 제외)을 맞춰야 함 |

바꾼 뒤 재실행 범위: 01을 바꾸면 01\~, 02를 바꾸면 02\~, 03을 바꾸면 03\~, 학습 관련을 바꾸면 07\~(9번).

## 4. 누가 정했나 (README의 "내가 내린 결정"에 쓸 때 구분하세요)

**사용자가 정한 것**: 프로젝트 범위(논문 식 (1) PSNR_bic로 패치를 고르는 아이디어만 작은 모델로 확인. 논문 재현 아님), 실험 질문, A/B/C 3조건 설계, 중앙값 기준, 패치 크기는 4의 배수(96), 시드 3회 평균, 지표 PSNR·SSIM·LPIPS, 데이터셋(DIV2K, Set5/Set14), SRCNN류 소형 모델, 조건 간 학습 설정 동일.

**Claude가 제안·선택한 것** (모두 이후 실험에 그대로 쓰였다. 이 절은 baseline 시점의 정리이며, 그 뒤의 결정 주체는 EXPERIMENT_LOG.md의 [사용자]/[Claude] 표기를 본다):
- 패치 풀: 이미지당 10개 무작위, 풀은 POOL_SEED=0으로 고정(시드마다 달라지는 건 그룹 뽑기뿐)
- N=1000, 단색에 가까운 패치(60 dB 이상 34개, 0.43%)는 풀에서 제거하지 않음 (연속적 분포라 기준이 자의적이라는 판단)
- 04 그림에서만 148 dB(완전 단색)를 건너뜀
- 모델: 잔차 학습(입력에 보정만 더함) + 마지막 층 0 초기화, RGB 3채널, 파라미터 약 2만 개
- 학습: Adam, LR 1e-4, 100 에폭, 배치 16, 회전·반전 증강. 학습 MSE만 보고 정함
- 입력 생성: PIL bicubic 4배 축소 후 4배 확대 (학습과 평가 동일)
- 평가: Y(밝기) 채널 PSNR/SSIM(가우시안 창 11, sigma 1.5), 가장자리 4px 제외, 4의 배수로 자름, 8비트 반올림, LPIPS는 alex·RGB 전체, bicubic 기준선 행 추가
- 같은 시드 번호가 그룹 뽑기와 학습의 무작위성(초기화, 배치 순서)을 동시에 결정함

## 5. 알려진 한계와 주의

- 시드 3개(탐색 단계): 표준편차(n=3)는 불안정한 추정치. 최종 후보는 시드 10개로 다시 확인했다(EXPERIMENT_LOG.md "최종 확인 결과"). 평가 사진이 고정이라 ± 값은 "학습 무작위성"만 반영하고 "사진 선택의 불확실성"은 반영하지 않음 (10의 사진 단위 비교가 이를 보완)
- 평가 사진 19장(Set5 5, Set14 14). "전체" 평균은 사진당 가중이라 Set14 비중이 큼. 최종 확인에서만 독립 세트 BSD100(100장)을 추가
- 작은 모델·작은 데이터(패치 1000개). 큰 모델이나 GAN 계열에서도 같은 결론이 나오는지는 모름
- A는 전체 풀에서 뽑으므로 B, C와 일부 겹칠 수 있음
- 같은 시드로 재실행해도 GPU 연산 특성상 미세하게 다를 수 있음(결정론 설정을 하지 않음)
- 코드가 쓰는 bicubic은 PIL 방식(축소 시 안티에일리어싱 포함). OpenCV `INTER_CUBIC`과 다르므로 점수·결과가 달라질 수 있음
- Set5/Set14는 Hugging Face 커뮤니티 미러(eugenesiow)에서 받음. 공식 배포처 아님, 학술 연구용 라이선스
- 07이 출력하는 "학습 후 PSNR"은 학습에 쓴 패치 기준이라 진단용일 뿐, 성능 근거가 아님

## 6. 첫 결과에서 나온 열린 질문 (설명을 확정하지 못한 것)

(baseline 시드 3개 시점의 질문. 상태는 2026-10-08 기준)
- C의 시드 간 표준편차가 A, B보다 훨씬 작음(PSNR ±0.00 대 ±0.05). 이유 미확인. `results.csv`에서 시드별 값 확인 가능 → **여전히 미확인**(C는 시드 3개만 학습)
- B-A 차이가 LPIPS에서는 시드 편차의 약 2배인데 PSNR·SSIM에서는 편차 안에 있음. 사진 단위 짝비교(같은 사진에서 B가 A를 이긴 비율)로 볼 수 있음 → **시드 10개에서 확인**: 19장·BSD100 모두 PSNR·LPIPS 우세, SSIM은 19장 우세·BSD100 보류
- 논문 Table III(Real-ESRGAN)에서는 임계값 선택의 LPIPS가 무작위보다 나빴으나 이 실험(SRCNN)에서는 B가 더 좋음. 모델이 달라 직접 비교는 불가

## 7. 처음부터 다시 돌리는 순서

    (아래는 baseline과 새 실험의 일반 순서. 이 저장소의 baseline·coh_strat 최종 실행 순서 전체는 REPORT.md "재현 방법" 참고)

    python 01_make_patches.py ; python 02_score_patches.py ; python 03_make_groups.py baseline ; python 04_visualize.py
    for c in A B C; do for s in 0 1 2; do python -u 07_train.py $c $s baseline | tee outputs/exp/baseline/train_${c}${s}.log; done; done
    python 08_evaluate.py baseline ; python 09_compare_images.py ; python 10_paired_compare.py baseline

    새 실험 (예: 이름 myexp): splits.py에 기준 함수를 추가하고 SPLITS에 "myexp"를 등록한 뒤
    python 03_make_groups.py myexp
    for c in B C; do for s in 0 1 2; do python -u 07_train.py $c $s myexp | tee outputs/exp/myexp/train_${c}${s}.log; done; done
    python 08_evaluate.py myexp
    python 10_paired_compare.py myexp

    최종 후보 확인 (시드 10개, 사전에 판단 규칙을 EXPERIMENT_LOG에 적은 뒤):
    python 03_make_groups.py baseline final ; python 03_make_groups.py myexp final      # 시드 3\~9 그룹 추가 (0\~2는 검증만)
    for s in 3 4 5 6 7 8 9; do python -u 07_train.py A $s baseline | tee outputs/exp/baseline/train_A${s}.log; done   # A 확장은 처음 한 번만
    for s in 3 4 5 6 7 8 9; do python -u 07_train.py B $s myexp | tee outputs/exp/myexp/train_B${s}.log; done
    python 08_evaluate.py myexp final ; python 10_paired_compare.py myexp final         # results_final.csv, 시드 단위 95% 신뢰구간
    (07/08은 같은 이름의 결과가 이미 있으면 덮어쓰지 않고 멈춘다. 03은 이미 있는 시드는 다시 뽑아 기존 파일과 같은지 검증만 하고, 다르면 멈춘다)
