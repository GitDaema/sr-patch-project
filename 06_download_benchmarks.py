"""평가용 벤치마크 Set5(5장), Set14(14장), BSD100(100장)의 HR 이미지를 받아 data/Set5, data/Set14, data/BSD100 에 정리한다.
BSD100은 최종 확인 전용 독립 테스트 세트다 (08의 final 모드에서만 평가).

출처: Hugging Face의 커뮤니티 미러(eugenesiow). 공식 배포처가 아님에 주의.
원 논문: Set5 = Bevilacqua et al. 2012 (BMVC), Set14 = Zeyde et al. 2010, BSD100 = Martin et al. 2001. 라이선스: 학술 연구용.
LR 이미지는 받지 않는다. 학습 때와 똑같은 방식(PIL bicubic 4배 축소)으로 직접 만들어야 비교가 공정하기 때문이다."""
import shutil, tarfile, urllib.request   # shutil: 파일 복사·폴더 삭제 / tarfile: .tar.gz 풀기 / urllib.request: 다운로드
from pathlib import Path                 # 경로 객체 (현재 이 파일에서는 쓰지 않음)
from PIL import Image                    # 받은 이미지의 크기·모드를 확인할 때 사용
import config as cfg                     # 프로젝트 설정

SETS = {   # 이름: (HR 압축파일 주소, 기대 장수)
    "Set5":  ("https://huggingface.co/datasets/eugenesiow/Set5/resolve/main/data/Set5_HR.tar.gz", 5),
    "Set14": ("https://huggingface.co/datasets/eugenesiow/Set14/resolve/main/data/Set14_HR.tar.gz", 14),
    # BSD100: 최종 확인 전용 독립 테스트 세트 (가설 수립·탐색에 쓰지 않음). 원 출처 Martin et al. 2001 (BSD300의 테스트 100장). 같은 커뮤니티 미러
    "BSD100": ("https://huggingface.co/datasets/eugenesiow/BSD100/resolve/main/data/BSD100_HR.tar.gz", 100),
}
EXTS = {".png", ".jpg", ".jpeg", ".bmp"}   # 이미지로 인정할 확장자 집합(set)

def safe_extract(tar, dest):
    """압축 안의 파일이 dest 밖으로 나가거나 링크이면 거부한다.
    입력: tar = 열린 tarfile 객체, dest = 풀 폴더(Path)
    출력: 없음 (dest 아래에 압축을 풂). 위험한 항목이 있으면 RuntimeError
    이유: 악의적인 압축 파일은 '../../' 같은 경로로 엉뚱한 위치에 파일을 쓸 수 있어서 미리 검사한다."""
    dest = dest.resolve()                    # 절대 경로로 바꿈 (비교를 정확하게 하려고)
    for m in tar.getmembers():               # 압축 안의 모든 항목을 하나씩
        if m.issym() or m.islnk() or not str((dest / m.name).resolve()).startswith(str(dest)):   # 심볼릭 링크 / 하드 링크 / dest 밖으로 나가는 경로이면
            raise RuntimeError(f"허용하지 않는 항목: {m.name}")                                  # 오류를 내고 멈춤
    try:
        tar.extractall(dest, filter="data")   # 최신 파이썬 권장 방식 (filter="data": 위험한 항목을 파이썬이 한 번 더 막아 줌)
    except TypeError:                         # filter 인자를 모르는 파이썬이면 TypeError가 나므로
        tar.extractall(dest)                  # filter 인자가 없는 구버전 파이썬

data_dir = cfg.ROOT / "data"     # 데이터 폴더
data_dir.mkdir(exist_ok=True)    # 없으면 만듦

for name, (url, expected) in SETS.items():   # 딕셔너리의 (키, 값) 쌍을 하나씩. 값 튜플은 (url, expected)로 바로 풀어 받음
    out = data_dir / name                    # 최종 폴더: data/Set5, data/Set14 또는 data/BSD100
    if out.exists() and any(out.iterdir()):  # 폴더가 있고 안에 파일이 하나라도 있으면
        print(f"[{name}] 이미 {out} 에 있어 건너뜀 (다시 받으려면 폴더를 지우세요)")
        continue                             # 다운로드하지 않고 다음 세트로
    tgz, raw = data_dir / f"{name}_HR.tar.gz", data_dir / f"{name}_raw"   # 압축 파일 경로, 임시로 풀 폴더 경로
    print(f"[{name}] 다운로드: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})   # 브라우저처럼 보이는 헤더를 붙인 요청 (헤더 없는 요청을 거부하는 서버 대비)
    with urllib.request.urlopen(req) as r, open(tgz, "wb") as f:   # 주소를 열고(r), 저장할 파일을 바이너리 쓰기 모드로 염(f)
        shutil.copyfileobj(r, f)                                   # 받은 내용을 그대로 파일에 복사
    raw.mkdir(exist_ok=True)                 # 임시 폴더 생성
    with tarfile.open(tgz) as t:             # .tar.gz 열기
        safe_extract(t, raw)                 # 안전 검사 후 임시 폴더에 풀기
    out.mkdir(exist_ok=True)                 # 최종 폴더 생성
    files = sorted(p for p in raw.rglob("*") if p.suffix.lower() in EXTS and not p.name.startswith("._"))   # 하위 폴더까지(rglob) 이미지 파일만. "._"로 시작하는 macOS 숨김 파일은 제외
    for p in files:
        shutil.copy(p, out / p.name)         # 폴더 구조 없이 이미지만 최종 폴더로 복사
    shutil.rmtree(raw)                       # 임시 폴더 통째로 삭제

    print(f"[{name}] {len(files)}장 (기대 {expected}장)")
    for p in sorted(out.iterdir()):          # 최종 폴더의 파일을 이름순으로
        with Image.open(p) as im:            # 이미지를 열어서
            print(f"    {p.name:20s} {im.size[0]}x{im.size[1]}  {im.mode}")   # 이름, 가로x세로, 색 모드(RGB/L 등) 출력
    if len(files) != expected:               # 장수가 기대와 다르면 경고
        print(f"    주의: 장수가 기대와 다릅니다. 결과를 가져와 확인하세요.")
