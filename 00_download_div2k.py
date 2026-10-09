"""DIV2K 학습용 HR 이미지(약 3GB대)를 받아 data/ 아래에 푼다. 한 번만 실행하면 된다.
브라우저로 직접 받아도 된다: http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
직접 받았다면 zip을 풀어서 data/DIV2K_train_HR/0001.png ... 형태가 되게만 하면 이 스크립트는 필요 없다."""
import urllib.request, zipfile, sys   # urllib.request: 인터넷에서 파일 받기 / zipfile: zip 압축 풀기 / sys: 화면 출력 제어
import config as cfg                  # 프로젝트 설정(경로 등)을 cfg라는 이름으로 불러옴

URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"   # DIV2K 공식 배포 주소 (학습용 HR 800장)
zip_path = cfg.ROOT / "data" / "DIV2K_train_HR.zip"                  # 받은 zip 파일을 저장할 위치
zip_path.parent.mkdir(exist_ok=True)                                 # zip이 들어갈 data 폴더를 만든다 (이미 있으면 넘어감)

def progress(block, block_size, total):
    """다운로드 진행률 표시 함수. urlretrieve가 블록을 받을 때마다 자동으로 호출한다.
    입력: block = 지금까지 받은 블록 수, block_size = 블록 하나의 바이트 수, total = 전체 파일 바이트 수(모르면 -1 이하)
    출력: 없음 (화면의 같은 줄에 'xxx / yyy MB'를 덮어써서 보여 줌)"""
    done = block * block_size          # 지금까지 받은 바이트 수 (근사값)
    if total > 0:                      # 서버가 전체 크기를 알려 준 경우에만 표시
        sys.stdout.write(f"\r{done/1e6:8.0f} / {total/1e6:.0f} MB")   # \r: 줄 맨 앞으로 돌아가 같은 줄에 덮어쓰기. 1e6으로 나눠 MB 단위로
        sys.stdout.flush()             # 출력 버퍼를 즉시 화면에 내보냄 (안 하면 늦게 보일 수 있음)

if not zip_path.exists():              # zip을 이미 받아 뒀다면 다시 받지 않는다
    print("다운로드 시작:", URL)
    urllib.request.urlretrieve(URL, zip_path, progress)   # URL의 파일을 zip_path에 저장하면서 progress로 진행률 표시
    print()                                               # 진행률 줄 다음으로 줄바꿈
print("압축 해제 중...")
with zipfile.ZipFile(zip_path) as z:   # zip 파일을 열고, with 블록이 끝나면 자동으로 닫음
    z.extractall(cfg.ROOT / "data")    # data/ 아래에 전부 풀기 -> data/DIV2K_train_HR/0001.png ...
print("완료:", cfg.DIV2K_DIR, "->", len(list(cfg.DIV2K_DIR.glob("*.png"))), "장")   # glob("*.png"): 폴더 안 png 파일 목록. 개수가 800이어야 정상
