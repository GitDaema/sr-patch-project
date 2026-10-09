"""단계 ①: HR 이미지에서 PATCH_SIZE x PATCH_SIZE 패치를 무작위로 잘라 한 파일로 저장한다.
출력: outputs/patches.npy    (8000, 96, 96, 3) uint8, 값 0~255  (패치 번호 i = 이미지 순서 x 10 + 이미지 안 순서)
      outputs/patch_meta.npy (8000, 3) 정수, 각 행 = (이미지 번호, x, y) 패치 왼쪽 위 좌표"""
import numpy as np        # 배열 계산 라이브러리. np라는 짧은 이름으로 씀
from PIL import Image     # 이미지 열기·자르기·크기 바꾸기 라이브러리(Pillow)
import config as cfg      # 프로젝트 설정

def random_patches(img, size, n, rng):
    """이미지 1장에서 무작위 위치로 size x size 패치 n개를 자른다.
    입력: img = PIL RGB 이미지, size = 패치 한 변 픽셀 수, n = 자를 개수, rng = numpy 난수 생성기
    출력: out = 길이 n 리스트, 각 원소 (size, size, 3) uint8 배열 / pos = 길이 n 리스트, 각 원소 (x, y) 왼쪽 위 좌표
    같은 이미지에서 뽑은 패치끼리 겹칠 수 있다(겹침을 막지 않음)."""
    W, H = img.size                              # PIL의 size는 (가로, 세로) 순서
    out, pos = [], []                            # 잘라낸 패치와 그 좌표를 담을 빈 리스트 두 개
    for _ in range(n):                           # n번 반복 (_ 는 반복 번호를 쓰지 않는다는 표시)
        x = int(rng.integers(0, W - size + 1))   # 왼쪽 위 x 좌표를 0 ~ W-size 중 무작위로. integers의 끝 값은 포함되지 않으므로 +1
        y = int(rng.integers(0, H - size + 1))   # 왼쪽 위 y 좌표도 같은 방식 (패치가 이미지 밖으로 나가지 않게)
        out.append(np.asarray(img.crop((x, y, x + size, y + size))))   # crop(왼, 위, 오른, 아래)로 자른 뒤 numpy 배열 (size, size, 3)로 변환해 추가
        pos.append((x, y))                       # 좌표 기록
    return out, pos                              # 패치 리스트와 좌표 리스트를 함께 돌려줌

rng = np.random.default_rng(cfg.POOL_SEED)       # 시드를 고정한 난수 생성기 -> 몇 번 실행해도 같은 위치가 잘림 (풀 고정)
files = sorted(cfg.DIV2K_DIR.glob("*.png"))      # 폴더 안 png 파일 목록을 이름순(0001, 0002, ...)으로 정렬. 순서가 바뀌면 풀도 바뀌므로 정렬이 중요
if cfg.MAX_IMAGES:                               # MAX_IMAGES가 None이 아니면(숫자가 들어 있으면)
    files = files[:cfg.MAX_IMAGES]               # 앞에서부터 그 개수만 사용 (빠른 테스트용 슬라이싱)
assert files, f"이미지가 없습니다: {cfg.DIV2K_DIR}"   # 리스트가 비어 있으면 멈춤 (00을 먼저 실행해야 함)

patches, meta = [], []   # meta: (이미지 번호, x, y)  -> 나중에 "이 패치가 어디서 왔나" 추적용
for f in files:                                  # 이미지 파일을 하나씩
    img = Image.open(f).convert("RGB")           # 흑백/알파 채널 이미지도 RGB로 통일
    ps, pos = random_patches(img, cfg.PATCH_SIZE, cfg.PATCHES_PER_IMAGE, rng)   # 이 이미지에서 패치 10개와 좌표 10개
    patches += ps                                # 리스트 이어 붙이기 (전체 패치 리스트에 추가)
    meta += [(int(f.stem), x, y) for x, y in pos]   # f.stem = 확장자 뺀 파일 이름("0001") -> 정수 1. 좌표와 묶어 추가

patches = np.stack(patches)                      # 리스트를 하나의 배열로 쌓음 -> (패치 수, 96, 96, 3) uint8
meta = np.array(meta)                            # (패치 수, 3) 정수 배열
np.save(cfg.OUT_DIR / "patches.npy", patches)    # 이후 단계(02~07)가 모두 이 파일을 읽음
np.save(cfg.OUT_DIR / "patch_meta.npy", meta)
print(f"이미지 {len(files)}장 -> 패치 {len(patches)}개, 배열 모양 {patches.shape}")
