"""패치의 출처(어느 원본 이미지에서 왔는지) 정보를 다루는 작은 도우미."""
import numpy as np     # 배열 계산
import config as cfg   # 프로젝트 설정 (OUT_DIR)

IMG_ID = np.load(cfg.OUT_DIR / "patch_meta.npy")[:, 0]   # (8000,) 정수. i번 패치가 나온 DIV2K 이미지 번호(1~800). [:, 0] = 모든 행의 0번 열


def n_images(idx):
    """패치 번호 배열 -> 그 패치들이 나온 서로 다른 원본 이미지 수.
    입력: idx = (n,) 정수 패치 번호 / 출력: 정수 (최대 800)"""
    return len(np.unique(IMG_ID[idx]))   # np.unique: 중복을 없앤 값들. 그 개수 = 서로 다른 이미지 수
