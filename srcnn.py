"""SRCNN 모델과, 학습 입력을 만드는 함수 (단계 ③, ④에서 공통으로 사용).
07_train.py, 08_evaluate.py, 09_compare_images.py가 `from srcnn import ...`로 불러 쓴다."""
import numpy as np          # 배열 계산
import torch                # PyTorch: 텐서(GPU에서 계산 가능한 배열)와 자동 미분
import torch.nn as nn       # 신경망 층(합성곱 등)을 모아 둔 모듈
from PIL import Image       # bicubic 축소·확대


class SRCNN(nn.Module):
    """아주 작은 SR 모델 (파라미터 약 2만 개). 구조: 9x9 합성곱 -> 1x1 -> 5x5.
    입력은 '4배 축소했다가 bicubic으로 다시 키운 이미지'이고, 출력은 그것을 정답(HR)에 가깝게 고친 이미지.
    입력·출력 모양: (배치, 3, H, W) float32, 값 범위 대략 0~1. padding 때문에 H, W는 바뀌지 않는다."""

    def __init__(self):
        """층을 만들고 마지막 층을 0으로 초기화한다. 입력 없음."""
        super().__init__()                   # 부모 클래스(nn.Module)의 초기화를 먼저 실행 (PyTorch 모델의 필수 규칙)
        self.body = nn.Sequential(           # 아래 층들을 순서대로 이어 붙인 하나의 블록
            nn.Conv2d(3, 64, kernel_size=9, padding=4), nn.ReLU(inplace=True),   # RGB 3채널 -> 특징 64채널, 9x9 창. padding=4로 크기 유지. ReLU: 음수를 0으로
            nn.Conv2d(64, 32, kernel_size=1), nn.ReLU(inplace=True),             # 64 -> 32채널, 1x1 (픽셀마다 채널만 섞음)
            nn.Conv2d(32, 3, kernel_size=5, padding=2),                          # 32 -> 3채널(RGB 보정값), 5x5 창. 마지막이라 ReLU 없음 (보정은 음수도 가능)
        )
        # 마지막 층을 0으로 시작 -> 학습 전에는 입력을 그대로 내보낸다(= bicubic과 똑같은 출력)
        nn.init.zeros_(self.body[-1].weight)   # body[-1] = 마지막 층(5x5 합성곱)의 가중치를 전부 0으로
        nn.init.zeros_(self.body[-1].bias)     # 편향도 0으로

    def forward(self, x):
        """입력 x: (배치, 3, H, W) float, 0~1 -> 출력: 같은 모양. 모델을 model(x)로 부르면 이 함수가 실행된다."""
        return x + self.body(x)   # 입력에 '고쳐야 할 부분'만 더하는 방식(잔차 학습)


def bicubic_down_up(hr):
    """(H, W, 3) uint8 -> (H, W, 3) uint8. 4배 축소 후 4배 확대. 02_score_patches.py의 psnr_bic 안에서 하는 것과 같은 연산.
    H, W가 4의 배수가 아니면 원래 크기로 정확히 돌아오지 않으므로, 호출하는 쪽에서 4의 배수로 맞춰 둔다."""
    im = Image.fromarray(hr)                          # numpy 배열 -> PIL 이미지
    w, h = im.size                                    # PIL은 (가로, 세로) 순서
    lr = im.resize((w // 4, h // 4), Image.BICUBIC)   # 1/4로 축소 (PIL 방식: 안티에일리어싱 포함)
    return np.asarray(lr.resize((w, h), Image.BICUBIC))   # 원래 크기로 다시 bicubic 확대한 뒤 numpy 배열로 (uint8)


def to_tensor(arr):
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float, 값 범위 0~1.
    PyTorch 합성곱은 채널이 두 번째 축에 오는 순서(N, C, H, W)를 쓰기 때문에 축 순서를 바꾼다."""
    return torch.from_numpy(np.array(arr)).permute(0, 3, 1, 2).float() / 255.0   # numpy 배열을 복사해 텐서로 -> 축 순서 (0,3,1,2)로 재배치 -> 실수로 -> 255로 나눠 0~1


@torch.no_grad()   # 이 함수 안에서는 기울기(학습용 미분 정보)를 기록하지 않음 -> 메모리 절약, 평가 전용
def predict(model, x, chunk=50):
    """x를 chunk개씩 나눠 모델에 통과시킨다 (한꺼번에 넣으면 GPU 메모리가 부족할 수 있음). 출력은 0~1로 자른다.
    입력: model = SRCNN, x = (N, 3, H, W) float 텐서 (model과 같은 장치에 있어야 함), chunk = 한 번에 넣을 개수
    출력: (N, 3, H, W) float 텐서, 값 0~1"""
    outs = [model(x[i:i + chunk]).clamp(0, 1) for i in range(0, len(x), chunk)]   # 0, 50, 100, ... 위치에서 chunk개씩 잘라 통과. clamp: 0 미만은 0, 1 초과는 1로
    return torch.cat(outs)   # 조각 결과들을 첫 번째 축(배치)으로 다시 이어 붙임
