import config
import numpy as np
import torch
import sys
from srcnn import to_tensor
from my_psnr import my_per_patch_psnr
from pathlib import Path
from contextlib import redirect_stdout

def get_next_log_file_name(name="output"):
    i = 1
    while True:
        file_path = Path(f"verification/{name}{i}.txt")
        if not file_path.exists():
            return file_path
        i += 1

# 기존 함수의 복사본
def ai_per_patch_psnr(a, b):
    """패치마다 PSNR을 구한다. 02의 psnr_bic과 같은 공식(255 기준, mse 하한 1e-10).
    입력: a, b = (N, 3, H, W) float 텐서, 값 0~1
    출력: (N,) float64 텐서, 단위 dB"""
    mse255 = ((a.double() - b.double()) ** 2).mean(dim=(1, 2, 3)) * 255 ** 2   # double = 64비트 실수. 채널·세로·가로 축 평균 -> 패치별 mse. 0~1 단위라 255^2를 곱해 0~255 단위로
    return 10 * torch.log10(255 ** 2 / mse255.clamp_min(1e-10))              # clamp_min: 1e-10보다 작은 값은 1e-10으로 (0으로 나누기 방지)


patches = np.load(config.OUT_DIR / "patches.npy", mmap_mode='r')

patch_sample = patches[range(5)]

patch_orig = to_tensor(patch_sample)

patch_pred = patch_orig.clone()

torch.manual_seed(1009) # 시드 고정 후 약한 노이즈
weak_noise = torch.randn_like(patch_pred[0]) * 0.05
patch_pred[0] = torch.clamp(patch_pred[0] + weak_noise, 0.0, 1.0)

torch.manual_seed(1009) # 시드 고정 후 강한 노이즈
strong_noise = torch.randn_like(patch_pred[1]) * 0.5
patch_pred[1] = torch.clamp(patch_pred[1] + strong_noise, 0.0, 1.0)

# 밝기 올리기는 값을 더하는 거니 랜덤 없이 구현 
patch_pred[2] = torch.clamp(patch_pred[2] + 0.4, 0.0, 1.0)

torch.manual_seed(1009)
N, C, H, W = patch_pred.shape
# 검은 박스의 가로 세로는 패치 가로 세로의 1 // 2
black_box_height, black_box_width = H // 2, W // 2

# 검은 박스의 시작 위치(왼쪽 위 꼭짓점)
start_x = torch.randint(0, W - black_box_width + 1, (1,)).item()
start_y = torch.randint(0, H - black_box_height + 1, (1,)).item()

# 시작 위치부터 오른쪽 밑 꼭짓점까지 0으로 망가뜨리기
patch_pred[3, :, start_y:start_y+black_box_height, start_x:start_x+black_box_width] = 0.0

my_result = my_per_patch_psnr(patch_orig, patch_pred)
ai_result = ai_per_patch_psnr(patch_orig, patch_pred)


# 반올림 비교는 원래는 오차 범위 내의 작은 값만 차이나는 두 수도 반올림 경계에 서로 나눠지면
# 큰 차이가 나서 다르다고 판단될 수 있음
# torch.isclose는 두 수의 실제 차이를 구해서 기준값 이하인지 확인하므로 안전
# atol이 절대 오차 경계, rtol이 상대 오차이므로 rtol은 0.0으로 비활성화
is_same = torch.isclose(my_result, ai_result, atol=1e-4, rtol=0.0).all().item()

if is_same:
    output_name = "success"
    message = "통과"
else:
    diff = torch.abs(my_result - ai_result)
    max_diff = torch.max(diff).item()
    output_name = "fail"
    message = f"실패(최대 차이:{max_diff:.4f})"

with open(get_next_log_file_name(output_name), "w", encoding="utf-8") as f:
    with redirect_stdout(f):
        print(f"Python: {sys.version}")
        print(f"PyTorch: {torch.__version__}")
        print(f"numpy: {np.__version__}")
        print("===================\n")
        print(my_result)
        print(ai_result)
        print(message)

