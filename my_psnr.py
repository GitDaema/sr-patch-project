import torch
import config
import numpy as np

# MAX_SQ
MAX_SQ = 255 ** 2

# 하한값이 필요하다는 것을 알았을 때, 이것이 결과에 영향을 줄 수 있다는 것을 인지
# 따라서 하한값은 AI가 작성한 코드와 같아야 공정한 비교가 가능하므로 값만 따로 참고했음
MIN_MSE = 1e-10 

# orig와 pred는 각각 원본과 예측 텐서
# srcnn에서 to_tensor 함수로 변환을 거쳤기 때문에 (N, C, H, W), 자료형과 값 범위는 float 0~1
# 반환값은 패치별 PSNR로 쓰이며 (N,) 모양 
def my_per_patch_psnr(orig, pred):

    # 지금까지 생긴 오차는 최대값이 0 ~ 255가 아닌 0 ~ 1이었기 때문
    # ((255 기준의 차이)/255)의 제곱의 분모는 255의 제곱
    # 따라서 다시 255 최댓값의 제곱을 곱해 줘야 255가 기준이 됨
    mse = ((orig.double() - pred.double()) ** 2).mean(dim=(1, 2, 3)) * MAX_SQ
    return mse_to_psnr(mse)

def mse_to_psnr(mse):
    return 10 * torch.log10(MAX_SQ / mse.clamp_min(MIN_MSE))
