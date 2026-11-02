import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout
import sys
import PIL
from PIL import Image

import config
from srcnn import to_tensor
from my_psnr import my_per_patch_psnr

LIB_MODE = 'Pillow'
IS_AA_ON = True
AFTER_DOWN_POST_MODE = 'round'
AFTER_UP_POST_MODE = 'round'
IS_AC_ON = False

def my_post_processing(patch_resized, post_mode = 'none'):
    if post_mode == 'clip':
        patch_resized = torch.clamp(patch_resized, 0.0, 1.0)
    elif post_mode == 'round':
        patch_resized = torch.clamp(patch_resized, 0.0, 1.0)
        patch_resized = torch.round(patch_resized * 255) / 255
    else:
        post_mode = 'none'
        pass
    return patch_resized

'''
이름 | 모양 | 자료형 | 값 범위
patch_orig | torch.Size([8000, 3, 96, 96]) | torch.float32 | 0.0 ~ 1.0
patch_resized | torch.Size([8000, 3, 96, 96]) | torch.float32 | 0.0 ~ 1.0
scores | (8000,) | float64 | PSNR 참고

반환값: (실험에 쓴 라이브러리 이름, 정답 수, 전부 PSNR 하한값인지 여부(불리언), 
        최대 차이, 평균 절대 차이) 
'''
def get_diff_info(name, patch_orig, patch_resized, scores):
    my_scores = my_per_patch_psnr(patch_orig, patch_resized)
    t_scores = torch.tensor(scores, device=my_scores.device);
    same_scores = torch.isclose(my_scores, t_scores, rtol=0, atol=1e-4)
    same_scores_count = same_scores.sum().item()

    if same_scores_count > 0:
        same_values = my_scores[same_scores]
        target = torch.tensor(148.1308, dtype=my_scores.dtype, device=my_scores.device)
        print(same_values)
        is_all_target = torch.isclose(same_values, target, rtol=0, atol=1e-4).all().item()
    else:
        is_all_target = False

    score_diff = my_scores - t_scores
    
    score_diff_abs = score_diff.abs()
    max_diff = score_diff_abs.max().item()
    avg_diff = score_diff_abs.mean().item()
    return (name, same_scores_count, is_all_target, max_diff, avg_diff)

verification_dir = Path('verification/p02_split')
verification_dir.mkdir(parents = True, exist_ok = True)

scores = np.load(config.OUT_DIR / 'scores.npy', mmap_mode='r')

patches = np.load(config.OUT_DIR / "patches.npy", mmap_mode='r')
patch_orig = to_tensor(patches)

if LIB_MODE == 'torch':
    patch_resized = F.interpolate(patch_orig, scale_factor = 0.25, mode = 'bicubic', 
                                align_corners = IS_AC_ON, antialias = IS_AA_ON)
    patch_resized = my_post_processing(patch_resized, AFTER_DOWN_POST_MODE)

    patch_resized = F.interpolate(patch_resized, scale_factor = 4, mode = 'bicubic',
                                align_corners = IS_AC_ON)
    patch_resized = my_post_processing(patch_resized, AFTER_UP_POST_MODE)
elif LIB_MODE == 'Pillow':
    N, H, W, C = patches.shape
    out_list = []

    for i in range(N):
        img = Image.fromarray(patches[i])
        img_down = img.resize((W // 4, H // 4), resample=Image.Resampling.BICUBIC)
        img_up = img_down.resize((W, H), resample=Image.Resampling.BICUBIC)
        out_list.append(np.array(img_up))

    patch_resized_uint8 = np.stack(out_list, axis=0)

    patch_resized = torch.from_numpy(np.transpose(patch_resized_uint8, 
                                    (0, 3, 1, 2)).astype(np.float32) / 255.0)
else:
    print('사전에 지정하지 않은 라이브러리 모드입니다.')
    sys.exit()

diff_info = get_diff_info(LIB_MODE, patch_orig, patch_resized, scores)
print(diff_info)

time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
file_name = f'{time_stamp}.txt'
file_path = verification_dir/file_name

with open(file_path, "w", encoding="utf-8") as f:
    with redirect_stdout(f):
        print(f"Python: {sys.version}")
        print(f"torch: {torch.__version__}")
        print(f"Numpy: {np.__version__}")
        print(f"Pillow: {PIL.__version__}")
        print("===================\n")
        if LIB_MODE == 'torch':
            print('앤티에일리어싱:', IS_AA_ON, '| align_corners:', IS_AC_ON,
                '| 축소 직후 처리:', AFTER_DOWN_POST_MODE, '| 확대 직후 처리:', AFTER_UP_POST_MODE)
        lib, s_count, is_all_min_psnr, max_diff, avg_diff = diff_info 
        print('라이브러리:', lib, '| 정답 수:', s_count, '| 전부 하한값:', is_all_min_psnr)
        print('최대 차이:', max_diff, '| 평균 절대 차이:', avg_diff)