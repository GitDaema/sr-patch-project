import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout
import sys
import PIL

import config
from srcnn import to_tensor
from my_psnr import my_per_patch_psnr

IS_AA_ON = False
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

def get_diff_info(name, patch_orig, patch_resized, scores):
    my_scores = my_per_patch_psnr(patch_orig, patch_resized)
    score_diff = my_scores - torch.tensor(scores, device=my_scores.device)
    same_scores = torch.isclose(score_diff, torch.tensor(0.0, dtype=score_diff.dtype), atol=1e-7)
    same_scores_count = same_scores.sum().item()
    max_diff = score_diff.abs().max().item()
    return (name, same_scores_count, max_diff)

verification_dir = Path('verification/p02_split')
verification_dir.mkdir(parents = True, exist_ok = True)

scores = np.load(config.OUT_DIR / 'scores.npy', mmap_mode='r')

patches = np.load(config.OUT_DIR / "patches.npy", mmap_mode='r')
patch_orig = to_tensor(patches)


patch_resized = F.interpolate(patch_orig, scale_factor = 0.25, mode = 'bicubic', 
                              align_corners = IS_AC_ON, antialias = IS_AA_ON)
patch_resized = my_post_processing(patch_resized, AFTER_DOWN_POST_MODE)

patch_resized = F.interpolate(patch_resized, scale_factor = 4, mode = 'bicubic',
                              align_corners = IS_AC_ON)
patch_resized = my_post_processing(patch_resized, AFTER_UP_POST_MODE)

diff_info = get_diff_info('torch', patch_orig, patch_resized, scores)

'''
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

pil_diff_info = get_diff_info('Pillow', patch_orig, patch_resized, scores)
print(pil_diff_info)
'''

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
        print('앤티에일리어싱:', IS_AA_ON, '| align_corners:', IS_AC_ON,
            '| 축소 직후 처리:', AFTER_DOWN_POST_MODE, '| 확대 직후 처리:', AFTER_UP_POST_MODE)
        lib, s_count, max_diff = diff_info 
        print('라이브러리:', lib, '| 정답 수:', s_count, '| 최대 차이:', max_diff)