import numpy as np
import config
from srcnn import to_tensor
from my_psnr import my_per_patch_psnr
import torch.nn.functional as F

patches = np.load(config.OUT_DIR / "patches.npy", mmap_mode='r')

patch_sample = patches[range(5)]

patch_orig = to_tensor(patch_sample)

patch_resized = F.interpolate(patch_orig, scale_factor = 0.25, 
                              mode = 'bicubic', antialias = True)

patch_resized = F.interpolate(patch_resized, scale_factor = 4, 
                              mode = 'bicubic', antialias = True)

my_result = my_per_patch_psnr(patch_orig, patch_resized)
print('나만의 score:', my_result[:5].tolist())

scores = np.load(config.OUT_DIR / "scores.npy", mmap_mode='r')
print('score 앞부분:', scores.ravel()[:5])