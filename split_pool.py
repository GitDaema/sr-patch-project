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

verification_dir = Path('verification/p02_split')
verification_dir.mkdir(parents = True, exist_ok = True)

scores = np.load(config.OUT_DIR / 'scores.npy', mmap_mode='r')
median_score = np.nanmedian(scores)

time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
file_name = f'{time_stamp}.txt'
file_path = verification_dir/file_name

scores_med_diff = np.abs(scores - median_score)
min_med_diff = np.min(scores_med_diff)

with open(file_path, "w", encoding="utf-8") as f:
    with redirect_stdout(f):
        print(f"Python: {sys.version}")
        print(f"torch: {torch.__version__}")
        print(f"Numpy: {np.__version__}")
        print(f"Pillow: {PIL.__version__}")
        print("===================\n")
        print('중앙값:', median_score)
        print('중앙값과 가장 가까운 점수까지의 거리:', min_med_diff)
        print('내 점수의 오차가 만드는 최대 차이', 1.42e-05)
        print('B/C 구성이 (가)의 작은 오차에 영향을 받는가? =>', 
            '예' if min_med_diff <= 1.42e-05 else '아니오')
