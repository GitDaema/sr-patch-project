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

with open(file_path, "w", encoding="utf-8") as f:
    with redirect_stdout(f):
        print(f"Python: {sys.version}")
        print(f"torch: {torch.__version__}")
        print(f"Numpy: {np.__version__}")
        print(f"Pillow: {PIL.__version__}")
        print("===================\n")
        print('중앙값:', median_score)
