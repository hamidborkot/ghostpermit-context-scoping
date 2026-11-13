"""Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True and list /kaggle/input.

Source: ghost-p-p (5).ipynb, code cell 4 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
print(os.listdir("/kaggle/input"))   # verify where your stage-3 dataset mounted
