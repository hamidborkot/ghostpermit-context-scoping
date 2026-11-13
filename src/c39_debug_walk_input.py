"""DEBUG: walk /kaggle/input/datasets/mdhamidborkottulla and list files.

Source: ghost-p-p (5).ipynb, code cell 40 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

import os
for root, dirs, files in os.walk('/kaggle/input/datasets/mdhamidborkottulla'):
    for f in files:
        p = os.path.join(root, f)
        print(p, os.path.getsize(p))
