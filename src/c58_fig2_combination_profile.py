"""Figure 2: combination profile (10 combinations x 5 sources).

Source: ghost-p-p (5).ipynb, code cell 59 of 71.
Output file(s): see the cell body and the README in data/processed/ or data/tests/.
"""

import os, itertools
from math import sqrt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

OUT = "/kaggle/working/figures"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})

def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return 100*(c-h), 100*(c+h)
