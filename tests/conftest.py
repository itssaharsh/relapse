import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle"))   # the helpers every Kaggle kernel ran
sys.path.insert(0, os.path.join(ROOT, "scripts"))
