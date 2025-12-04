from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent / "../.."))

from pytracking.evaluation import Tracker

tracker = Tracker('dimp', 'DeT_DiMP50_Max')
tracker.run_vot()
