import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import utils
from pathlib import Path
import math

base_path = Path(__file__).parent
data_path = (base_path / "../data/Filtered - Adult.csv").resolve()
df = pd.read_csv(data_path)

# Keys for working with Jason's exports
tdd_key = "insulin_total_daily_geomean"
basal_key = "scheduled_basal_total_daily_insulin_expected"
carb_key = "carbs_total_daily_geomean"
bmi_key = "bmi"
isf_key = "insulin_weighted_isf"
icr_key = "carb_weighted_carb_ratio"
age_key = "ageAtBaseline"

print(df[tdd_key])
utils.box_plot(list(df[tdd_key]))