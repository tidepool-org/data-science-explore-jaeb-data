import pandas as pd
import numpy as np
import utils
from pathlib import Path
import math

base_path = Path(__file__).parent
data_path = (
    base_path
    # / "../data/PHI-unique-settings-with-3hr-hysteresis-from-all-data-five-minute-8hr-outcomes-2020-08-19-23-v0-1-0-ed.csv"
    / "../data/t1d_exchange.csv"
).resolve()
df = pd.read_csv(data_path)

# Keys for working with exports
""" Jaeb """
# tdd_key = "total_daily_dose_avg"
# basal_key = "total_daily_basal_insulin_avg"  # Total daily basal
# carb_key = "total_daily_carb_avg"  # Total daily CHO
# bmi_key = "bmi_at_baseline"
# bmi_percentile = "bmi_perc_at_baseline"
# isf_key = "isf"
# icr_key = "carb_ratio"
# age_key = "age_at_baseline"
# tir_key = "percent_70_180_2week"

# keys = [tdd_key, basal_key, carb_key, bmi_key, bmi_percentile, isf_key, icr_key, age_key, tir_key]

""" T1D Exchange """
# tdd_key = "total_daily_dose_avg"
# basal_key = "total_daily_basal_insulin_avg"  # Total daily basal
carb_key = "total_daily_carb_avg"  # Total daily CHO
bmi_key = "bmi"
isf_key = "InsulinSensitivity"
icr_key = "InsulinCarbRatio"

keys = [isf_key, icr_key, carb_key, bmi_key]

""" Print out distributions for all the different columns """
for key in keys:
    print(df[key].describe())
    print()

relevent_data = df[keys]
