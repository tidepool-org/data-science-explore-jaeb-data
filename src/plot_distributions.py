import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import utils
from pathlib import Path
import math

base_path = Path(__file__).parent
data_path = (base_path / "../data/PHI-unique-settings-with-3hr-hysteresis-from-all-data-five-minute-8hr-outcomes-2020-08-19-23-v0-1-0-ed.csv").resolve()
df = pd.read_csv(data_path)

# Keys for working with Jason's exports
tdd_key = "insulin_total_daily_geomean"
basal_key = "scheduled_basal_total_daily_insulin_expected"
carb_key = "carbs_total_daily_geomean"
bmi_key = "bmi_at_baseline"
bmi_percentile = "bmi_perc_at_baseline"
isf_key = "insulin_weighted_isf"
icr_key = "carb_weighted_carb_ratio"
age_key = "age_at_baseline"
tir_key = "percent_70_180_2week"

utils.two_dimension_plot(df[bmi_percentile], df[tir_key])

r = range(0, 100, 5)
peds_boxplot_data, peds_ticks = utils.generate_boxplot_data(df, tir_key, r, bmi_key)
utils.box_plot(peds_boxplot_data, peds_ticks, ["BMI Percentile", "TIR"], "TIR vs BMI: Child")

# tdd_range = range(0, 100, 100)
# tdd_boxplot_data, tdd_plot_ticks = utils.generate_boxplot_data(df, tdd_key, tdd_range)
# utils.box_plot(tdd_boxplot_data, tdd_plot_ticks)