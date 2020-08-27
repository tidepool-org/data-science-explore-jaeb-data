import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import utils
from pathlib import Path
import math

base_path = Path(__file__).parent
data_path = (
    base_path
    / "../data/PHI-unique-settings-with-3hr-hysteresis-from-all-data-five-minute-8hr-outcomes-2020-08-19-23-v0-1-0-ed.csv"
).resolve()
df = pd.read_csv(data_path)

# Keys for working with exports
tdd_key = "total_daily_dose_avg"
basal_key = "total_daily_basal_insulin_avg"  # Total daily basal
carb_key = "total_daily_carb_avg"  # Total daily CHO
bmi_key = "bmi_at_baseline"
bmi_percentile = "bmi_perc_at_baseline"
isf_key = "isf"
icr_key = "carb_ratio"
age_key = "age_at_baseline"
tir_key = "percent_70_180_2week"

""" BMI vs TIR Plot """
# r = range(10, 50, 1)
# bmi_boxplot_data, bmi_ticks = utils.generate_boxplot_data(df, tir_key, r, bmi_key)
# utils.box_plot(bmi_boxplot_data, bmi_ticks, ["BMI", "TIR"], "TIR vs BMI: Overall")

""" TDD """
# utils.box_plot(df[tdd_key], data_axis_labels=["TDD", ""], title="TDD Distribution: Overall")
# df["log_" + tdd_key] = np.log(df[tdd_key])
# utils.box_plot(df["log_" + tdd_key], data_axis_labels=["Log TDD", ""], title="Log TDD Distribution: Overall")

""" Carbs per day """
# utils.box_plot(df[carb_key], data_axis_labels=["CHO Per Day", ""], title="Daily CHO Distribution: Overall")
# df["log_" + carb_key] = np.log(df[carb_key])
# utils.box_plot(df["log_" + carb_key], data_axis_labels=["Log CHO Per Day", ""], title="Log Daily CHO Distribution: Overall")

""" BMI """
# utils.box_plot(df[bmi_key], data_axis_labels=["BMI", ""], title="BMI Distribution: Overall")
# df["log_" + bmi_key] = np.log(df[bmi_key])
# utils.box_plot(df["log_" + bmi_key], data_axis_labels=["Log BMI", ""], title="Log BMI Distribution: Overall")
# utils.plot_by_frequency(df, bmi_key, title="BMI", x_axis_label="BMI", bins=15)

""" Insulin Sensitivity Factor """
# utils.box_plot(df[isf_key], data_axis_labels=["ISF", ""], title="ISF Distribution: Overall")
# log_isf_key = "log_" + isf_key
# df[log_isf_key] = np.log(df[isf_key])
# df[log_isf_key] = df[df[log_isf_key] > -np.inf]
# utils.box_plot(df[log_isf_key], data_axis_labels=["Log ISF", ""], title="Log ISF Distribution: Overall")
# utils.plot_by_frequency(df, isf_key, title="ISF", x_axis_label="ISF (mg/dL/U)", bins=15)

""" Insulin to Carb Ratio """
# utils.box_plot(df[icr_key], data_axis_labels=["ICR", ""], title="ICR Distribution: Overall")
# df["log_" + icr_key] = np.log(df[icr_key])
# utils.box_plot(df["log_" + icr_key], data_axis_labels=["Log ICR", ""], title="Log ICR Distribution: Overall")

""" Time in Range """
# utils.box_plot(df[tir_key], data_axis_labels=["TIR", ""], title="TIR Distribution: Overall")
# df["log_" + tir_key] = np.log(df[tir_key])
# utils.box_plot(df["log_" + tir_key], data_axis_labels=["Log TIR", ""], title="Log TIR Distribution: Overall")
