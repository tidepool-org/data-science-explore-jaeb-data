import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import utils
import math

data_path = utils.find_full_path("PHI-unique-settings-with-3hr-hysteresis-from-all-data-five-minute-8hr-outcomes-2020-08-19-23-v0-1-0-ed", ".csv")
# data_path = utils.find_full_path("t1d_exchange", ".csv")
df = pd.read_csv(data_path)

""" Keys for working with Jaeb exports """
# tdd_key = "total_daily_dose_avg"
# basal_key = "total_daily_basal_insulin_avg"  # Total daily basal
# carb_key = "total_daily_carb_avg"  # Total daily CHO
# bmi_key = "bmi_at_baseline"
# bmi_percentile = "bmi_perc_at_baseline"
# isf_key = "isf"
# icr_key = "carb_ratio"
# tir_key = "percent_70_180_2week"

""" Keys for working with T1D exchange exports """
# tdd_key = None  # "total_daily_dose_avg" TODO once Jaeb publishes basal data
# basal_key = None  # "total_daily_basal_insulin_avg"  # Total daily basal, TODO once Jaeb publishes basal data
# carb_key = "total_daily_carb_avg"  # Total daily carbs
# bmi_key = "bmi"
# isf_key = "InsulinSensitivity"
# icr_key = "InsulinCarbRatio"
# tir_key = None

""" BMI vs TIR Plot """
# r = range(10, 50, 1)
# bmi_boxplot_data, bmi_ticks = utils.generate_boxplot_data(df, tir_key, r, bmi_key)
# utils.box_plot(bmi_boxplot_data, bmi_ticks, ["BMI", "TIR"], "TIR vs BMI: Overall")

""" TDD """
# log_tdd_key = "log_" + tdd_key
# # Avoid divide by zero error
# df[tdd_key] = df[tdd_key].replace(0, 1)
# df[log_tdd_key] = np.log(df[tdd_key])
# df = df[df[log_tdd_key] > -np.inf]
# # utils.box_plot(df[tdd_key], data_axis_labels=["TDD", ""], title="TDD Distribution: Overall")
# # utils.box_plot(df["log_" + tdd_key], data_axis_labels=["Log TDD", ""], title="Log TDD Distribution: Overall")
# utils.plot_by_frequency(df, tdd_key, title="TDD", x_axis_label="TDD (U)", bins=15, should_export=True, x_lim=[0, 175])
# utils.plot_by_frequency(df, log_tdd_key, title="Log TDD", x_axis_label="Log TDD (U)", bins=15, should_export=True, x_lim=[0, 6])

""" Total daily basal """
# log_basal_key = "log_" + basal_key
# # Avoid divide by zero error
# df[basal_key] = df[basal_key].replace(0, 1)
# df[log_basal_key] = np.log(df[basal_key])
# df = df[df[log_basal_key] > -np.inf]
# utils.plot_by_frequency(df, basal_key, title="Total Daily Basal", x_axis_label="Total Daily Basal (U)", bins=15, should_export=True, x_lim=[0, 160])
# utils.plot_by_frequency(df, log_basal_key, title="Log Total Daily Basal", x_axis_label="Log Total Daily Basal (U)", bins=15, should_export=True, x_lim=[0, 6])

""" Carbs per day """
log_carb_key = "log_" + carb_key
# Avoid divide by zero error
df[carb_key] = df[carb_key].replace(0, 1)
df[log_carb_key] = np.log(df[carb_key])
df = df[df[log_carb_key] > -np.inf]
# utils.box_plot(df[carb_key], data_axis_labels=["CHO Per Day", ""], title="Daily CHO Distribution: Overall")
# utils.box_plot(df[log_carb_key], data_axis_labels=["Log CHO Per Day", ""], title="Log Daily CHO Distribution: Overall")
utils.plot_by_frequency(
    df,
    carb_key,
    title="Daily CHO",
    x_axis_label="Daily CHO (g)",
    bins=15,
    should_export=True,
    x_lim=[0, 500],
)
utils.plot_by_frequency(
    df,
    log_carb_key,
    title="Log Daily CHO",
    x_axis_label="Log Daily CHO (g)",
    bins=30,
    should_export=True,
    x_lim=[0, 7],
)

""" BMI """
# utils.box_plot(df[bmi_key], data_axis_labels=["BMI", ""], title="BMI Distribution: Overall")
# df["log_" + bmi_key] = np.log(df[bmi_key])
# utils.box_plot(df["log_" + bmi_key], data_axis_labels=["Log BMI", ""], title="Log BMI Distribution: Overall")
utils.plot_by_frequency(
    df.dropna(subset=[bmi_key]),
    bmi_key,
    title="BMI",
    x_axis_label="BMI",
    bins=30,
    should_export=True,
    x_lim=[0, 50],
)

""" Insulin Sensitivity Factor """
log_isf_key = "log_" + isf_key
# Avoid divide by zero error
df[isf_key] = df[isf_key].replace(0, 1)
df[log_isf_key] = np.log(df[isf_key])
df = df[df[log_isf_key] > -np.inf]
# utils.box_plot(df[isf_key], data_axis_labels=["ISF", ""], title="ISF Distribution: Overall")
# utils.box_plot(df[log_isf_key], data_axis_labels=["Log ISF", ""], title="Log ISF Distribution: Overall")
utils.plot_by_frequency(
    df,
    isf_key,
    title="ISF",
    x_axis_label="ISF (mg/dL/U)",
    bins=15,
    should_export=True,
    x_lim=[0, 500],
)
utils.plot_by_frequency(
    df,
    log_isf_key,
    title="Log ISF",
    x_axis_label="Log ISF (mg/dL/U)",
    bins=15,
    should_export=True,
    x_lim=[0, 7],
)

""" Insulin to Carb Ratio """
log_icr_key = "log_" + icr_key
# Avoid divide by zero error
df[icr_key] = df[icr_key].replace(0, 1)
df[log_icr_key] = np.log(df[icr_key])
# utils.box_plot(df[icr_key], data_axis_labels=["ICR", ""], title="ICR Distribution: Overall")
# utils.box_plot(df[log_icr_key], data_axis_labels=["Log ICR", ""], title="Log ICR Distribution: Overall")
utils.plot_by_frequency(
    df,
    icr_key,
    title="ICR",
    x_axis_label="ICR (g/U)",
    bins=15,
    should_export=True,
    x_lim=[0, 55],
)
utils.plot_by_frequency(
    df,
    log_icr_key,
    title="Log ICR",
    x_axis_label="Log ICR (g/U)",
    bins=15,
    should_export=True,
    x_lim=[0, 4],
)

""" Time in Range """
# utils.box_plot(df[tir_key], data_axis_labels=["TIR", ""], title="TIR Distribution: Overall")
# df["log_" + tir_key] = np.log(df[tir_key])
# utils.box_plot(df["log_" + tir_key], data_axis_labels=["Log TIR", ""], title="Log TIR Distribution: Overall")
