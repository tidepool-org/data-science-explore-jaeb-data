import pandas as pd
import utils
from pathlib import Path
import math


input_file_name = "PHI-filtered-subjects-80-100"
data_path = utils.find_full_path(input_file_name, ".csv")

tir = "80 - 100%"
df = pd.read_csv(data_path)

df = df[(df.basal_total_daily_geomean > 1)]


def basal_eq(tdd, carbs):
    return 0.6507 * tdd * math.exp(-0.001498 * carbs)


def isf_eq(tdd, bmi):
    return 40250 / (tdd * bmi)


def icr_eq(tdd, carbs):
    return (1.31 * carbs + 136.3) / tdd


# Keys for working with Jason's exports
tdd_key = "total_daily_dose_avg"
basal_key = "total_daily_basal_insulin_avg"  # Total daily basal
carb_key = "total_daily_carb_avg"  # Total daily CHO
bmi_key = "bmi_at_baseline"
bmi_percentile = "bmi_perc_at_baseline"
isf_key = "isf"
icr_key = "carb_ratio"
age_key = "age_at_baseline"
tir_key = "percent_70_180_2week"

""" Basal Analysis """
df["predicted_basals"] = df.apply(lambda x: basal_eq(x[tdd_key], x[carb_key]), axis=1)
df.dropna(subset=["predicted_basals"])

basal_residual = df[basal_key] - df["predicted_basals"]
# utils.two_dimension_plot(df[basal_key], basal_residual, ["Basal", "Residual"], tir + " TIR", [-40, 40])

""" ISF Analysis """
df["predicted_isf"] = df.apply(lambda x: isf_eq(x[tdd_key], x[bmi_key]), axis=1)
df["1800_isf"] = df.apply(lambda x: 1800 / x[tdd_key], axis=1)
isf_residual = df[isf_key] - df["predicted_isf"]
# utils.two_dimension_plot(df[isf_key], isf_residual, ["ISF", "Residual"], tir + " TIR", [-150, 150])
# utils.two_dimension_plot(df[carb_key], isf_residual, ["ISF", "Residual"])

""" ICR Analysis """
df["predicted_icr"] = df.apply(lambda x: icr_eq(x[tdd_key], x[carb_key]), axis=1)
df["1800_icr"] = df.apply(lambda x: 500 / x[tdd_key], axis=1)
# utils.two_dimension_plot(df[icr_key], df[icr_key] - df["predicted_icr"], ["ICR", "Residual"], tir + " TIR", [-25, 25])
# utils.two_dimension_plot(df[icr_key], df[icr_key] - df["1800_icr"], ["1800 ICR", "Residual"])

utils.two_dimension_plot(
    df["predicted_basals"] / df[basal_key],
    df["predicted_isf"] / df[isf_key],
    ["Basal Residual Ratio", "ISF Residual Ratio"],
    tir + " TIR",
    [0, 10],
)

""" 3D plot of all residual ratios """
utils.three_dimension_plot(
    df["predicted_basals"] / df[basal_key],
    df["predicted_isf"] / df[isf_key],
    df["predicted_icr"] / df[icr_key],
    ["Basal Residual Ratio", "ISF Residual Ratio", "ICR Residual Ratio"],
    tir + " TIR",
)
