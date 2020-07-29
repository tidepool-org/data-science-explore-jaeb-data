import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import utils
from pathlib import Path
import math


base_path = Path(__file__).parent
# data_path = (base_path / "../data/PHI-filtered-subjects.csv").resolve()
data_path = (base_path / "../data/PHI-less-ideal-setting-subjects.csv").resolve()
df = pd.read_csv(data_path)

def basal_eq(tdd, carbs):
    return 0.6507 * tdd * math.exp(-0.001498 * carbs)

def bmi_basal_eq(tdd, carbs, bmi):
    return 0.6187139 * tdd * math.exp(-0.001498 * carbs) + 0.06408586 * bmi

def isf_eq(tdd, bmi, age):
    return 5471/(math.log(tdd) * bmi)

def icr_eq(tdd, carbs):
    return (1.31 * carbs + 136.3) / tdd

df = df[
    (df.basal_total_daily_geomean > 1)
    # Normal weight
    # & (df.bmi < 25)
    # & (df.bmi > 10)

    # Overweight
    # & (df.bmi > 25)
    # & (df.bmi < 30)
]

# Keys for working with Jason's exports
tdd_key = "insulin_total_daily_geomean"
basal_key = "scheduled_basal_total_daily_insulin_expected"
carb_key = "carbs_total_daily_geomean"
bmi_key = "bmi"
isf_key = "insulin_weighted_isf"
icr_key = "carb_weighted_carb_ratio"
age_key = "ageAtBaseline"

""" Basal Analysis """
df["predicted_basals"] = df.apply(lambda x: basal_eq(x[tdd_key], x[carb_key]), axis=1)
df["predicted_basals_bmi"] = df.apply(lambda x: bmi_basal_eq(x[tdd_key], x[carb_key], x[bmi_key]), axis=1)
df.dropna(subset=["predicted_basals", "predicted_basals_bmi"])

basal_residual = df[basal_key] - df["predicted_basals"]
utils.two_dimension_plot(df[basal_key], basal_residual, ["Basal", "Residual"])
# utils.two_dimension_plot(df[age_key], basal_residual)

""" ISF Analysis """
df["predicted_isf"] = df.apply(lambda x: isf_eq(x[tdd_key], x[bmi_key], x[age_key]), axis=1)
df["1800_isf"] = df.apply(lambda x: 1800 / x[tdd_key], axis=1)
isf_residual = df[isf_key] - df["predicted_isf"]
print(df.head())
print("ISF RMSE:", utils.rmse(df[isf_key], df["predicted_isf"]))
# utils.two_dimension_plot(df[isf_key], isf_residual, ["ISF", "Residual"])
utils.two_dimension_plot(df[isf_key], df[isf_key] - df["1800_isf"], ["ISF", "1800 Residual"])
# utils.two_dimension_plot(df[carb_key], isf_residual, ["ISF", "Residual"])
# utils.three_dimension_plot(isf_residual, df[tdd_key], df[bmi_key], ["Residual", "TDD", "BMI"])

""" ICR Analysis """
df["predicted_icr"] = df.apply(lambda x: icr_eq(x[tdd_key], x[carb_key]), axis=1)
df["1800_icr"] = df.apply(lambda x: 500 / x[tdd_key], axis=1)
utils.two_dimension_plot(df[icr_key], df[icr_key] - df["predicted_icr"], ["ICR", "Residual"])
# utils.two_dimension_plot(df[icr_key], df[icr_key] - df["1800_icr"], ["1800 ICR", "Residual"])


