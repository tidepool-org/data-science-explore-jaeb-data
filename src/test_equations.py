import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import utils
from pathlib import Path
import math


base_path = Path(__file__).parent
data_path = (base_path / "../data/PHI-issue-reports-with-surrounding-2week-data-summary-stats-2020-07-23.csv").resolve()
df = pd.read_csv(data_path)

def basal_eq(tdd, carbs, bmi):
    return 0.641 * tdd ** 1.086 * math.exp(-0.001709 * carbs)

def isf_eq(tdd, bmi, age):
    return 40080/(tdd * bmi)

def icr_eq(tdd, carbs):
    return 11.92 * (0.1069 * carbs + 11.94) / tdd

df = df[
    (df.ageAtBaseline >= 18)
    & (df.basal_total_daily_geomean > 1)
    # Normal weight
    # & (df.bmi < 25)
    # & (df.bmi > 10)

    # Overweight
    & (df.bmi > 25)
    & (df.bmi < 30)
]

tdd_key = "basal_total_daily_geomean"
basal_key = "scheduled_basal_rate_geomean"
carb_key = "carbs_total_daily_geomean"
bmi_key = "bmi"
isf_key = "isf_geomean"
icr_key = "carb_ratio_geomean"
age_key = "ageAtBaseline"

df["predicted_basals"] = df.apply(lambda x: basal_eq(x[tdd_key], x[carb_key], x[bmi_key]), axis=1)
utils.two_dimension_plot(df[basal_key], df[basal_key] - df["predicted_basals"], ["Basal", "Residual"])

df["predicted_isf"] = df.apply(lambda x: isf_eq(x[tdd_key], x[bmi_key], x[age_key]), axis=1)
utils.two_dimension_plot(df[isf_key], df[isf_key] - df["predicted_isf"], ["ISF", "Residual"])
utils.three_dimension_plot(df[isf_key] - df["predicted_isf"], df[tdd_key], df[bmi_key], ["Residual", "TDD", "BMI"])

df["predicted_icr"] = df.apply(lambda x: icr_eq(x[tdd_key], x[carb_key]), axis=1)
print(df[[icr_key, "predicted_icr"]].head())
utils.two_dimension_plot(df[icr_key], df[icr_key] - df["predicted_icr"], ["ICR", "Residual"])


