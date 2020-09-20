import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import utils
from pathlib import Path
import math


data_path = utils.find_full_path("phi-uniq_set-3hr_hyst-2020_08_29_23-v0_1_develop-12c5af2", ".csv")
df = pd.read_csv(data_path)

def aic(k, sum_squared_errors):
    return 2 * k - 2 * np.log(sum_squared_errors)

""" Basal equations """
def jaeb_basal_equation(tdd, carbs):
    return 0.6507 * tdd * math.exp(-0.001498 * carbs)

def traditional_basal_equation(tdd):
    return 0.5 * tdd

""" ISF Equations """
def jaeb_isf_equation(tdd, bmi):
    return 40250 / (tdd * bmi)

def traditional_isf_equation(tdd):
    return 1500 / tdd


""" ICR Equations """
def jaeb_icr_equation(tdd, carbs):
    return (1.31 * carbs + 136.3) / tdd

def traditional_icr_equation(tdd):
    return 500 / tdd

# Keys for working with Jaeb exports
tdd_key = "total_daily_dose_avg"
basal_key = "total_daily_basal_insulin_avg"  # Total daily basal
carb_key = "total_daily_carb_avg"  # Total daily CHO
bmi_key = "bmi_at_baseline"
bmi_percentile = "bmi_perc_at_baseline"
isf_key = "isf"
icr_key = "carb_ratio"
age_key = "age_at_baseline"
tir_key = "percent_70_180_2week"

df = df[
    (df[basal_key] > 1)
    # Normal weight
    # & (df.bmi < 25)
    # & (df.bmi > 10)
    # Overweight
    # & (df.bmi > 25)
    # & (df.bmi < 30)
]

""" Basal Analysis """
df["jaeb_predicted_basals"] = df.apply(lambda x: jaeb_basal_equation(x[tdd_key], x[carb_key]), axis=1)
df["traditional_predicted_basals"] = df.apply(lambda x: traditional_basal_equation(x[tdd_key]), axis=1)

df["jaeb_basal_residual"] = df[basal_key] - df["jaeb_predicted_basals"]
df["traditional_basal_residual"] = df[basal_key] - df["traditional_predicted_basals"]

jaeb_sum_squared_errors = sum(df["jaeb_basal_residual"]**2)
traditional_sum_squared_errors = sum(df["traditional_basal_residual"]**2)

jaeb_basal_aic = aic(2, jaeb_sum_squared_errors)
traditional_basal_aic = aic(1, traditional_sum_squared_errors)

print("Basal: Jaeb", jaeb_basal_aic, "Traditional", traditional_basal_aic)
print("Jaeb - Traditional:", jaeb_basal_aic - traditional_basal_aic)
print()

""" ISF Analysis """
df["jaeb_predicted_isf"] = df.apply(lambda x: jaeb_isf_equation(x[tdd_key], x[bmi_key]), axis=1)
df["traditional_predicted_isf"] = df.apply(lambda x: traditional_isf_equation(x[tdd_key]), axis=1)
df = df.dropna(subset=["jaeb_predicted_isf", "traditional_predicted_isf"])

df["jaeb_isf_residual"] = df[isf_key] - df["jaeb_predicted_isf"]
df["traditional_isf_residual"] = df[isf_key] - df["traditional_predicted_isf"]

jaeb_isf_sum_squared_errors = sum(df["jaeb_isf_residual"]**2)
traditional_isf_sum_squared_errors = sum(df["traditional_isf_residual"]**2)

jaeb_isf_aic = aic(2, jaeb_isf_sum_squared_errors)
traditional_isf_aic = aic(1, traditional_isf_sum_squared_errors)

print("ISF: Jaeb", jaeb_isf_aic, "Traditional", traditional_isf_aic)
print("Jaeb - Traditional:", jaeb_isf_aic - traditional_isf_aic)
print()


""" ICR Analysis """
df["jaeb_predicted_icr"] = df.apply(lambda x: jaeb_icr_equation(x[tdd_key], x[carb_key]), axis=1)
df["traditional_predicted_icr"] = df.apply(lambda x: traditional_icr_equation(x[tdd_key]), axis=1)

df["jaeb_icr_residual"] = df[icr_key] - df["jaeb_predicted_icr"]
df["traditional_icr_residual"] = df[icr_key] - df["traditional_predicted_icr"]

jaeb_icr_sum_squared_errors = sum(df["jaeb_icr_residual"]**2)
traditional_icr_sum_squared_errors = sum(df["traditional_icr_residual"]**2)
print(jaeb_icr_sum_squared_errors, traditional_icr_sum_squared_errors)

jaeb_icr_aic = aic(2, jaeb_icr_sum_squared_errors)
traditional_icr_aic = aic(1, traditional_icr_sum_squared_errors)

print("ICR: Jaeb", jaeb_icr_aic, "Traditional", traditional_icr_aic)
print("Jaeb - Traditional:", jaeb_icr_aic - traditional_icr_aic)