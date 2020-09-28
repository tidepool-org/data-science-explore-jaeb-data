import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import utils
from pathlib import Path
import math

input_file_name = "Filtered - Adult"
data_path = utils.find_full_path(input_file_name, ".csv")
df = pd.read_csv(data_path)

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


def equation_without_bmi(matrix, a, b):
    # x[0] = tdd, x[1] = carbs
    transpose = zip(*matrix)
    return [a * x[0] * math.exp(b * x[1]) for x in transpose]


def equation_with_bmi(matrix, a, b, c):
    # x[0] = tdd, x[1] = carbs, x[2] = bmi
    transpose = zip(*matrix)
    return [a * x[0] * math.exp(b * x[1]) + c * x[2] for x in transpose]


df = df.dropna(subset=[tdd_key, carb_key, bmi_key, basal_key])

tdd = df[tdd_key]
carbs = df[carb_key]
bmi = df[bmi_key]
basal = df[basal_key]

popt, pcov = curve_fit(equation_without_bmi, [tdd, carbs], basal, p0=[0, -0.001])
print(popt)

popt, pcov = curve_fit(
    equation_with_bmi, [tdd, carbs, bmi], basal, p0=[0, -0.001, 0.05]
)
print(popt)
