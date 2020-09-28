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

# Seeing if age term improves equation. Conclusion: no.
def equation_without_age(matrix, a):
    # x[0] = tdd, x[1] = bmi
    transpose = zip(*matrix)
    return [a / (x[0] * x[1]) for x in transpose]


def equation_with_age(matrix, a, b):
    # x[0] = tdd, x[1] = bmi, x[2] = age
    transpose = zip(*matrix)
    return [a / (x[0] * x[1]) + b * x[2] for x in transpose]


df = df.dropna(subset=[tdd_key, bmi_key, age_key, isf_key])

tdd = df[tdd_key]
bmi = df[bmi_key]
age = df[age_key]
isf = df[isf_key]

popt, pcov = curve_fit(equation_without_age, [tdd, bmi], isf, p0=[4000])
print(popt)

popt, pcov = curve_fit(equation_with_age, [tdd, bmi, age], isf, p0=[4000, -0.1])
print(popt)
