import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import utils
from pathlib import Path
import math

base_path = Path(__file__).parent
data_path = (base_path / "../data/Filtered - Adult.csv").resolve()
df = pd.read_csv(data_path)

# Keys for working with Jason's exports
tdd_key = "insulin_total_daily_geomean"
basal_key = "scheduled_basal_total_daily_insulin_expected"
carb_key = "carbs_total_daily_geomean"
bmi_key = "bmi"
isf_key = "insulin_weighted_isf"
icr_key = "carb_weighted_carb_ratio"
age_key = "ageAtBaseline"

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

popt, pcov = curve_fit(equation_without_bmi, [tdd, carbs], basal, p0=[0, -.001])
print(popt)

popt, pcov = curve_fit(equation_with_bmi, [tdd, carbs, bmi], basal, p0=[0, -.001, .05])
print(popt)

