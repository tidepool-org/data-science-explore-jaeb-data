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
