import pandas as pd
import numpy as np
import scipy
import utils
from pathlib import Path
import matplotlib.pyplot as plt

base_path = Path(__file__).parent
data_path = (
    base_path
    / "../data/PHI-issue-reports-with-surrounding-2week-data-summary-stats-2020-07-28.csv"
).resolve()
df = pd.read_csv(data_path)

# Keys for working with Jason's exports
tdd_key = "insulin_total_daily_geomean"
basal_key = "scheduled_basal_total_daily_insulin_expected"
carb_key = "carbs_total_daily_geomean"
bmi_key = "bmi"
bmi_percentile = "bmiPerc"
isf_key = "insulin_weighted_isf"
icr_key = "carb_weighted_carb_ratio"
age_key = "ageAtBaseline"
tir_key = "percent_70_180"

print(df.bmiPerc.head())
peds = df[
    (df[age_key] < 18)
    & (df[tir_key] > 0)
    & (df.bmiPerc != ".")
]
peds.bmiPerc = peds.bmiPerc.apply(utils.extract_bmi_percentile)

df = df[
    (df[age_key] >= 18)
    & (df[tir_key] > 0)
]

underweight = df[
    (df.bmi < 18.5)
]

normal_weight = df[
    (df.bmi >= 18.5)
    & (df.bmi < 25)
]

overweight = df[
    (df.bmi >= 25)
]

boxplot_data = []
ticks = [str(bmi) for bmi in range(18, 48)]
for bmi in range(18, 48):
    filtered = df[
        (df.bmi >= bmi)
        & (df.bmi < bmi + 1)
    ]
    boxplot_data.append(filtered[tir_key].tolist())

peds_boxplot_data = []
peds_ticks = [str(bmi) for bmi in range(0, 100, 5)]
for perc in range(0, 100, 5):
    filtered = peds[
        (peds.bmiPerc >= perc)
        & (peds.bmiPerc < perc + 5)
    ]
    peds_boxplot_data.append(filtered[tir_key].tolist())
    

""" Distribution plots """
# utils.two_dimension_plot(underweight[bmi_key], underweight[tir_key], ["BMI", "TIR"], "TIR vs BMI: Underweight")
# utils.two_dimension_plot(normal_weight[bmi_key], normal_weight[tir_key], ["BMI", "TIR"], "TIR vs BMI: Normal Weight")
# utils.two_dimension_plot(overweight[bmi_key], overweight[tir_key], ["BMI", "TIR"], "TIR vs BMI: Overweight")
# utils.two_dimension_plot(df[bmi_key], df[tir_key], ["BMI", "TIR"], "TIR vs BMI: Overall")

""" Box and whisker plots """
# utils.box_plot(boxplot_data, ticks, ["BMI", "TIR"], "TIR vs BMI: Adult")
# utils.box_plot(peds_boxplot_data, peds_ticks, ["BMI Percentile", "TIR"], "TIR vs BMI: Child")

""" BMI Distribution for Peds """
# bins = [num for num in range(0, 100, 5)]
# counts = pd.cut(peds.bmiPerc, bins).value_counts().sort_index()
# counts.plot.bar()
# plt.show()

""" BMI Distribution for Adults """
# bins = [num for num in range(10, 48)]
# counts = pd.cut(df.bmi, bins).value_counts().sort_index()
# counts.plot.bar()
# plt.show()

""" Log BMI Distribution for Adults"""
np.log(df.bmi).value_counts().sort_index().plot.bar()
plt.show()
