
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import utils
from pathlib import Path
from scipy.optimize import curve_fit

base_path = Path(__file__).parent
data_path = (base_path / "../data/PHI-issue-reports-with-surrounding-2week-data-summary-stats-2020-07-23.csv").resolve()
df = pd.read_csv(data_path)

df = utils.filter_data_for_equations(df)
print(df.count)

# % Basals
'''
tdd: insulin_total_daily_geomean
carbs: carbs_total_daily_geomean
basal: basal_total_daily_geomean
'''
labels = ["TDD", "Carbs", "Basal Rate"]
# utils.three_dimension_plot(
#     df["insulin_total_daily_geomean"], 
#     df["carbs_total_daily_geomean"],
#     df["basal_total_daily_geomean"],
#     labels=labels
# )

df["carb_adj_basal"] = df["basal_total_daily_geomean"] / df["carbs_total_daily_geomean"]
df["bmi_adj_basal"] = df["basal_total_daily_geomean"] / df["bmi"]
df["bmi_and_carb_adj_basal"] = df["carb_adj_basal"] / df["bmi"]

# utils.two_dimension_plot(
#     df["basal_total_daily_geomean"],
#     df["bmi_adj_basal"],
#     labels=["Basal", "BMI-Adjusted Basal"]
# )
# utils.two_dimension_plot(
#     df["carbs_total_daily_geomean"],
#     df["bmi_adj_basal"],
#     labels=["Carbs", "BMI-Adjusted Basal"]
# )


# Graph for residuals
utils.two_dimension_plot(
    df["ageAtBaseline"],
    df["bmi_adj_basal"],
    labels=["Age", "BMI-Adjusted Basal"]
)



# plt.scatter(df["ageAtBaseline"], df["bmi_and_carb_adj_basal"])
# plt.plot(df["ageAtBaseline"], exponential(df["ageAtBaseline"], *coeff), linestyle='--', linewidth=2, color='black')
# plt.show()