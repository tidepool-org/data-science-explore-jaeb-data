import pandas as pd
import numpy as np
import utils
import math

# file_name = "PHI-unique-settings-with-3hr-hysteresis-from-all-data-five-minute-8hr-outcomes-2020-08-19-23-v0-1-0-ed"
file_name = "t1d_exchange"
data_path = utils.find_full_path(file_name, ".csv")
df = pd.read_csv(data_path)

distribution_stats = ["mean", "std", "min", "25%", "75%", "max"]
output_df = pd.DataFrame(columns=distribution_stats)
analysis_name = "analyze-demographics"

# Keys for working with exports
""" Jaeb """
# tdd_key = "total_daily_dose_avg"
# basal_key = "total_daily_basal_insulin_avg"  # Total daily basal
# carb_key = "total_daily_carb_avg"  # Total daily CHO
# bmi_key = "bmi_at_baseline"
# bmi_percentile = "bmi_perc_at_baseline"
# isf_key = "isf"
# icr_key = "carb_ratio"
# age_key = "age_at_baseline"
# tir_key = "percent_70_180_2week"

# keys = [tdd_key, basal_key, carb_key, bmi_key, bmi_percentile, isf_key, icr_key, age_key, tir_key]

""" T1D Exchange """
# tdd_key = "total_daily_dose_avg"
# basal_key = "total_daily_basal_insulin_avg"  # Total daily basal
carb_key = "total_daily_carb_avg"  # Total daily CHO
bmi_key = "bmi"
isf_key = "InsulinSensitivity"
icr_key = "InsulinCarbRatio"

keys = [isf_key, icr_key, carb_key, bmi_key]

""" Print out distributions for all the different columns """
for key in keys:
    row = []
    distribution = df[key].describe(include="all")
    for stat in distribution_stats:
        row.append(distribution.loc[stat])
    assert len(row) == len(distribution_stats)
    output_df.loc[key] = row

output_df.to_csv(
    utils.get_save_path_with_file(
        file_name, analysis_name, "distributions.csv", "data-processing"
    )
)
