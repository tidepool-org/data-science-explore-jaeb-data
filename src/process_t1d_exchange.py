import pandas as pd
import numpy as np
import utils
from pathlib import Path
import math

base_path = Path(__file__).parent
isf_icr_path = (
    base_path
    / "../data/HDeviceWizard.csv"
).resolve()
age_path = (
    base_path
    / "../data/HPtRoster.csv"
)
demographics_path = (
    base_path
    / "../data/HScreening.csv"
)

icr_isf_df = pd.read_csv(isf_icr_path)
age_df = pd.read_csv(age_path)
demographics_df = pd.read_csv(demographics_path)

# Keys for working with exports
""" T1D Exchange """
# tdd_key = "total_daily_dose_avg" TODO once Jaeb publishes basal data
# basal_key = "total_daily_basal_insulin_avg"  # Total daily basal, TODO once Jaeb publishes basal data
carb_key = "total_daily_carb_avg"  # Total daily carbs
bmi_key = "bmi"
weight_key = "Weight" # in cm
height_key = "Height" # in lbs
isf_key = "InsulinSensitivity"
icr_key = "InsulinCarbRatio"
age_key = "AgeAsOfEnrollDt"
# tir_key = "percent_70_180_2week"

relevent_data = icr_isf_df[[isf_key, icr_key]]
relevent_data[isf_key] *= 18.0182 # Convert from mmol to mg/dL
# Get total daily carb intake
relevent_data[carb_key] = icr_isf_df.groupby(["PtId", "DeviceDtTmDaysFromEnroll"])["CarbInput"].sum().reset_index()["CarbInput"]
relevent_data[age_key] = age_df[age_key]
relevent_data[bmi_key] = demographics_df[[height_key, weight_key]].apply(utils.find_bmi, axis=1)

relevent_data.to_csv("t1d_exchange.csv")