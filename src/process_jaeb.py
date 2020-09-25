import pandas as pd
import numpy as np
import utils
from utils import DemographicSelection


def filter_df_by_demographic(df, demographic):
    if demographic == DemographicSelection.PEDIATRIC:
        return df[df[age_key] < 18]
    elif demographic == DemographicSelection.ADULT:
        return df[df[age_key] >= 18]
    elif demographic == DemographicSelection.ASPIRATIONAL:
        return df[
            # Normal weight
            (df[bmi_key] < 25)
            & (df[bmi_key] >= 18.5)
            # Enough data to evaluate
            & (df[percent_cgm_available] >= 90)
            & (df[days_insulin] >= 14)
            # Good CGM distributions
            & (df[below_40] == 0)
            & (df[below_54] < 1)
            & (df[percent_70_180] >= 70)
        ]
    elif demographic == DemographicSelection.NON_ASPIRATIONAL:
        return df[
            # Normal weight
            (df[bmi_key] >= 25)
            | (df[bmi_key] < 18.5)
            # Enough data to evaluate
            | (df[percent_cgm_available] >= 90)
            | (df[days_insulin] >= 14)
            # Good CGM distributions
            | (df[below_40] != 0)
            | (df[below_54] >= 1)
            | (df[percent_70_180] <= 70)
        ]

    # Don't do anything if it's 'overall'
    return df


data_path = utils.find_full_path(
    "PHI-unique-settings-with-3hr-hysteresis-from-all-data-five-minute-8hr-outcomes-2020-08-19-23-v0-1-0-ed",
    ".csv",
)
data = pd.read_csv(data_path)
analysis_name = "analyze-demographics"

age_key = "age_at_baseline"
percent_cgm_available = "percent_cgm_available_2week"
below_40 = "percent_below_40_2week"
below_54 = "percent_below_54_2week"
percent_70_180 = "percent_70_180_2week"
days_insulin = "days_with_insulin"
bmi_key = "bmi_at_baseline"

demographics_to_get = DemographicSelection.ADULT

data = filter_df_by_demographic(data, demographics_to_get)
data.to_csv(
    utils.get_demographic_export_path(demographics_to_get, "jaeb", analysis_name)
)
