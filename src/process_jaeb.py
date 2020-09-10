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
        raise Exception("Not implemented yet")
    elif demographic == DemographicSelection.NON_ASPIRATIONAL:
        raise Exception("Not implemented yet")

    # Don't do anything if it's 'overall'
    return df

data_path = utils.find_full_path("PHI-unique-settings-with-3hr-hysteresis-from-all-data-five-minute-8hr-outcomes-2020-08-19-23-v0-1-0-ed", ".csv")
data = pd.read_csv(data_path)

age_key = "age_at_baseline"
demographics_to_get = DemographicSelection.ADULT

data = filter_df_by_demographic(data, demographics_to_get)
data.to_csv(
    utils.get_demographic_export_path(demographics_to_get, "jaeb")
)