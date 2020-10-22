import pandas as pd

import utils


input_file_name = "phi-uniq_set-3hr_hyst-2020_08_29_23-v0_1_develop-12c5af2"
data_path = utils.find_full_path(input_file_name, ".csv")
df = pd.read_csv(data_path)
analysis_name = "evaluate-equations"

tdd = "total_daily_dose_avg"
basal = "total_daily_basal_insulin_avg"  # Total daily basal
carb = "total_daily_carb_avg"  # Total daily CHO
bmi = "bmi_at_baseline"
bmi_percentile = "bmi_perc_at_baseline"
isf = "isf"
icr = "carb_ratio"
age = "age_at_baseline"
tir = "percent_70_180_2week"
percent_below_40 = "percent_below_40_2week"
percent_below_54 = "percent_below_54_2week"
percent_above_250 = "percent_above_250_2week"
percent_cgm = "percent_cgm_available"
days_insulin = "days_with_insulin"

keys = {
    "age": age,
    "bmi": bmi,
    "bmi_perc": bmi_percentile,
    "total_daily_basal": basal,
    "percent_cgm_available": percent_cgm,
    "days_with_insulin": days_insulin,
    "percent_below_40": percent_below_40,
    "percent_below_54": percent_below_54,
    "percent_70_180": tir,
    "percent_above_250": percent_above_250,
}
aspirational_adults = utils.filter_aspirational_data_adult(df, keys)
aspirational_peds = utils.filter_aspirational_data_peds(df, keys)
aspirational_overall = pd.concat([aspirational_adults, aspirational_peds])
print(aspirational_overall.shape)

utils.find_and_export_kfolds(
    aspirational_adults,
    input_file_name,
    analysis_name,
    utils.DemographicSelection.ADULT,
    n_splits=5,
)
utils.find_and_export_kfolds(
    aspirational_peds,
    input_file_name,
    analysis_name,
    utils.DemographicSelection.PEDIATRIC,
    n_splits=5,
)
utils.find_and_export_kfolds(
    aspirational_overall,
    input_file_name,
    analysis_name,
    utils.DemographicSelection.OVERALL,
    n_splits=5,
)


aspirational_adults.to_csv(
    utils.get_save_path_with_file(
        input_file_name, analysis_name, "adult_aspirational.csv", "data-processing",
    )
)

aspirational_peds.to_csv(
    utils.get_save_path_with_file(
        input_file_name, analysis_name, "peds_aspirational.csv", "data-processing",
    )
)

aspirational_overall.to_csv(
    utils.get_save_path_with_file(
        input_file_name, analysis_name, "overall_aspirational.csv", "data-processing",
    )
)
