import pandas as pd

import utils


input_file_name = "phi-uniq_set-3hr_hyst-2020_08_29_23-v0_1_develop-12c5af2"
data_path = utils.find_full_path(input_file_name, ".csv")
df = pd.read_csv(data_path)
analysis_name = "evaluate-equations"


aspirational_adults = utils.filter_aspirational_data_adult(df)
aspirational_peds = utils.filter_aspirational_data_peds(df)
aspirational_overall = pd.concat([aspirational_adults, aspirational_peds])

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
