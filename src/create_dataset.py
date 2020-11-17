import utils
import pandas as pd
import glob
import os

"""
Create a dataset from 30-minute data intervals

get directory
create output dataset

for each file in directory that matches regex:
    pull the row with the max value of 'percent_true_over_outcome'

add info from other datasets to cleaned settings
export all patients
"""

# From the individual issue report data
tdd = "avg_total_insulin_per_day_outcomes"
total_daily_basal = "avg_basal_insulin_per_day_outcomes"  # Total daily basal
total_daily_carbs = "avg_carbs_per_day_outcomes"  # Total daily CHO
isf = "avg_isf"
icr = "weighted_cir_outcomes"
tir = "percent_70_180"
percent_below_40 = "percent_below_40"
percent_below_54 = "percent_below_54"
percent_below_70 = "percent_below_70"
percent_above_180 = "percent_above_180"
percent_above_250 = "percent_above_250"
percent_above_400 = "percent_above_400"
percent_cgm = "percent_cgm_available"
issue_report_date = "issue_report_date"
loop_id = "loop_id"
percent_true = "percent_true_over_outcome"

# From the survey data
bmi = "BMI"
bmi_percentile = "BMIPercentile"
age = "Age"

# From our aggregation
num_reports = "num_reports_used"

rows_without_demographic_data = [
    loop_id,
    percent_true,
    percent_cgm,
    tdd,
    total_daily_basal,
    total_daily_carbs,
    isf,
    icr,
    percent_below_40,
    percent_below_54,
    percent_below_70,
    tir,
    percent_above_180,
    percent_above_250,
    percent_above_400,
]

aggregate_output_rows = rows_without_demographic_data.copy()
aggregate_output_rows.extend([age, bmi, bmi_percentile, num_reports])

input_directory = "phi-all-setting-schedule_dataset-basal_minutes_30-outcome_hours_3-expanded_windows_False-days_around_issue_report_7"
analysis_name = "make_dataset"
all_patient_files = glob.glob(
    os.path.join("..", "jaeb-analysis", "data", ".PHI", "*LOOP*",)
)

all_output_rows_df = None

for file_path in all_patient_files:
    print("Loading file at {}".format(file_path))
    df = pd.read_csv(file_path)

    # Initialize our df using the column data from our first file
    if all_output_rows_df is None:
        all_output_rows_df = pd.DataFrame(columns=df.columns)

    df.dropna(subset=rows_without_demographic_data)

    # Don't include data where Loop wasn't running for most of the time
    # TODO: consult with the others about this threshold once we have rest of data
    if df[percent_true].max() < 75:
        continue

    best_rows = df[
        (df[percent_true] == df[percent_true].max()) & (df[percent_cgm] >= 90)
    ]

    if len(best_rows.index) < 1:
        print("Skipping file at {} due to no rows fitting criteria".format(file_path))
        continue

    all_output_rows_df = all_output_rows_df.append(best_rows.iloc[0], ignore_index=True)

short_file_name = "processed-30-min-win"


def export(dataframe, df_descriptor):
    dataframe.to_csv(
        utils.get_save_path_with_file(
            short_file_name,
            analysis_name,
            short_file_name + "_" + df_descriptor + ".csv",
            "dataset-creation",
        )
    )


patient_dfs = all_output_rows_df.groupby(loop_id)
all_patients_df = pd.DataFrame(columns=aggregate_output_rows)
print("We have {} unique patient IDs".format(len(all_output_rows_df[loop_id].unique())))

# Aggregate all the rows together for each patient
for id, df in patient_dfs:
    print("Aggregating data for patient {}".format(id))
    row = [
        id,
        df[percent_true].mean(),
        df[percent_cgm].mean(),
        df[tdd].mean(),
        df[total_daily_basal].mean(),
        df[total_daily_carbs].mean(),
        df[isf].mean(),
        df[icr].mean(),
        df[percent_below_40].mean(),
        df[percent_below_54].mean(),
        df[percent_below_70].mean(),
        df[tir].mean(),
        df[percent_above_180].mean(),
        df[percent_above_250].mean(),
        df[percent_above_400].mean(),
        # Set a placeholder in the demographics columns
        None,
        None,
        None,
        len(df.index),
    ]

    # Check that percents make sense
    assert (
        99.9
        <= df[percent_below_70].mean() + df[tir].mean() + df[percent_above_180].mean()
        <= 100.1
    )

    all_patients_df.loc[len(all_patients_df.index)] = row


survey_data_file_name = "Primary-Outcome-Listings"
survey_path = utils.find_full_path(survey_data_file_name, ".csv")
survey_df = pd.read_csv(survey_path)
survey_data_loop_id = "PtID"

# Add survey data
for i in range(len(all_patients_df.index)):
    patient_id = all_patients_df.loc[i, loop_id]
    print("Adding survey data for patient {}".format(patient_id))
    rows = survey_df[survey_df[survey_data_loop_id] == patient_id]

    if len(rows.index) < 1:
        print("Couldn't find demographic info for patient {}".format(patient_id))
        continue

    row = rows.iloc[0]

    all_patients_df.loc[i, age] = row[age]
    if row[bmi] != ".":
        all_patients_df.loc[i, bmi] = row[bmi]
    if row[bmi_percentile] != ".":
        all_patients_df.loc[i, bmi_percentile] = row[bmi_percentile]

print(all_patients_df.head())

export(all_output_rows_df, "all_selected_rows")
export(all_patients_df, "aggregated_rows_per_patient")
