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
percent_above_250 = "percent_above_250"
percent_above_400 = "percent_above_400"
percent_cgm = "percent_cgm_available"
issue_report_date = "issue_report_date"
loop_id = "loop_id"
percent_true = "percent_true_over_outcome"

# From the survey data
bmi = "bmi_at_baseline"
bmi_percentile = "bmi_perc_at_baseline"
age = "age_at_baseline"

# From our aggregation
num_reports = "num_reports_used"

aggregate_output_rows = [
    loop_id,
    num_reports,
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
    percent_above_250,
    percent_above_400,
    age,
    bmi,
    bmi_percentile,
]

input_directory = "phi-all-setting-schedule_dataset-basal_minutes_30-outcome_hours_3-expanded_windows_False-days_around_issue_report_7"
analysis_name = "make_dataset"
all_patient_files = glob.glob(
    os.path.join("..", "jaeb-analysis", "data", ".PHI", "*LOOP*",)
)

all_output_rows_df = None

for file_path in all_patient_files:
    # print("Analyzing file at {}".format(file_path))
    df = pd.read_csv(file_path)

    # Initialize our df using the column data from our first file
    if all_output_rows_df is None:
        all_output_rows_df = pd.DataFrame(columns=df.columns)

    df.dropna(
        subset=[
            tdd,
            total_daily_basal,
            total_daily_carbs,
            isf,
            icr,
            tir,
            percent_below_40,
            percent_below_54,
            percent_below_70,
            percent_above_250,
            percent_above_400,
            percent_cgm,
        ]
    )

    # Don't include data where Loop wasn't running for most of the time
    # TODO: consult with others about this threshold once we have rest of data
    if df[percent_true].max() < 70:
        continue

    best_rows = df[
        (df[percent_true] == df[percent_true].max()) & (df[percent_cgm] >= 90)
    ]
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

for id, df in patient_dfs:
    row = [
        id,
        len(df.index),
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
        df[percent_above_250].mean(),
        df[percent_above_400].mean(),
        # Set -1 as a placeholder in the demographics columns
        -1,
        -1,
        -1,
    ]
    all_patients_df.loc[len(all_patients_df.index)] = row

print(all_patients_df)
# TODO: add survey data
export(all_output_rows_df, "all_selected_rows")
export(all_patients_df, "aggregated_rows_per_patient")

