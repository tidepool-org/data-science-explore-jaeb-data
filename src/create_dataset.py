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
basal = "avg_basal_insulin_per_day_outcomes"  # Total daily basal
carb = "avg_carbs_per_day_outcomes"  # Total daily CHO
isf = "avg_isf"
icr = "weighted_cir_outcomes"
tir = "percent_70_180"
percent_below_40 = "percent_below_40"
percent_below_54 = "percent_below_54"
percent_above_250 = "percent_above_250"
percent_cgm = "percent_cgm_available"
issue_report_date = "issue_report_date"
loop_id = "loop_id"
percent_true = "percent_true_over_outcome"

# From the survey data
bmi = "bmi_at_baseline"
bmi_percentile = "bmi_perc_at_baseline"
age = "age_at_baseline"

input_directory = "phi-all-setting-schedule_dataset-basal_minutes_30-outcome_hours_3-expanded_windows_False-days_around_issue_report_7"
analysis_name = "make_dataset"
all_patient_files = glob.glob(
    os.path.join("..", "jaeb-analysis", "data", ".PHI", "*LOOP*",)
)

output_rows_df = None

for file_path in all_patient_files:
    df = pd.read_csv(file_path)

    # Initialize our df using the column data from our first file
    if output_rows_df is None:
        output_rows_df = pd.DataFrame(columns=df.columns)

    df.dropna(
        subset=[
            tdd,
            basal,
            carb,
            isf,
            icr,
            tir,
            percent_below_40,
            percent_below_54,
            percent_above_250,
            percent_cgm,
        ]
    )

    best_rows = df[
        (df[percent_true] == df[percent_true].max()) & (df[percent_cgm] >= 90)
    ]
    output_rows_df = output_rows_df.append(best_rows.iloc[0], ignore_index=True)

# TODO: add survey data & average by patient!!

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


export(output_rows_df, "all_selected_rows")

