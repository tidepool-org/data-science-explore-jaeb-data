import utils
import pandas as pd

"""
Create a dataset from 30-minute data intervals

break into each patient's data
    break each patient's data by issue-report date
        drop all cols where there isn't TIR info
        for row
            check that temp basal is within 5% on either side of scheduled
            check that TIR is aspirational
            add chunk to issue report df
        take average of chunks and add to overall output
"""

"""
TODOS: 
- export data properly
- calculate summary stats for each patient in correct format
- figure out how to get:
    TDD
    total daily basal
    BMI
    BMI percentile

"""
tdd = "total_daily_dose_avg"
basal = "total_daily_basal_insulin_avg"  # Total daily basal
carb = "total_daily_carb_avg"  # Total daily CHO
bmi = "bmi_at_baseline"
bmi_percentile = "bmi_perc_at_baseline"
isf = "isf"
icr = "carb_ratio"
age = "age_at_baseline"
tir = "percent_70_180"
percent_below_40 = "percent_below_40"
percent_below_54 = "percent_below_54"
percent_above_250 = "percent_above_250"
percent_cgm = "percent_cgm_available"
issue_report_date = "issue_report_date"
loop_id = "loop_id"
scheduled_basal = "basal_rate"
temp_basal = "effective_basal_rate"

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
    "scheduled_basal": scheduled_basal,
    "temp_basal": temp_basal,
}

input_file_name = "phi-all-setting-schedule_dataset-basal_minutes_30-outcome_hours_3-expanded_windows_False-days_around_issue_report_7"
data_path = utils.find_full_path(input_file_name, ".csv")
df = pd.read_csv(data_path)
output_df = pd.DataFrame(columns=df.columns)

patient_ids = df[loop_id].unique()
for patient_id in patient_ids:
    patient_df = df[df[loop_id] == patient_id]
    is_pediatric = patient_df[age].iloc[0] < 18

    for unique_report in patient_df[issue_report_date].unique():
        report_df = patient_df[patient_df[issue_report_date] == unique_report]

        # Grab only the rows with 'aspirational' results
        if is_pediatric:
            report_df = utils.filter_aspirational_data_peds(report_df, keys)
        else:
            report_df = utils.filter_aspirational_data_adult(report_df, keys)

        output_df = output_df.append(report_df)

