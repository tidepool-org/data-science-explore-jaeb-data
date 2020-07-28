"""
parse_jaeb_issue_reports.py

Author: Jason Meno

Description:
    Parses Loop Issue Reports collected from the Jaeb Loop Study and converts them into
    a single summary .csv

Dependencies:
    - A root folder containing Jaeb Study participant subfolders
    and each participant's subfolder contains issue report .md
    Example:
        './Loop Issue Reports/LOOP-0001/Loop Report.md'

"""
from loop.issue_report import parser
import io
from contextlib import redirect_stdout
import os
import pandas as pd
import numpy as np
import datetime
import time

# %%
start_time = time.time()
issue_reports_location = "PHI-Loop Issue Report files for Tidepool - 2020-05-29"
paths = []
files = []
unhandled_section_list = []
issue_report_df_list = []
print_buffer = []

for dirpath, dirnames, filenames in os.walk(issue_reports_location):
    for filename in [f for f in filenames if f.endswith(".md")]:
        paths.append(dirpath)
        files.append(filename)

passed = []
failed = []

# %%
for i in range(len(files)):

    if i % 100 == 0:
        print(i)
    try:
        lr = parser.LoopReport()

        with io.StringIO() as buf, redirect_stdout(buf):
            parsed_issue_report_dict = lr.parse_by_file(paths[i], files[i])
            print_buffer_output = buf.getvalue()
            if len(print_buffer_output) > 0:
                print_buffer += print_buffer_output.split('\n\n')

        parsed_issue_report_df = pd.DataFrame(columns=parsed_issue_report_dict.keys(), index=[0])
        parsed_issue_report_df = parsed_issue_report_df.astype("object")
        for k in parsed_issue_report_dict.keys():
            parsed_issue_report_df[k][0] = parsed_issue_report_dict[k]

        loop_id = paths[i].split("/")[-1]

        with open(os.path.join(paths[i], files[i]), "r") as file:
            report = file.read()

        report_timestamp = report.split("Generated: ")[1].split("\n")[0]
        parsed_issue_report_df.insert(0, "loop_id", loop_id)
        parsed_issue_report_df.insert(1, "report_timestamp", report_timestamp)

        issue_report_df_list.append(parsed_issue_report_df)
        passed.append(i)
    except:
        failed.append(i)


def find_closest_issue_report_to_event(loop_id, all_loop_settings_df, event_date):
    reports_from_one_id = all_loop_settings_df[all_loop_settings_df["loop_id"] == loop_id].copy()
    reports_from_one_id["report_timestamp"] = pd.to_datetime(reports_from_one_id["report_timestamp"], utc=True)
    event_date = pd.to_datetime(event_date, utc=True)
    reports_from_one_id["report_days_away_from_event"] = abs(
        reports_from_one_id["report_timestamp"] - event_date
    ).dt.days
    closest_index = reports_from_one_id["report_days_away_from_event"].idxmin()
    report_days_away_from_event = reports_from_one_id.loc[closest_index, "report_days_away_from_event"]

    return closest_index, report_days_away_from_event


def add_bmi_data_to_results(all_loop_settings_df, bmi_data_location):
    bmi_data = pd.read_csv(bmi_data_location)
    bmi_data.rename(columns={"PtID": "loop_id"}, inplace=True)

    all_loop_settings_df = all_loop_settings_df.merge(bmi_data, on="loop_id", how="left")

    return all_loop_settings_df


def add_sh_event_data_to_results(all_loop_settings_df, sh_data_location):
    sh_data = pd.read_csv(sh_data_location)
    sh_data.rename(columns={"Participant ID": "loop_id"}, inplace=True)

    # Replace unknown event dates with survey date - 1 day
    missing_dates = sh_data["reported  SH date"] == "Participant doesn't know"
    sh_data.loc[missing_dates, "reported SH date"] = pd.to_datetime(
        sh_data.loc[missing_dates, "survey date"]
    ) - datetime.timedelta(days=1)

    sh_events_with_dates = sh_data[sh_data["reported  SH date"].notnull()]
    confirmed_sh_events = sh_events_with_dates[
        sh_events_with_dates["confirmed SH event"].str.lower() == "yes"
    ].reset_index(drop=True)

    all_loop_settings_df["confirmed_sh_event"] = 0
    all_loop_settings_df["report_days_away_from_sh_event"] = np.nan

    for sh_event_idx in range(len(confirmed_sh_events)):
        sh_event = confirmed_sh_events.loc[sh_event_idx]
        reported_sh_date = sh_event["reported  SH date"]
        loop_id = sh_event["loop_id"]
        if loop_id in all_loop_settings_df["loop_id"].values:
            closest_index, report_days_away_from_event = find_closest_issue_report_to_event(
                loop_id, all_loop_settings_df, reported_sh_date
            )
            all_loop_settings_df.loc[closest_index, "confirmed_sh_event"] = 1
            all_loop_settings_df.loc[closest_index, "report_days_away_from_sh_event"] = report_days_away_from_event

    return all_loop_settings_df


def add_dka_event_data_to_results(all_loop_settings_df, dka_data_location):
    dka_data = pd.read_csv(dka_data_location)
    dka_data.rename(columns={"ID": "loop_id"}, inplace=True)
    # Replace unknown event dates with survey date - 1 day
    missing_dates = dka_data["reported DKA date"] == "Participant doesn't know"
    dka_data.loc[missing_dates, "reported SH date"] = pd.to_datetime(
        dka_data.loc[missing_dates, "survey date"]
    ) - datetime.timedelta(days=1)

    dka_events_with_dates = dka_data[dka_data["reported DKA date"].notnull()]

    confirmed_dka_events = dka_events_with_dates[
        dka_events_with_dates["confirmed DKA event"].str.lower() == "yes"
    ].reset_index(drop=True)

    all_loop_settings_df["confirmed_dka_event"] = 0
    all_loop_settings_df["report_days_away_from_dka_event"] = np.nan

    for dka_event_idx in range(len(confirmed_dka_events)):
        dka_event = confirmed_dka_events.loc[dka_event_idx]
        reported_dka_date = dka_event["reported DKA date"]
        loop_id = dka_event["loop_id"]
        if loop_id in all_loop_settings_df["loop_id"].values:
            closest_index, report_days_away_from_event = find_closest_issue_report_to_event(
                loop_id, all_loop_settings_df, reported_dka_date
            )
            all_loop_settings_df.loc[closest_index, "confirmed_dka_event"] = 1
            all_loop_settings_df.loc[closest_index, "report_days_away_from_dka_event"] = report_days_away_from_event

    return all_loop_settings_df
# %%
# print("Unique Parser Print Buffer Contents: \n")
# set(print_buffer)
# %%
today_date_str = datetime.datetime.now().strftime("%Y-%m-%d")
all_issue_reports = pd.concat(issue_report_df_list, sort=False)
all_issue_reports.to_csv("all-parsed-results-{}.csv.gz".format(today_date_str), compression="gzip", index=False)

# %% Filter just Loop settings dataframe
settings_cols = [
    "loop_id",
    "report_timestamp",
    "file_name",
    "loop_version",
    "rileyLink_radio_firmware",
    "rileyLink_ble_firmware",
    "carb_ratio_unit",
    "carb_ratio_timeZone",
    "carb_ratio_schedule",
    "carb_default_absorption_times_fast",
    "carb_default_absorption_times_medium",
    "carb_default_absorption_times_slow",
    "insulin_sensitivity_factor_schedule",
    "insulin_sensitivity_factor_timeZone",
    "insulin_sensitivity_factor_unit",
    "basal_rate_timeZone",
    "basal_rate_schedule",
    "insulin_model",
    "insulin_action_duration",
    "pump_manager_type",
    "pump_model",
    "maximum_basal_rate",
    "maximum_bolus",
    "retrospective_correction_enabled",
    "suspend_threshold",
    "suspend_threshold_unit",
    "correction_range_schedule",
    "override_range_workout_minimum",
    "override_range_workout_maximum",
    "override_range_premeal_minimum",
    "override_range_premeal_maximum",
    "is_watch_app_installed",
    "basalProfileApplyingOverrideHistory_timeZone",
    "basalProfileApplyingOverrideHistory_items",
    "insulinSensitivityScheduleApplyingOverrideHistory_timeZone",
    "insulinSensitivityScheduleApplyingOverrideHistory_units",
    "insulinSensitivityScheduleApplyingOverrideHistory_items",
    "carbRatioScheduleApplyingOverrideHistory_timeZone",
    "carbRatioScheduleApplyingOverrideHistory_units",
    "carbRatioScheduleApplyingOverrideHistory_items",
]

all_loop_settings_df = all_issue_reports[settings_cols]

bmi_data_location = "data/Loop BMI Data.csv"
sh_data_location = "data/Loop SH Review.csv"
dka_data_location = "data/Loop DKA Review.csv"

all_loop_settings_df = add_bmi_data_to_results(all_loop_settings_df, bmi_data_location)
all_loop_settings_df = add_sh_event_data_to_results(all_loop_settings_df, sh_data_location)
all_loop_settings_df = add_dka_event_data_to_results(all_loop_settings_df, dka_data_location)


all_loop_settings_df.to_csv("PHI-parsed-loop-settings-from-issue-reports-{}.csv".format(today_date_str), index=False)
end_time = time.time()
elapsed_minutes = round((end_time - start_time) / 60, 4)
elapsed_time_message = str(len(all_loop_settings_df)) + "loop reports processed in: " + str(elapsed_minutes) + " minutes\n"
print(elapsed_time_message)