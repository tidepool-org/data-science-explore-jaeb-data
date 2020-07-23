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
# %%
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
all_loop_settings_df.to_csv("PHI-parsed-loop-settings-from-issue-reports-{}.csv".format(today_date_str), index=False)
