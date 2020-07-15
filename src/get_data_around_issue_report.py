"""
get_data_around_issue_report.py

Author: Jason Meno

Description:
    Processes issue report schedules and gets ± 1 week of data surrounding issue reports and performs analytics
     TODO: fills out a 5-min time-series normalized dataframe with settings for larger analyses

Dependencies:
    - Compressed files:
        data/processed/PHI-compressed-data/LOOP-####.gz
    - Parsed issue reports file:
        data/Jaeb Loop Study Data/PHI-parsed-loop-settings-from-issue-reports-2020-07-01.csv

"""

# %% Imports
import os
import datetime
import pytz
import time
import pandas as pd
import numpy as np
import ast
from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

# %% Constants
GLUCOSE_CONVERSION_FACTOR = 18.01559
ROUND_PRECISION = 4
DAILY_CGM_POINTS = 288
ANALYSIS_WINDOW_DAYS = 7

# %% Functions


def get_iob_from_sbr(sbr, cir, isf):
    # NOTE: this function assumes that the scheduled basal rate (sbr)
    # was constant over the previous 8 hours leading up to the simulation.
    _, _, _, _, iob_sbr = simple_metabolism_model(carb_amount=0, insulin_amount=sbr / 12, CIR=cir, ISF=isf)

    iob_with_zeros = np.append(iob_sbr, np.zeros(8 * 12))
    iob_matrix = np.tile(iob_with_zeros, (8 * 12, 1)).T
    nrows, ncols = np.shape(iob_matrix)
    # shift the iob by 1 each time
    for t_pre in np.arange(1, ncols):
        iob_matrix[:, t_pre] = np.roll(iob_matrix[:, t_pre], t_pre)
    # fill the upper triangle with zeros
    iob_matrix_tri = iob_matrix * np.tri(nrows, ncols, 0)
    iob_sbr_t = np.sum(iob_matrix_tri, axis=1)[95:-1]

    return iob_sbr_t


def get_bgri(bg_df):
    # Calculate LBGI and HBGI using equation from
    # Clarke, W., & Kovatchev, B. (2009)
    bgs = bg_df.copy()
    bgs[bgs < 1] = 1  # this is added to take care of edge case BG <= 0
    transformed_bg = 1.509 * ((np.log(bgs) ** 1.084) - 5.381)
    risk_power = 10 * (transformed_bg) ** 2
    low_risk_bool = transformed_bg < 0
    high_risk_bool = transformed_bg > 0
    rlBG = risk_power * low_risk_bool
    rhBG = risk_power * high_risk_bool
    LBGI = np.mean(rlBG)
    HBGI = np.mean(rhBG)
    BGRI = LBGI + HBGI

    return LBGI, HBGI, BGRI


def lbgi_risk_score(lbgi):
    if lbgi > 10:
        risk = 4
    elif lbgi > 5:
        risk = 3
    elif lbgi > 2.5:
        risk = 2
    elif lbgi > 0:
        risk = 1
    else:
        risk = 0
    return risk


def hbgi_risk_score(hbgi):
    if hbgi > 18:
        risk = 4
    elif hbgi > 9:
        risk = 3
    elif hbgi > 4.5:
        risk = 2
    elif hbgi > 0:
        risk = 1
    else:
        risk = 0
    return risk


def get_steady_state_iob_from_sbr(sbr):
    return sbr * 2.111517


def get_dka_risk_hours(temp_basals, iob_array, sbr):
    steady_state_iob = get_steady_state_iob_from_sbr(sbr)

    fifty_percent_steady_state_iob = steady_state_iob / 2

    indices_with_less_50percent_sbr_iob = iob_array < fifty_percent_steady_state_iob

    hours_with_less_50percent_sbr_iob = np.sum(indices_with_less_50percent_sbr_iob) * 5 / 60

    return hours_with_less_50percent_sbr_iob


def dka_risk_score(hours_with_less_50percent_sbr_iob):
    if hours_with_less_50percent_sbr_iob >= 21:
        risk = 4
    elif hours_with_less_50percent_sbr_iob >= 14:
        risk = 3
    elif hours_with_less_50percent_sbr_iob >= 8:
        risk = 2
    elif hours_with_less_50percent_sbr_iob >= 2:
        risk = 1
    else:
        risk = 0
    return risk


def remove_overlapping_issue_reports(reports_from_one_id):

    for i in range(len(reports_from_one_id)):
        no_overlap = ~(reports_from_one_id["report_timestamp"].diff().dt.days < ANALYSIS_WINDOW_DAYS)
        reports_from_one_id = reports_from_one_id[no_overlap]

    return reports_from_one_id


def build_schedule_24hr_array(schedule_list):

    # freq=1min
    schedule_24hr_array = pd.DataFrame(index=np.arange(1440), columns=["schedule"])

    for schedule in schedule_list:
        startTime = schedule["startTime"]
        if isinstance(startTime, str):
            startTime = float(startTime)
        startTime_minutes = int(startTime / 60)
        schedule_24hr_array.loc[startTime_minutes, "schedule"] = schedule["value"]

    schedule_24hr_array.schedule.ffill(inplace=True)

    return schedule_24hr_array


def process_carb_ratio_schedule(carb_ratio_schedule_string):
    schedule_list = ast.literal_eval(carb_ratio_schedule_string)
    carb_ratio_schedule_count = len(schedule_list)
    carb_ratio_24hr_schedule = build_schedule_24hr_array(schedule_list)
    carb_ratio_median = carb_ratio_24hr_schedule["schedule"].median()
    carb_ratio_geomean = np.exp(np.log(carb_ratio_24hr_schedule["schedule"]).mean())

    return carb_ratio_schedule_count, carb_ratio_median, carb_ratio_geomean


def process_isf_schedule(isf_schedule_string, units):
    schedule_list = ast.literal_eval(isf_schedule_string)
    isf_schedule_count = len(schedule_list)
    isf_24hr_schedule = build_schedule_24hr_array(schedule_list)

    if "mmol" in units:
        isf_24hr_schedule["schedule"] = round(
            isf_24hr_schedule["schedule"] * GLUCOSE_CONVERSION_FACTOR, ROUND_PRECISION
        )
    isf_median = isf_24hr_schedule["schedule"].median()
    isf_geomean = np.exp(np.log(isf_24hr_schedule["schedule"]).mean())

    return isf_schedule_count, isf_median, isf_geomean


def process_basal_rate_schedule(basal_rate_schedule_string):
    schedule_list = ast.literal_eval(basal_rate_schedule_string)
    basal_rate_schedule_count = len(schedule_list)
    basal_rate_24hr_schedule = build_schedule_24hr_array(schedule_list)
    basal_rate_median = basal_rate_24hr_schedule["schedule"].median()
    basal_rate_geomean = np.exp(np.log(basal_rate_24hr_schedule["schedule"]).mean())

    return basal_rate_schedule_count, basal_rate_median, basal_rate_geomean


def process_correction_range_schedule(correction_range_schedule_string, units):
    schedule_list = ast.literal_eval(correction_range_schedule_string)
    correction_range_schedule_count = len(schedule_list)
    correction_range_24hr_schedule = build_schedule_24hr_array(schedule_list)

    correction_range_24hr_schedule["bg_target_lower"] = correction_range_24hr_schedule["schedule"].apply(lambda x: x[0])
    correction_range_24hr_schedule["bg_target_upper"] = correction_range_24hr_schedule["schedule"].apply(lambda x: x[1])
    correction_range_24hr_schedule["bg_target_midpoint"] = correction_range_24hr_schedule["schedule"].apply(
        lambda x: np.mean(x)
    )

    if "mmol" in units:
        correction_range_24hr_schedule["bg_target_lower"] = round(
            correction_range_24hr_schedule["bg_target_lower"] * GLUCOSE_CONVERSION_FACTOR, ROUND_PRECISION
        )

        correction_range_24hr_schedule["bg_target_upper"] = round(
            correction_range_24hr_schedule["bg_target_upper"] * GLUCOSE_CONVERSION_FACTOR, ROUND_PRECISION
        )

        correction_range_24hr_schedule["bg_target_midpoint"] = round(
            correction_range_24hr_schedule["bg_target_midpoint"] * GLUCOSE_CONVERSION_FACTOR, ROUND_PRECISION
        )

    bg_target_lower_median = correction_range_24hr_schedule["bg_target_lower"].median()
    bg_target_lower_geomean = np.exp(np.log(correction_range_24hr_schedule["bg_target_lower"]).mean())

    bg_target_midpoint_median = correction_range_24hr_schedule["bg_target_midpoint"].median()
    bg_target_midpoint_geomean = np.exp(np.log(correction_range_24hr_schedule["bg_target_midpoint"]).mean())

    bg_target_upper_median = correction_range_24hr_schedule["bg_target_upper"].median()
    bg_target_upper_geomean = np.exp(np.log(correction_range_24hr_schedule["bg_target_upper"]).mean())

    return (
        correction_range_schedule_count,
        bg_target_lower_median,
        bg_target_lower_geomean,
        bg_target_midpoint_median,
        bg_target_midpoint_geomean,
        bg_target_upper_median,
        bg_target_upper_geomean,
    )


def process_schedules(single_report, report_results):
    basal_rate_schedule_string = single_report["basal_rate_schedule"]

    if pd.notnull(basal_rate_schedule_string):
        basal_rate_schedule_count, basal_rate_median, basal_rate_geomean = process_basal_rate_schedule(
            basal_rate_schedule_string
        )

        report_results['scheduled_basal_rate_schedule_count'] = basal_rate_schedule_count
        report_results['scheduled_basal_rate_median'] = basal_rate_median
        report_results['scheduled_basal_rate_geomean'] = basal_rate_geomean

    isf_schedule_string = single_report["insulin_sensitivity_factor_schedule"]
    if pd.notnull(isf_schedule_string):
        isf_schedule_count, isf_median, isf_geomean = process_isf_schedule(
            isf_schedule_string, str(single_report["insulin_sensitivity_factor_unit"])
        )

        report_results['isf_schedule_count'] = isf_schedule_count
        report_results['isf_median'] = isf_median
        report_results['isf_geomean'] = isf_geomean

    carb_ratio_schedule_string = single_report["carb_ratio_schedule"]

    if pd.notnull(carb_ratio_schedule_string):
        carb_ratio_schedule_count, carb_ratio_median, carb_ratio_geomean = process_carb_ratio_schedule(
            carb_ratio_schedule_string
        )

        report_results['carb_ratio_schedule_count'] = carb_ratio_schedule_count
        report_results['carb_ratio_median'] = carb_ratio_median
        report_results['carb_ratio_geomean'] = carb_ratio_geomean

    correction_range_schedule_string = single_report["correction_range_schedule"]
    if pd.notnull(correction_range_schedule_string):
        (
            correction_range_schedule_count,
            bg_target_lower_median,
            bg_target_lower_geomean,
            bg_target_midpoint_median,
            bg_target_midpoint_geomean,
            bg_target_upper_median,
            bg_target_upper_geomean,
        ) = process_correction_range_schedule(
            correction_range_schedule_string, str(single_report["insulin_sensitivity_factor_unit"])
        )

        report_results['correction_range_schedule_count'] = correction_range_schedule_count
        report_results['bg_target_lower_median'] = bg_target_lower_median
        report_results['bg_target_lower_geomean'] = bg_target_lower_geomean
        report_results['bg_target_midpoint_median'] = bg_target_midpoint_median
        report_results['bg_target_midpoint_geomean'] = bg_target_midpoint_geomean
        report_results['bg_target_upper_median'] = bg_target_upper_median
        report_results['bg_target_upper_geomean'] = bg_target_upper_geomean

    return report_results


def process_cgm_data(cgm_data, report_results):
    cgm_values = cgm_data["mg_dL"].values
    cgm_count = len(cgm_values)
    possible_cgm_points = days_of_data * DAILY_CGM_POINTS
    percent_cgm_available = round(100 * (cgm_count / possible_cgm_points), ROUND_PRECISION)
    report_results["percent_cgm_available"] = percent_cgm_available

    percent_above_250 = round(100 * (sum(cgm_values > 250) / cgm_count), ROUND_PRECISION)
    percent_above_180 = round(100 * (sum(cgm_values > 180) / cgm_count), 4)
    percent_70_180 = round(
        100 * (sum((cgm_values >= 70) & (cgm_values <= 180)) / cgm_count), ROUND_PRECISION
    )
    percent_54_70 = round(100 * (sum((cgm_values >= 54) & (cgm_values <= 70)) / cgm_count), ROUND_PRECISION)
    percent_below_70 = round(100 * (sum(cgm_values < 70)) / cgm_count, ROUND_PRECISION)
    percent_below_54 = round(100 * (sum(cgm_values < 54)) / cgm_count, ROUND_PRECISION)
    percent_below_40 = round(100 * (sum(cgm_values < 40)) / cgm_count, ROUND_PRECISION)

    LBGI, HBGI, BGRI = get_bgri(cgm_values)
    LBGI_RS = lbgi_risk_score(LBGI)
    HBGI_RS = hbgi_risk_score(HBGI)

    report_results["percent_above_250"] = percent_above_250
    report_results["percent_above_180"] = percent_above_180
    report_results["percent_70_180"] = percent_70_180
    report_results["percent_54_70"] = percent_54_70
    report_results["percent_below_70"] = percent_below_70
    report_results["percent_below_54"] = percent_below_54
    report_results["percent_below_40"] = percent_below_40
    report_results["cgm_mean"] = cgm_values.mean()
    report_results["cgm_gmi"] = 3.31 + (0.02392 * cgm_values.mean())
    report_results["cgm_std"] = np.std(cgm_values)
    report_results["cgm_median"] = np.median(cgm_values)
    report_results["cgm_geomean"] = np.exp(np.log(cgm_values).mean())
    report_results["cgm_geostd"] = np.exp(np.log(cgm_values).std())
    report_results["LBGI"] = LBGI
    report_results["HBGI"] = HBGI
    report_results["BGRI"] = BGRI
    report_results["LBGI_RS"] = LBGI_RS
    report_results["HBGI_RS"] = HBGI_RS

    return report_results
# %%
start_time = time.time()

dataset_location = "data/processed/PHI-compressed-data/"
# results_location = "data/processed/individual_report_results/"

# if not os.path.exists(results_location):
#     os.makedirs(results_location)

dataset_list = os.listdir(dataset_location)
issue_reports = pd.read_csv("data/PHI-parsed-loop-settings-from-issue-reports-2020-07-14.csv")
issue_reports["report_timestamp"] = pd.to_datetime(issue_reports["report_timestamp"], utc=True)
unique_loop_ids = issue_reports["loop_id"].unique()

all_results = []

for loop_id_index in range(len(unique_loop_ids)):

    if loop_id_index % 10 == 0:
        print(loop_id_index)

    loop_id = unique_loop_ids[loop_id_index]

    matched_dataset_index = [i for i, s in enumerate(dataset_list) if loop_id in s]
    matched_dataset_count = len(matched_dataset_index)

    if matched_dataset_count == 0:
        pass
        # print("SKIPPING {} - No data".format(loop_id))
    elif matched_dataset_count > 1:
        print("SKIPPING {} - Multiple datasets".format(loop_id))
    else:
        dataset_name = dataset_list[matched_dataset_index[0]]
        dataset_path = os.path.join(dataset_location, dataset_name)
        data = pd.read_csv(dataset_path, sep="\t", compression="gzip", low_memory=False)
        data["utc_time"] = pd.to_datetime(data["time"], utc=True)

        reports_from_one_id = issue_reports[issue_reports["loop_id"] == loop_id].copy()
        reports_from_one_id.sort_values(by="report_timestamp", ascending=True, inplace=True)
        reports_from_one_id = remove_overlapping_issue_reports(reports_from_one_id)
        reports_from_one_id.reset_index(drop=True, inplace=True)

        for report_idx in range(len(reports_from_one_id)):
            single_report = reports_from_one_id.loc[report_idx]
            report_results = pd.DataFrame(single_report).T
            report_results = process_schedules(single_report, report_results)

            issue_report_date = single_report["report_timestamp"]

            sample_start_time = issue_report_date - datetime.timedelta(days=ANALYSIS_WINDOW_DAYS)
            sample_end_time = issue_report_date + datetime.timedelta(days=ANALYSIS_WINDOW_DAYS)
            days_of_data = (sample_end_time - sample_start_time).days

            utc_offset = datetime.timedelta(seconds=single_report["basal_rate_timeZone"])
            local_timezone = [tz for tz in pytz.all_timezones if utc_offset == pytz.timezone(tz)._utcoffset][0]
            print("{} -- {} timezone calculated: {}".format(loop_id, single_report["file_name"], local_timezone))

            sample_data = data[(data["utc_time"] >= sample_start_time) & (data["utc_time"] < sample_end_time)].copy()
            sample_data["local_time"] = sample_data["utc_time"].dt.tz_convert(local_timezone)

            cgm_data = sample_data[sample_data["type"] == "cbg"].copy()
            cgm_data["mg_dL"] = round(cgm_data["value"] * GLUCOSE_CONVERSION_FACTOR).astype(int)

            if len(cgm_data) > 0:
                report_results = process_cgm_data(cgm_data, report_results)
            else:
                report_results["percent_cgm_available"] = 0

            # results_filename = '{}-report-{}-results.csv'.format(loop_id, report_idx)
            # report_results.to_csv(os.path.join(results_location, results_filename), index=False)
            all_results.append(report_results)


all_results = pd.concat(all_results).reset_index(drop=True)
today_timestamp = datetime.datetime.now().strftime("%Y-%m-%d")

bmi_data_location = "data/Loop BMI Data.csv"
bmi_data = pd.read_csv(bmi_data_location)
bmi_data.columns = ['loop_id'] + list(bmi_data.columns[1:])

all_results = all_results.merge(bmi_data, on='loop_id', how='left')
all_results.to_csv("data/processed/PHI-issue-report-analysis-results-{}.csv".format(today_timestamp), index=False)

end_time = time.time()
elapsed_minutes = round((end_time - start_time) / 60, 4)
elapsed_time_message = str(len(all_results)) + " processed in: " + str(elapsed_minutes) + " minutes\n"
print(elapsed_time_message)
