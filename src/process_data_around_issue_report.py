"""
process_data_around_issue_report.py

Author: Jason Meno

Description:
    Performs processing operations to gather summary statistics on ± 1 week of data surrounding an issue report

Dependencies:
    - Compressed files:
        data/processed/PHI-compressed-data/LOOP-####.gz
    - Parsed issue reports file:
        data/PHI-parsed-loop-settings-from-issue-reports-2020-07-14.csv
"""

# %% Imports
import os
import sys
import datetime
import pytz
import pandas as pd
import numpy as np
import ast
import traceback
import argparse

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

# %% Constants
GLUCOSE_CONVERSION_FACTOR = 18.01559
ROUND_PRECISION = 4
DAILY_POSSIBLE_CGM_POINTS = 288
ANALYSIS_WINDOW_DAYS = 7

# %% Parse external arguments


def get_args():
    code_description = "Get data summary statistics per loop_id's and issue report"

    parser = argparse.ArgumentParser(description=code_description)

    parser.add_argument(
        "-loop_id", dest="loop_id", default="", help="The loop participant id",
    )

    parser.add_argument(
        "-dataset_path", dest="dataset_path", default="", help="The file path of the dataset file to process",
    )

    parser.add_argument(
        "-individual_report_results_save_path",
        dest="individual_report_results_save_path",
        default="data/processed/individual_report_results/",
        help="The folder path of where to save results",
    )

    parser.add_argument(
        "-individual_data_samples_save_path",
        dest="individual_data_samples_save_path",
        default="data/processed/individual_14day_data_samples/",
        help="The folder path of where to save 14day data samples",
    )

    parser.add_argument(
        "--mode", dest="_", default="_", help="Temp PyCharm console arg",
    )

    parser.add_argument(
        "--port", dest="_", default="_", help="Temp PyCharm console arg",
    )

    return parser.parse_args()


# %% Functions


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


def get_dka_risk_hours(iob_array, sbr):
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


def get_start_and_end_times(single_report, local_timezone):
    issue_report_date = single_report["report_timestamp"].tz_convert(local_timezone).floor(freq="1d")
    sample_start_time = issue_report_date - datetime.timedelta(days=ANALYSIS_WINDOW_DAYS)
    sample_end_time = issue_report_date + datetime.timedelta(days=ANALYSIS_WINDOW_DAYS) - datetime.timedelta(minutes=5)

    return sample_start_time, sample_end_time


def get_sample_data_with_buffer(data, sample_start_time, sample_end_time, local_timezone):
    temp_time = data["time"].copy()
    temp_rounded_local_time = temp_time.dt.ceil(freq="5min").dt.tz_convert(local_timezone)

    buffered_start_time = sample_start_time - datetime.timedelta(days=1)

    buffered_sample_data = data[
        (temp_rounded_local_time >= buffered_start_time) & (temp_rounded_local_time <= sample_end_time)
    ].copy()
    buffered_sample_data["rounded_local_time"] = (
        buffered_sample_data["time"].dt.ceil(freq="5min").dt.tz_convert(local_timezone)
    )

    return buffered_sample_data


def get_timezone_from_issue_report(single_report, data):
    if pd.notnull(single_report["basal_rate_timeZone"]):
        utc_offset = datetime.timedelta(seconds=single_report["basal_rate_timeZone"])
    elif "timezoneOffset" in data.columns:
        most_common_data_offset = data["timezoneOffset"].mode()[0]
        utc_offset = datetime.timedelta(minutes=most_common_data_offset)
    else:
        # print("No timezone could be calculated. Defaulting to GMT-6")
        utc_offset = datetime.timedelta(minutes=360)

    local_timezone = [tz for tz in pytz.all_timezones if utc_offset == pytz.timezone(tz)._utcoffset][0]
    # print("{} -- {} timezone calculated: {}".format(loop_id, single_report["file_name"], local_timezone))

    return local_timezone


def build_schedule_24hr_array(schedule_list, schedule_name):

    freq_5min_array = np.arange(0, 1440, 5)
    schedule_24hr_array = pd.DataFrame(index=freq_5min_array, columns=[schedule_name])

    for schedule in schedule_list:
        startTime = schedule["startTime"]
        if isinstance(startTime, str):
            startTime = float(startTime)
        if (startTime < 0) and (len(schedule_list) == 1):
            startTime = 0
        startTime_minutes = int(startTime / 60)

        if startTime_minutes % 5 != 0:
            raise Exception("Invalid Schedule Start Time: {}".format(str(startTime_minutes)))
        else:
            schedule_24hr_array.loc[startTime_minutes, schedule_name] = schedule["value"]

    schedule_24hr_array[schedule_name].ffill(inplace=True)

    if pd.isnull(schedule_24hr_array.loc[0])[schedule_name]:
        # No schedules at midnight, carry forward value from end of 24hr period
        schedule_24hr_array.loc[0, schedule_name] = schedule_24hr_array.loc[1435, schedule_name]
        schedule_24hr_array[schedule_name].ffill(inplace=True)

    schedule_24hr_array["day_interval_5min"] = freq_5min_array

    return schedule_24hr_array


def convert_setting_to_mg_dl(value, units):
    if units == "mmol":
        value = round(value * GLUCOSE_CONVERSION_FACTOR, ROUND_PRECISION)

    return value


def check_settings_units(single_report):
    all_fields = []
    contains_incorrect_settings_value = False

    # Suspend Threshold
    suspend_units = str(single_report["suspend_threshold_unit"])
    single_report["suspend_threshold"] = convert_setting_to_mg_dl(single_report["suspend_threshold"], suspend_units)
    all_fields += ["suspend_threshold"]
    if (single_report["suspend_threshold"] < 0) | (single_report["suspend_threshold"] > 400):
        contains_incorrect_settings_value = True

    # ISF
    isf_fields = ["isf_median", "isf_geomean"]
    all_fields += isf_fields
    isf_units = str(single_report["insulin_sensitivity_factor_unit"])

    for isf_field in isf_fields:
        if isf_field in single_report:
            single_report[isf_field] = convert_setting_to_mg_dl(single_report[isf_field], isf_units)
            if single_report[isf_field] < 0:
                contains_incorrect_settings_value = True

    # Target Ranges
    target_range_fields = [
        "override_range_workout_minimum",
        "override_range_workout_maximum",
        "override_range_premeal_minimum",
        "override_range_premeal_maximum",
        "bg_target_lower_median",
        "bg_target_lower_geomean",
        "bg_target_midpoint_median",
        "bg_target_midpoint_geomean",
        "bg_target_upper_median",
        "bg_target_upper_geomean",
        "bg_target_span_median",
    ]
    all_fields += target_range_fields

    for field in target_range_fields:
        if ("mmol" in suspend_units) | ("mmol" in isf_units):
            target_units = "mmol"
        else:
            target_units = "mg/dL"

        if field in single_report:
            single_report[field] = convert_setting_to_mg_dl(single_report[field], target_units)

    single_report["contains_incorrect_settings_value"] = contains_incorrect_settings_value

    return single_report


def process_carb_ratio_schedule(carb_ratio_schedule_string):
    schedule_list = ast.literal_eval(carb_ratio_schedule_string)
    carb_ratio_schedule_count = len(schedule_list)
    carb_ratio_24hr_schedule = build_schedule_24hr_array(schedule_list, "carb_ratio")
    carb_ratio_median = carb_ratio_24hr_schedule["carb_ratio"].median()
    carb_ratio_geomean = np.exp(np.log(carb_ratio_24hr_schedule["carb_ratio"]).mean())

    return carb_ratio_schedule_count, carb_ratio_median, carb_ratio_geomean, carb_ratio_24hr_schedule


def process_isf_schedule(isf_schedule_string):
    schedule_list = ast.literal_eval(isf_schedule_string)
    isf_schedule_count = len(schedule_list)
    isf_24hr_schedule = build_schedule_24hr_array(schedule_list, "isf")
    isf_median = isf_24hr_schedule["isf"].median()
    isf_geomean = np.exp(np.log(isf_24hr_schedule["isf"]).mean())

    return isf_schedule_count, isf_median, isf_geomean, isf_24hr_schedule


def process_basal_rate_schedule(basal_rate_schedule_string):
    basal_rate_schedule_list = ast.literal_eval(basal_rate_schedule_string)
    basal_rate_schedule_count = len(basal_rate_schedule_list)
    basal_rate_24hr_schedule = build_schedule_24hr_array(basal_rate_schedule_list, "sbr")
    basal_rate_median = basal_rate_24hr_schedule["sbr"].median()
    basal_rate_geomean = np.exp(np.log(basal_rate_24hr_schedule["sbr"]).mean())

    return basal_rate_schedule_count, basal_rate_median, basal_rate_geomean, basal_rate_24hr_schedule


def process_correction_range_schedule(correction_range_schedule_string):
    schedule_list = ast.literal_eval(correction_range_schedule_string)
    correction_range_schedule_count = len(schedule_list)
    correction_range_24hr_schedule = build_schedule_24hr_array(schedule_list, "correction_range")

    correction_range_24hr_schedule["bg_target_lower"] = correction_range_24hr_schedule["correction_range"].apply(
        lambda x: x[0]
    )
    correction_range_24hr_schedule["bg_target_upper"] = correction_range_24hr_schedule["correction_range"].apply(
        lambda x: x[1]
    )
    correction_range_24hr_schedule["bg_target_midpoint"] = correction_range_24hr_schedule["correction_range"].apply(
        lambda x: np.mean(x)
    )
    correction_range_24hr_schedule["bg_target_span"] = (
        correction_range_24hr_schedule["bg_target_upper"] - correction_range_24hr_schedule["bg_target_lower"]
    )

    bg_target_lower_median = correction_range_24hr_schedule["bg_target_lower"].median()
    bg_target_lower_geomean = np.exp(np.log(correction_range_24hr_schedule["bg_target_lower"]).mean())

    bg_target_midpoint_median = correction_range_24hr_schedule["bg_target_midpoint"].median()
    bg_target_midpoint_geomean = np.exp(np.log(correction_range_24hr_schedule["bg_target_midpoint"]).mean())

    bg_target_upper_median = correction_range_24hr_schedule["bg_target_upper"].median()
    bg_target_upper_geomean = np.exp(np.log(correction_range_24hr_schedule["bg_target_upper"]).mean())

    bg_target_span_median = correction_range_24hr_schedule["bg_target_span"].median()

    return (
        correction_range_schedule_count,
        bg_target_lower_median,
        bg_target_lower_geomean,
        bg_target_midpoint_median,
        bg_target_midpoint_geomean,
        bg_target_upper_median,
        bg_target_upper_geomean,
        bg_target_span_median,
        correction_range_24hr_schedule,
    )


def process_schedules(single_report):
    basal_rate_schedule_string = single_report["basal_rate_schedule"]
    basal_rate_24hr_schedule = pd.DataFrame()
    if pd.notnull(basal_rate_schedule_string):
        (
            basal_rate_schedule_count,
            basal_rate_median,
            basal_rate_geomean,
            basal_rate_24hr_schedule,
        ) = process_basal_rate_schedule(basal_rate_schedule_string)

        single_report["scheduled_basal_rate_schedule_count"] = basal_rate_schedule_count
        single_report["scheduled_basal_rate_median"] = basal_rate_median
        single_report["scheduled_basal_rate_geomean"] = basal_rate_geomean
        single_report["scheduled_basal_to_max_basal_ratio"] = (
            single_report["maximum_basal_rate"] / single_report["scheduled_basal_rate_median"]
        )

    isf_schedule_string = single_report["insulin_sensitivity_factor_schedule"]
    isf_24hr_schedule = pd.DataFrame()
    if pd.notnull(isf_schedule_string):
        isf_schedule_count, isf_median, isf_geomean, isf_24hr_schedule = process_isf_schedule(isf_schedule_string)

        single_report["isf_schedule_count"] = isf_schedule_count
        single_report["isf_median"] = isf_median
        single_report["isf_geomean"] = isf_geomean

    carb_ratio_schedule_string = single_report["carb_ratio_schedule"]
    carb_ratio_24hr_schedule = pd.DataFrame()
    if pd.notnull(carb_ratio_schedule_string):
        (
            carb_ratio_schedule_count,
            carb_ratio_median,
            carb_ratio_geomean,
            carb_ratio_24hr_schedule,
        ) = process_carb_ratio_schedule(carb_ratio_schedule_string)

        single_report["carb_ratio_schedule_count"] = carb_ratio_schedule_count
        single_report["carb_ratio_median"] = carb_ratio_median
        single_report["carb_ratio_geomean"] = carb_ratio_geomean

    correction_range_schedule_string = single_report["correction_range_schedule"]
    correction_range_24hr_schedule = pd.DataFrame()
    if pd.notnull(correction_range_schedule_string):
        (
            correction_range_schedule_count,
            bg_target_lower_median,
            bg_target_lower_geomean,
            bg_target_midpoint_median,
            bg_target_midpoint_geomean,
            bg_target_upper_median,
            bg_target_upper_geomean,
            bg_target_span_median,
            correction_range_24hr_schedule,
        ) = process_correction_range_schedule(correction_range_schedule_string)

        single_report["correction_range_schedule_count"] = correction_range_schedule_count
        single_report["bg_target_lower_median"] = bg_target_lower_median
        single_report["bg_target_lower_geomean"] = bg_target_lower_geomean
        single_report["bg_target_midpoint_median"] = bg_target_midpoint_median
        single_report["bg_target_midpoint_geomean"] = bg_target_midpoint_geomean
        single_report["bg_target_upper_median"] = bg_target_upper_median
        single_report["bg_target_upper_geomean"] = bg_target_upper_geomean
        single_report["bg_target_span_median"] = bg_target_span_median

    return (
        single_report,
        basal_rate_24hr_schedule,
        isf_24hr_schedule,
        carb_ratio_24hr_schedule,
        correction_range_24hr_schedule,
    )


def get_cgm_stats(cgm_data, single_report):
    cgm_day_delta = cgm_data["time"].max() - cgm_data["time"].min()
    days_of_cgm_data = cgm_day_delta.days + cgm_day_delta.seconds / 60 / 60 / 24
    cgm_values = cgm_data["mg_dL"].values
    cgm_count = len(cgm_values)
    possible_cgm_points = int(days_of_cgm_data * DAILY_POSSIBLE_CGM_POINTS) + 1
    percent_cgm_available = round(100 * (cgm_count / possible_cgm_points), ROUND_PRECISION)
    single_report["percent_cgm_available"] = percent_cgm_available

    percent_above_250 = round(100 * (sum(cgm_values > 250) / cgm_count), ROUND_PRECISION)
    percent_above_180 = round(100 * (sum(cgm_values > 180) / cgm_count), 4)
    percent_70_180 = round(100 * (sum((cgm_values >= 70) & (cgm_values <= 180)) / cgm_count), ROUND_PRECISION)
    percent_54_70 = round(100 * (sum((cgm_values >= 54) & (cgm_values <= 70)) / cgm_count), ROUND_PRECISION)
    percent_below_70 = round(100 * (sum(cgm_values < 70)) / cgm_count, ROUND_PRECISION)
    percent_below_54 = round(100 * (sum(cgm_values < 54)) / cgm_count, ROUND_PRECISION)
    percent_below_40 = round(100 * (sum(cgm_values < 40)) / cgm_count, ROUND_PRECISION)

    LBGI, HBGI, BGRI = get_bgri(cgm_values)
    LBGI_RS = lbgi_risk_score(LBGI)
    HBGI_RS = hbgi_risk_score(HBGI)

    # Possible Hourly cgm data
    # cgm_data['hour'] = cgm_data['rounded_local_time'].dt.hour
    # hourly_cgm_medians = cgm_data.groupby('hour')['mg_dL'].median()

    single_report["percent_above_250"] = percent_above_250
    single_report["percent_above_180"] = percent_above_180
    single_report["percent_70_180"] = percent_70_180
    single_report["percent_54_70"] = percent_54_70
    single_report["percent_below_70"] = percent_below_70
    single_report["percent_below_54"] = percent_below_54
    single_report["percent_below_40"] = percent_below_40
    single_report["cgm_mean"] = cgm_values.mean()
    single_report["cgm_gmi"] = 3.31 + (0.02392 * cgm_values.mean())
    single_report["cgm_std"] = np.std(cgm_values)
    single_report["cgm_median"] = np.median(cgm_values)
    single_report["cgm_geomean"] = np.exp(np.log(cgm_values).mean())
    single_report["cgm_geostd"] = np.exp(np.log(cgm_values).std())
    single_report["LBGI"] = LBGI
    single_report["HBGI"] = HBGI
    single_report["BGRI"] = BGRI
    single_report["LBGI_RS"] = LBGI_RS
    single_report["HBGI_RS"] = HBGI_RS

    return single_report


def process_cgm_data(single_report, buffered_sample_data, sample_start_time):
    # Truncate buffered sample to main sample start time and cgm data
    cgm_data_loc = (buffered_sample_data["rounded_local_time"] >= sample_start_time) & (
        buffered_sample_data["type"] == "cbg"
    )
    cgm_data = buffered_sample_data[cgm_data_loc].copy()
    cgm_data["mg_dL"] = (cgm_data["value"] * GLUCOSE_CONVERSION_FACTOR).round(ROUND_PRECISION).astype(int)
    cgm_points_before_deduplication = len(cgm_data)

    if cgm_points_before_deduplication > 0:
        cgm_data.sort_values(by=["rounded_local_time", "uploadId"], ascending=False, inplace=True)
        cgm_data = cgm_data[~cgm_data["rounded_local_time"].duplicated()].sort_values(
            "rounded_local_time", ascending=True
        )
        cgm_points_after_deduplication = len(cgm_data)
        single_report["cgm_deduplicated_points"] = cgm_points_before_deduplication - cgm_points_after_deduplication
        single_report = get_cgm_stats(cgm_data, single_report)
    else:
        single_report["percent_cgm_available"] = 0

    return single_report, cgm_data


def calculate_iob_for_timeseries(daily_5min_ts):
    smm = SimpleMetabolismModel(insulin_sensitivity_factor=1, carb_insulin_ratio=1)
    _, _, _, insulin_decay_vector = smm.run(carb_amount=0, insulin_amount=1)
    all_iob_arrays = daily_5min_ts["total_insulin_delivered"].apply(lambda x: x * insulin_decay_vector)
    decay_size = len(insulin_decay_vector)
    final_iob_array = [0] * len(all_iob_arrays) + [0] * decay_size

    for i in range(len(all_iob_arrays)):
        final_iob_array[i : (i + decay_size)] += all_iob_arrays[i]

    daily_5min_ts["iob"] = final_iob_array[:-decay_size]

    return daily_5min_ts


def create_5min_ts(first_timestamp, last_timestamp):

    contiguous_ts = pd.date_range(first_timestamp, last_timestamp, freq="5min")
    daily_5min_ts = pd.DataFrame(contiguous_ts, columns=["rounded_local_time"])

    return daily_5min_ts


def process_basal_data(single_report, buffered_sample_data, daily_5min_ts):
    basal_data = buffered_sample_data[buffered_sample_data["type"] == "basal"].copy()
    basals_before_deduplication = len(basal_data)

    if basals_before_deduplication > 0:
        basal_data.sort_values(by=["time", "uploadId"], ascending=False, inplace=True)
        basal_data = basal_data[~basal_data["rounded_local_time"].duplicated()].sort_values("time", ascending=True)
        basals_after_deduplication = len(basal_data)
        single_report["basals_deduplicated"] = basals_before_deduplication - basals_after_deduplication

        basal_rates = basal_data[["rounded_local_time", "rate"]]

        daily_5min_ts = pd.merge(daily_5min_ts, basal_rates, how="left", on="rounded_local_time")

        daily_5min_ts["basal_pulse_delivered"] = daily_5min_ts["rate"] / 12
        daily_5min_ts["basal_pulse_delivered"].ffill(limit=288, inplace=True)

    else:
        single_report["basals_deduplicated"] = 0
        daily_5min_ts["basal_pulse_delivered"] = np.nan

    return single_report, daily_5min_ts


def process_bolus_data(single_report, buffered_sample_data, daily_5min_ts):
    bolus_data = buffered_sample_data[buffered_sample_data["type"] == "bolus"].copy()
    boluses_before_deduplication = len(bolus_data)

    if boluses_before_deduplication > 0:
        bolus_data.sort_values(by=["time", "uploadId"], ascending=False, inplace=True)
        bolus_data = bolus_data[~bolus_data["time"].duplicated()].sort_values("time", ascending=True)
        boluses_after_deduplication = len(bolus_data)
        single_report["boluses_deduplicated"] = boluses_before_deduplication - boluses_after_deduplication

        # Merge boluses within 5-min together
        bolus_data = pd.DataFrame(bolus_data.groupby("rounded_local_time")["normal"].sum()).reset_index()
        boluses = bolus_data[["rounded_local_time", "normal"]]
        daily_5min_ts = pd.merge(daily_5min_ts, boluses, how="left", on="rounded_local_time")
        # daily_5min_ts["normal"].fillna(0, inplace=True)
    else:
        single_report["boluses_deduplicated"] = 0
        daily_5min_ts["normal"] = np.nan

    return single_report, daily_5min_ts


def process_carb_data(single_report, buffered_sample_data, daily_5min_ts):
    carb_data = buffered_sample_data[buffered_sample_data["type"] == "food"].copy()
    carb_entries_before_deduplication = len(carb_data)

    # Carbs may come from two difference sources: nutrition.carbohydrate(s), combine into one
    carb_data["carbs"] = 0
    if "nutrition.carbohydrates.net" in carb_data.columns:
        carb_data["carbs"] += carb_data["nutrition.carbohydrates.net"]

    if "nutrition.carbohydrate.net" in carb_data.columns:
        carb_data["carbs"] += carb_data["nutrition.carbohydrate.net"]

    if carb_entries_before_deduplication > 0:
        carb_data.sort_values(by=["time", "uploadId"], ascending=False, inplace=True)
        carb_data = carb_data[~carb_data["time"].duplicated()].sort_values("time", ascending=True)
        carb_entries_after_deduplication = len(carb_data)
        single_report["carb_entries_deduplicated"] = (
            carb_entries_before_deduplication - carb_entries_after_deduplication
        )

        # Merge carbs within 5-min together
        carb_data = pd.DataFrame(carb_data.groupby("rounded_local_time")["carbs"].sum()).reset_index()
        carb_entries = carb_data[["rounded_local_time", "carbs"]]
        daily_5min_ts = pd.merge(daily_5min_ts, carb_entries, how="left", on="rounded_local_time")
    else:
        single_report["carb_entries_deduplicated"] = 0
        daily_5min_ts["carbs"] = np.nan

    return single_report, daily_5min_ts


def process_daily_insulin_and_carb_data(single_report, buffered_sample_data, sample_start_time, local_timezone):
    buffered_5min_ts = create_5min_ts(
        first_timestamp=buffered_sample_data["rounded_local_time"].min(),
        last_timestamp=buffered_sample_data["rounded_local_time"].max(),
    )
    single_report, buffered_5min_ts = process_basal_data(single_report, buffered_sample_data, buffered_5min_ts)
    single_report, buffered_5min_ts = process_bolus_data(single_report, buffered_sample_data, buffered_5min_ts)
    single_report, buffered_5min_ts = process_carb_data(single_report, buffered_sample_data, buffered_5min_ts)
    buffered_5min_ts["total_insulin_delivered"] = buffered_5min_ts["basal_pulse_delivered"] + buffered_5min_ts[
        "normal"
    ].fillna(0)

    # Get insulin on board for all basal/bolus data
    buffered_5min_ts = calculate_iob_for_timeseries(buffered_5min_ts)

    local_start_time = sample_start_time.tz_convert(local_timezone)
    insulin_carb_5min_ts = buffered_5min_ts[buffered_5min_ts["rounded_local_time"] >= local_start_time].copy()
    insulin_carb_5min_ts["date"] = insulin_carb_5min_ts["rounded_local_time"].dt.date

    basal_daily_amounts = insulin_carb_5min_ts.groupby("date")["basal_pulse_delivered"].sum().values
    nonzero_daily_basals = basal_daily_amounts[basal_daily_amounts != 0]
    days_with_basals = len(nonzero_daily_basals)
    single_report["days_with_basals"] = days_with_basals

    if days_with_basals > 0:
        single_report["basal_total_daily_mean"] = np.mean(nonzero_daily_basals)
        single_report["basal_total_daily_median"] = np.median(nonzero_daily_basals)
        single_report["basal_total_daily_geomean"] = np.exp(np.log(nonzero_daily_basals).mean())

    bolus_daily_amounts = insulin_carb_5min_ts.groupby("date")["normal"].sum().values
    nonzero_daily_boluses = bolus_daily_amounts[bolus_daily_amounts != 0]
    days_with_boluses = len(nonzero_daily_boluses)
    single_report["days_with_boluses"] = days_with_boluses

    if days_with_boluses > 0:
        single_report["bolus_total_daily_mean"] = np.mean(nonzero_daily_boluses)
        single_report["bolus_total_daily_median"] = np.median(nonzero_daily_boluses)
        single_report["bolus_total_daily_geomean"] = np.exp(np.log(nonzero_daily_boluses).mean())

    insulin_daily_amounts = insulin_carb_5min_ts.groupby("date")["total_insulin_delivered"].sum().values
    nonzero_daily_insulin = insulin_daily_amounts[insulin_daily_amounts != 0]
    days_with_insulin = len(nonzero_daily_insulin)
    single_report["days_with_insulin"] = days_with_insulin
    if days_with_insulin > 0:
        single_report["insulin_total_daily_mean"] = np.mean(nonzero_daily_insulin)
        single_report["insulin_total_daily_median"] = np.median(nonzero_daily_insulin)
        single_report["insulin_total_daily_geomean"] = np.exp(np.log(nonzero_daily_insulin).mean())

    carbs_daily_amounts = insulin_carb_5min_ts.groupby("date")["carbs"].sum().values
    nonzero_daily_carbs = carbs_daily_amounts[carbs_daily_amounts != 0]
    days_with_carbs = len(nonzero_daily_carbs)
    single_report["days_with_carbs"] = days_with_carbs
    if days_with_carbs > 0:
        single_report["carbs_total_daily_mean"] = np.mean(nonzero_daily_carbs)
        single_report["carbs_total_daily_median"] = np.median(nonzero_daily_carbs)
        single_report["carbs_total_daily_geomean"] = np.exp(np.log(nonzero_daily_carbs).mean())

    # TODO: Calculate DKA Risk Index/Score
    # Use a rolling 24 hour window and get the median/max index and risk score
    # rolling_24hr_risk_hours = (
    #    insulin_carb_5min_ts["iob"]
    #    .rolling(window=DAILY_POSSIBLE_CGM_POINTS)
    #    .apply(lambda x: get_dka_risk_hours(iob_array=x, sbr=single_report["scheduled_basal_rate_median"]))
    # )

    return single_report, insulin_carb_5min_ts


def merge_data(left, right, merge_on):

    if len(right) > 0:
        merged_data = left.merge(right, on=merge_on, how="left")

    return merged_data


def combine_all_data_into_timeseries(
    sample_start_time,
    sample_end_time,
    cgm_data,
    insulin_carb_5min_ts,
    basal_rate_24hr_schedule,
    isf_24hr_schedule,
    carb_ratio_24hr_schedule,
    correction_range_24hr_schedule,
):
    combined_5min_ts = create_5min_ts(sample_start_time, sample_end_time)

    merge_on_rounded_time = [cgm_data["rounded_local_time", "mg_dL"], insulin_carb_5min_ts]
    merge_on_interval = [
        insulin_carb_5min_ts,
        basal_rate_24hr_schedule,
        isf_24hr_schedule,
        carb_ratio_24hr_schedule,
        correction_range_24hr_schedule,
    ]

    for right_item in merge_on_rounded_time:
        combined_5min_ts = merge_data(left=combined_5min_ts, right=right_item, merge_on="rounded_local_time")

    interval_hours_in_minutes = combined_5min_ts["rounded_local_time"].dt.hour * 60
    interval_minutes = combined_5min_ts["rounded_local_time"].dt.minute
    combined_5min_ts["day_interval_5min"] = interval_hours_in_minutes + interval_minutes

    for right_item in merge_on_interval:
        combined_5min_ts = merge_data(left=combined_5min_ts, right=right_item, merge_on="day_interval_5min")

    # Rename columns
    combined_5min_ts.rename(columns={"mg_dL": "cgm", "rate": "set_basal_rate", "normal": "bolus"}, inplace=True)
    drop_cols = ["date", "day_interval_5min", "correction_range", "bg_target_midpoint", "bg_target_span"]
    combined_5min_ts.drop(columns=drop_cols, inplace=True)

    return combined_5min_ts


def main(loop_id, issue_reports, dataset_path, individual_report_results_save_path, individual_data_samples_save_path):

    data = pd.read_csv(dataset_path, sep="\t", compression="gzip", low_memory=False)
    data["time"] = pd.to_datetime(data["time"], utc=True)

    reports_from_one_id = issue_reports[issue_reports["loop_id"] == loop_id].copy()
    reports_from_one_id.sort_values(by="report_timestamp", ascending=True, inplace=True)
    reports_from_one_id = remove_overlapping_issue_reports(reports_from_one_id)
    reports_from_one_id.reset_index(drop=True, inplace=True)

    for report_idx in range(len(reports_from_one_id)):
        try:
            single_report = reports_from_one_id.loc[report_idx].copy()
            local_timezone = get_timezone_from_issue_report(single_report, data)
            sample_start_time, sample_end_time = get_start_and_end_times(single_report, local_timezone)

            # To properly calculate IOB capture 14-day + 1 day buffer for insulin run-in.
            buffered_sample_data = get_sample_data_with_buffer(data, sample_start_time, sample_end_time, local_timezone)

            if len(buffered_sample_data) > 0:
                single_report["surrounding_data_available"] = True
                (
                    single_report,
                    basal_rate_24hr_schedule,
                    isf_24hr_schedule,
                    carb_ratio_24hr_schedule,
                    correction_range_24hr_schedule,
                ) = process_schedules(single_report)

                single_report = check_settings_units(single_report)
                single_report, cgm_data = process_cgm_data(single_report, buffered_sample_data, sample_start_time)
                single_report, insulin_carb_5min_ts = process_daily_insulin_and_carb_data(
                    single_report, buffered_sample_data, sample_start_time, local_timezone
                )

                combined_5min_ts = combine_all_data_into_timeseries(
                    sample_start_time,
                    sample_end_time,
                    cgm_data,
                    insulin_carb_5min_ts,
                    basal_rate_24hr_schedule,
                    isf_24hr_schedule,
                    carb_ratio_24hr_schedule,
                    correction_range_24hr_schedule,
                )

                data_filename = "{}-report-{}-time-series.csv".format(loop_id, report_idx)
                data_save_path = os.path.join(individual_data_samples_save_path, data_filename)
                combined_5min_ts.to_csv(data_save_path, index=False)

            else:
                single_report["surrounding_data_available"] = False

            save_filename = "PHI-{}-report-{}-summary-statistics.csv".format(loop_id, report_idx)
            save_path = os.path.join(individual_report_results_save_path, save_filename)
            single_report = pd.DataFrame(single_report).T
            single_report.insert(1, "report_num", report_idx)
            single_report.to_csv(save_path, index=False)

        except Exception as e:
            print("{} -- {} FAILED with error: {}".format(loop_id, single_report["file_name"], e))
            exception_text = traceback.format_exception(*sys.exc_info())
            print("\nException Text:\n")
            for text_string in exception_text:
                print(text_string)


# %%
if __name__ == "__main__":
    issue_reports = pd.read_csv("data/PHI-parsed-loop-settings-from-issue-reports-2020-07-14.csv")
    issue_reports["report_timestamp"] = pd.to_datetime(issue_reports["report_timestamp"], utc=True)

    args = get_args()
    loop_id, dataset_path, individual_report_results_save_path, individual_data_samples_save_path = (
        args.loop_id,
        args.dataset_path,
        args.individual_report_results_save_path,
        args.individual_data_samples_save_path,
    )

    main(loop_id, issue_reports, dataset_path, individual_report_results_save_path, individual_data_samples_save_path)
