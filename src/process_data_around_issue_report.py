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
from rolling_statistics import get_hourly_rolling_stats
import risk_metrics as risk_metrics
import schedule_parser as schedule_parser

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
        "-time_series_data_save_path",
        dest="time_series_data_save_path",
        default="data/processed/time-series-data-around-issue-reports/",
        help="The folder path of where to save 14day data samples",
    )

    parser.add_argument(
        "-time_series_with_stats_data_save_path",
        dest="time_series_with_stats_data_save_path",
        default="data/processed/time-series-data-with-stats-around-issue-reports/",
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


def remove_overlapping_issue_reports(reports_from_one_id):
    """
    Some issue reports occur on the same day or within a certain amount of time.
    Only use reports that are at least ANALYSIS_WINDOW_DAYS (default 7) away from the last.

    Parameters
    ----------
    reports_from_one_id : pandas.DataFrame
        A dataframe containing all issue reports from one loop_id

    Returns
    -------
        reports_from_one_id : pandas.DataFrame
            A dataframe containing all issue reports from one loop_id
    """

    for i in range(len(reports_from_one_id)):
        no_overlap = ~(reports_from_one_id["report_timestamp"].diff().dt.days < ANALYSIS_WINDOW_DAYS)
        reports_from_one_id = reports_from_one_id[no_overlap]

    return reports_from_one_id


def get_start_and_end_times(single_report, local_timezone):
    """
    The ± ANALYSIS_WINDOW_DAYS timestamps are needed to cut the data properly

    Parameters
    ----------
    single_report : pd.Series
        A single issue report containing settings, report time, and other information
    local_timezone : str
        The name of the local timezone name calculated from the issue report's timezone offset

    Returns
    -------
    sample_start_time : datetime
    sample_end_time : datetime

    """
    issue_report_date = single_report["report_timestamp"].tz_convert(local_timezone).floor(freq="1d")
    sample_start_time = issue_report_date - datetime.timedelta(days=ANALYSIS_WINDOW_DAYS)
    sample_end_time = issue_report_date + datetime.timedelta(days=ANALYSIS_WINDOW_DAYS) - datetime.timedelta(minutes=5)

    return sample_start_time, sample_end_time


def get_sample_data_with_buffer(data, sample_start_time, sample_end_time, local_timezone):
    """
    The data must be cut at the start/end times, but in order to calculate the proper insulin-on-board
    and basal rate data, an additional amount of buffer data is needed prior to the star time. In this case, 1 day.

    Parameters
    ----------
    data : pandas.DataFrame
        A complete flattened Jaeb / Tidepool dataset
    sample_start_time : datetime
    sample_end_time : datetime
    local_timezone : str

    Returns
    -------
    buffered_sample_data : pandas.DataFrame
        A slice from the main data containing ± ANALYSIS_WINDOW_DAYS and an additional 1-day buffer before the start
    """
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
    """
    The local time is needed to match up setting schedules to the data.
    To make this conversion, the local timezone is needed and often provided by the issue report.

    Parameters
    ----------
    single_report : pandas.Series
        A single issue report, containing the timezone offset in seconds
    data : pandas.DataFrame
        The entire Jaeb dataset from the study participant.
        The timezoneOffset column is used in case the issue report does not have a timezone set

    Returns
    -------
    local_timzone : str
        The name of the local timezone name calculated from the issue report's timezone offset

    """
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


def get_cgm_stats(cgm_data, single_report):
    """

    Parameters
    ----------
    cgm_data : pandas.DataFrame
        A dataframe of only cgm data with a converted "mg_dL" column

    single_report : pandas.Series
        The issue report to append cgm statistics to

    Returns
    -------
    single_report : pandas.Series

    """
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

    LBGI, HBGI, BGRI = risk_metrics.get_bgri(cgm_values)
    LBGI_RS = risk_metrics.lbgi_risk_score(LBGI)
    HBGI_RS = risk_metrics.hbgi_risk_score(HBGI)

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
    """
    The cgm data must be extracted and deduplicated. Only data during the actual sample start/end period is needed, so
    the buffered data is filtered.

    Parameters
    ----------
    single_report : pandas.Series
    buffered_sample_data : pandas.DataFrame
    sample_start_time : datetime

    Returns
    -------
    single_report : pandas.Series
        The same issue report, now with cgm stats added in from get_cgm_stats()
    cgm_data : pandas.DataFrame
        Cleaned, processed, and rounded time series of the cgm data from around the issue report

    """
    # Truncate buffered sample to main sample start time and cgm data
    cgm_data_loc = (buffered_sample_data["rounded_local_time"] >= sample_start_time) & (
        buffered_sample_data["type"] == "cbg"
    )
    cgm_data = buffered_sample_data[cgm_data_loc].copy()
    cgm_data["mg_dL"] = np.round(cgm_data["value"] * GLUCOSE_CONVERSION_FACTOR).astype(int)
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
    """
    Insulin-on-board is an important calculation for assessing DKA risk and other metrics.
    The Simple Diabetes Metabolism Model can be used to add all the iob effects of evey insulin delivery pulse
    within a time series.

    Parameters
    ----------
    daily_5min_ts : pandas.DataFrame
        A 5-min rounded time series containing the combined bolus and basal insulin delivered

    Returns
    -------
    daily_5min_ts : pandas.DataFrame
        The same dataframe, now with the "iob" data at every time step

    """
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
    """
    Throughout the data, 5-minute interval time series steps are used to align data.
    This function helps to create an empty time series which can be merged into.

    Parameters
    ----------
    first_timestamp : datetime
    last_timestamp : datetime

    Returns
    -------
    daily_5min_ts : pandas.DataFrame
        The 5-minute interval rounded time series dataframe

    """

    contiguous_ts = pd.date_range(first_timestamp, last_timestamp, freq="5min")
    daily_5min_ts = pd.DataFrame(contiguous_ts, columns=["rounded_local_time"])

    return daily_5min_ts


def process_basal_data(single_report, buffered_sample_data, daily_5min_ts):
    """
    The basal data must be extracted, deduplicated, and merged into a 5-minute rounded local time series

    Parameters
    ----------
    single_report : pandas.Series
    buffered_sample_data : pandas.DataFrame
    sample_start_time : datetime

    Returns
    -------
    single_report : pandas.Series
    daily_5min_ts : pandas.DataFrame

    """
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
    """
    The bolus data must be extracted, deduplicated, and merged into a 5-minute rounded local time series

    Parameters
    ----------
    single_report : pandas.Series
    buffered_sample_data : pandas.DataFrame
    sample_start_time : datetime

    Returns
    -------
    single_report : pandas.Series
    daily_5min_ts : pandas.DataFrame

    """
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
    """
    The carb data must be extracted, deduplicated, and merged into a 5-minute rounded local time series

    Parameters
    ----------
    single_report : pandas.Series
    buffered_sample_data : pandas.DataFrame
    sample_start_time : datetime

    Returns
    -------
    single_report : pandas.Series
    daily_5min_ts : pandas.DataFrame

    """
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

        carb_entries = pd.DataFrame(carb_data.groupby("rounded_local_time")["carbs"].sum()).reset_index()

        # Merge carb absorption data if it exists
        carb_absorption_col = "payload.com.loudnate.CarbKit.HKMetadataKey.AbsorptionTimeMinutes"

        if carb_absorption_col in carb_data.columns:
            carb_data.drop_duplicates("rounded_local_time", keep="last", inplace=True)

            carb_entries = pd.merge(
                carb_entries,
                carb_data[["rounded_local_time", carb_absorption_col]],
                how="left",
                on="rounded_local_time",
            )

            carb_entries.rename(columns={carb_absorption_col: "carb_absorption_minutes"}, inplace=True)
        else:
            carb_entries["carb_absorption_minutes"] = np.nan

        # Merge carbs within 5-min together
        daily_5min_ts = pd.merge(daily_5min_ts, carb_entries, how="left", on="rounded_local_time")
    else:
        single_report["carb_entries_deduplicated"] = 0
        daily_5min_ts["carbs"] = np.nan

    return single_report, daily_5min_ts


def process_daily_insulin_and_carb_data(single_report, buffered_sample_data, sample_start_time, local_timezone):
    """
    All bolus, basal, and carb information must be extracted, deduplicated, and merged into a common time series

    Parameters
    ----------
    single_report : pandas.Series
    buffered_sample_data : pandas.DataFrame
    sample_start_time : datetime
    local_timezone : str

    Returns
    -------
    single_report : pandas.Series
    insulin_carb_5min_ts : pandas.DataFrame

    """
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
    """
    A helper function to making looping through merges faster.

    Parameters
    ----------
    left : pandas.DataFrame
    right : pandas.DataFrame
    merge_on : str

    Returns
    -------
    merged_data : pandas.DataFrame

    """

    if len(right) > 0:
        merged_data = left.merge(right, on=merge_on, how="left")
    else:
        merged_data = left

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
    """
    Once all cgm, insulin, carb, and schedule information are processed, they can all be combined into one time series

    Parameters
    ----------
    sample_start_time : datetime
    sample_end_time : datetime
    cgm_data : pandas.DataFrame
    insulin_carb_5min_ts : pandas.DataFrame
    basal_rate_24hr_schedule : pandas.DataFrame
    isf_24hr_schedule : pandas.DataFrame
    carb_ratio_24hr_schedule : pandas.DataFrame
    correction_range_24hr_schedule : pandas.DataFrame

    Returns
    -------
    combined_5min_ts : pandas.DataFrame

    """
    combined_5min_ts = create_5min_ts(sample_start_time, sample_end_time)
    interval_hours_in_minutes = combined_5min_ts["rounded_local_time"].dt.hour * 60
    interval_minutes = combined_5min_ts["rounded_local_time"].dt.minute
    combined_5min_ts["day_interval_5min"] = interval_hours_in_minutes + interval_minutes

    merge_on_rounded_time = [cgm_data[["rounded_local_time", "mg_dL"]], insulin_carb_5min_ts]
    merge_on_interval = [
        basal_rate_24hr_schedule,
        isf_24hr_schedule,
        carb_ratio_24hr_schedule,
        correction_range_24hr_schedule,
    ]

    for right_item in merge_on_rounded_time:
        combined_5min_ts = merge_data(left=combined_5min_ts, right=right_item, merge_on="rounded_local_time")

    for right_item in merge_on_interval:
        combined_5min_ts = merge_data(left=combined_5min_ts, right=right_item, merge_on="day_interval_5min")

    # Rename columns
    combined_5min_ts.rename(columns={"mg_dL": "cgm", "rate": "set_basal_rate", "normal": "bolus"}, inplace=True)
    drop_cols = ["date", "day_interval_5min", "correction_range"]
    drop_cols = set(combined_5min_ts.columns) & set(drop_cols)
    combined_5min_ts.drop(columns=drop_cols, inplace=True)

    return combined_5min_ts


def add_additional_settings_to_ts(loop_id, report_idx, combined_5min_ts, single_report):
    """

    Parameters
    ----------
    loop_id : str
    report_idx : int
    combined_5min_ts : pandas.DataFrame
    single_report : pandas.Series

    Returns
    -------
    combined_5min_ts : pandas.DataFrame

    """

    combined_5min_ts.insert(0, "loop_id", loop_id)
    combined_5min_ts.insert(1, "report_num", report_idx)
    combined_5min_ts.insert(2, "age_at_baseline", single_report["ageAtBaseline"])
    combined_5min_ts.insert(3, "bmi_at_baseline", single_report["bmi"])
    combined_5min_ts.insert(4, "bmi_perc_at_baseline", single_report["bmiPerc"])

    combined_5min_ts["maximum_basal_rate"] = single_report["maximum_basal_rate"]
    combined_5min_ts["maximum_bolus"] = single_report["maximum_bolus"]
    combined_5min_ts["suspend_threshold"] = single_report["suspend_threshold"]
    retrospective_correction_enabled_bool = "true" in str(single_report["retrospective_correction_enabled"]).lower()
    combined_5min_ts["retrospective_correction_enabled"] = retrospective_correction_enabled_bool

    return combined_5min_ts


def add_carb_and_insulin_weighted_settings(single_report, combined_5min_ts):
    """
    Some settings have more weight based on their use.

    Carb ratio is weighted based on the amount of carbs entered
    ISF is weighted based on the insulin given

    Parameters
    ----------
    single_report : pandas.Series
    combined_5min_ts : pandas.DataFrame

    Returns
    -------
    single_report : pandas.Series

    """

    sum_weighted_cr = (combined_5min_ts["carbs"] * combined_5min_ts["carb_ratio"]).sum()
    total_carbs = combined_5min_ts["carbs"].sum()
    if total_carbs > 0:
        single_report["carb_weighted_carb_ratio"] = sum_weighted_cr / total_carbs

    sum_weighted_isf = (combined_5min_ts["total_insulin_delivered"] * combined_5min_ts["isf"]).sum()
    total_insulin = combined_5min_ts["total_insulin_delivered"].sum()
    if total_insulin > 0:
        single_report["insulin_weighted_isf"] = sum_weighted_isf / total_insulin

    return single_report


def main(
    loop_id,
    issue_reports,
    dataset_path,
    individual_report_results_save_path,
    time_series_data_save_path,
    time_series_with_stats_data_save_path,
):
    """
    For each loop_id, gather the issue reports that do not overlap.
    For each of of those issue reports, get the data ± ANALYSIS_WINDOW_DAYS around the report.
    Calculate statistics over cgm, insulin, carb, and setting schedules.
    Finally, combine all the data together into a single time series.

    Parameters
    ----------
    loop_id : string
        The Jaeb Participant loop id (LOOP-####)
    issue_reports : pandas.DataFrame
        The list of all issue reports (as created by parse_jaeb_issue_reports.py)
    dataset_path : string
        The path to the compressed and flattened dataset file (as created by batch-multiprocess-raw-jaeb-data.py)
    individual_report_results_save_path : string
        The folder path to save all the individual issue report summary results
    time_series_data_save_path : string
        The folder path to save the time series datasets to
    time_series_with_stats_data_save_path : string
        The folder path to save the time series datasets containing rolling stats

    Returns
    -------
    None

    """

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
                ) = schedule_parser.process_schedules(single_report)

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

                combined_5min_ts = add_additional_settings_to_ts(loop_id, report_idx, combined_5min_ts, single_report)
                single_report = add_carb_and_insulin_weighted_settings(single_report, combined_5min_ts)

                data_filename = "{}-report-{}-time-series.csv".format(loop_id, report_idx)
                data_save_path = os.path.join(time_series_data_save_path, data_filename)
                combined_5min_ts.to_csv(data_save_path, index=False)

                # Add rolling stats if cgm data is available
                if "cgm" in combined_5min_ts.columns:
                    hourly_values = [1, 2, 3, 5, 8]
                    combined_5min_ts_with_rolling_stats = get_hourly_rolling_stats(combined_5min_ts, hourly_values)
                    rolling_data_filename = "{}-report-{}-time-series-with-rolling-stats.csv.gz".format(loop_id, report_idx)
                    rolling_data_save_path = os.path.join(
                        time_series_with_stats_data_save_path, rolling_data_filename
                    )
                    combined_5min_ts_with_rolling_stats.to_csv(rolling_data_save_path, compression='gzip', index=False)

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

    main(
        args.loop_id,
        issue_reports,
        args.dataset_path,
        args.individual_report_results_save_path,
        args.time_series_data_save_path,
        args.time_series_with_stats_data_save_path,
    )
