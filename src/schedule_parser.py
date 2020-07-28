import pandas as pd
import numpy as np
import ast

# %% Constants
GLUCOSE_CONVERSION_FACTOR = 18.01559
ROUND_PRECISION = 4
DAILY_POSSIBLE_CGM_POINTS = 288
ANALYSIS_WINDOW_DAYS = 7


# %%
def build_schedule_24hr_array(schedule_dict, schedule_name):
    """
    Loop settings are stored in a schedule profile that describes the value and time they change in a 24-hour period.
    In order to calculate the time-weighted metrics of each setting, a continuous high frequency 24-hour time series
    is needed.

    Parameters
    ----------
    schedule_dict : dict
        The dictionary of settings, containing a start time from local time midnight in seconds and the setting's value
    schedule_name : str

    Returns
    -------
    schedule_24hr_array : pandas.DataFrame
        Contains a 5-minute interval time series of settings from 12am (interval 0) to 11:55pm (interval 1435)

    """

    freq_5min_array = np.arange(0, 1440, 5)
    schedule_24hr_array = pd.DataFrame(index=freq_5min_array, columns=[schedule_name])

    for schedule in schedule_dict:
        startTime = schedule["startTime"]
        if isinstance(startTime, str):
            startTime = float(startTime)
        if (startTime < 0) and (len(schedule_dict) == 1):
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
    """
    Converts a setting from mmol/L to mg/dL

    Parameters
    ----------
    value : int or float
        The value to be converted
    units : str
        The units of the value

    Returns
    -------
    value : float
        The value in mg/dL

    """
    if units == "mmol":
        value = round(value * GLUCOSE_CONVERSION_FACTOR, ROUND_PRECISION)

    return value


def check_settings_units(single_report):
    """
    Issue report settings should all be in mg/dL, but some reports have a mixture of both mg/dL and mmol/L.

    Parameters
    ----------
    single_report : pandas.Series
        The issue report's series of settings and results

    Returns
    -------
    single_report : pandas.Series
        An issue report with proper mg/dL settings values

    """
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
    """
    Converting the carb ratio schedule string into an actual 24 hour schedule and summary information

    Parameters
    ----------
    carb_ratio_schedule_string : str
        A string representation of the schedule dictionary as stored in the issue report

    Returns
    -------
    carb_ratio_schedule_count : int
    carb_ratio_median : float
    carb_ratio_geomean : float
    carb_ratio_24hr_schedule : pandas.DataFrame

    """
    schedule_dict = ast.literal_eval(carb_ratio_schedule_string)
    carb_ratio_schedule_count = len(schedule_dict)
    carb_ratio_24hr_schedule = build_schedule_24hr_array(schedule_dict, "carb_ratio")
    carb_ratio_median = carb_ratio_24hr_schedule["carb_ratio"].median()
    carb_ratio_geomean = np.exp(np.log(carb_ratio_24hr_schedule["carb_ratio"]).mean())

    return carb_ratio_schedule_count, carb_ratio_median, carb_ratio_geomean, carb_ratio_24hr_schedule


def process_isf_schedule(isf_schedule_string):
    """
    Converting the insulin sensitivity factor (isf) schedule string into an actual 24 hour schedule with summary info

    Parameters
    ----------
    isf_schedule_string : str
        A string representation of the schedule dictionary as stored in the issue report

    Returns
    -------
    isf_schedule_count : int
    isf_median : float
    isf_geomean : float
    isf_24hr_schedule : pandas.DataFrame

    """
    schedule_dict = ast.literal_eval(isf_schedule_string)
    isf_schedule_count = len(schedule_dict)
    isf_24hr_schedule = build_schedule_24hr_array(schedule_dict, "isf")
    isf_median = isf_24hr_schedule["isf"].median()
    isf_geomean = np.exp(np.log(isf_24hr_schedule["isf"]).mean())

    return isf_schedule_count, isf_median, isf_geomean, isf_24hr_schedule


def process_basal_rate_schedule(basal_rate_schedule_string):
    """
    Converting the scheduled basal rate dictionary string into an actual 24 hour schedule with summary info

    Parameters
    ----------
    basal_rate_schedule_string : str
        A string representation of the schedule dictionary as stored in the issue report

    Returns
    -------
    basal_rate_schedule_count : int
    basal_rate_median : float
    basal_rate_geomean : float
    basal_rate_24hr_schedule : pandas.DataFrame

    """
    basal_rate_schedule_dict = ast.literal_eval(basal_rate_schedule_string)
    basal_rate_schedule_count = len(basal_rate_schedule_dict)
    basal_rate_24hr_schedule = build_schedule_24hr_array(basal_rate_schedule_dict, "sbr")
    basal_rate_median = basal_rate_24hr_schedule["sbr"].median()
    basal_rate_geomean = np.exp(np.log(basal_rate_24hr_schedule["sbr"]).mean())

    hourly_divisor = len(basal_rate_24hr_schedule["sbr"]) / 24
    scheduled_basal_total_daily_insulin_expected = basal_rate_24hr_schedule["sbr"].sum() / hourly_divisor

    return (
        basal_rate_schedule_count,
        basal_rate_median,
        basal_rate_geomean,
        basal_rate_24hr_schedule,
        scheduled_basal_total_daily_insulin_expected,
    )


def process_correction_range_schedule(correction_range_schedule_string):
    """
    Converting the correction_range schedule dictionary string into an actual 24 hour schedule with summary info.
    The correction ranges are also in an array containing a the lower and upper target thresholds.

    Parameters
    ----------
    correction_range_schedule_string : str
        A string representation of the schedule dictionary as stored in the issue report

    Returns
    -------
    correction_range_schedule_count : int
    bg_target_lower_median : float
    bg_target_lower_geomean : float
    bg_target_midpoint_median : float
    bg_target_midpoint_geomean : float
    bg_target_upper_median : float
    bg_target_upper_geomean : float
    bg_target_span_median : float
    correction_range_24hr_schedule : pandas.DataFrame

    """
    schedule_dict = ast.literal_eval(correction_range_schedule_string)
    correction_range_schedule_count = len(schedule_dict)
    correction_range_24hr_schedule = build_schedule_24hr_array(schedule_dict, "correction_range")

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
    """
    The following 24-hour settings schedules need to be further parsed from the issue reports:
        Basal Rates, Insulin Sensitivity Factor, Carb Ratio, and Correction Range
    For each of these schedules, time-weighted statistics are calculated and the schedules are returned to be merged
    into the rest of the cgm, insulin, and carb time series data.

    Parameters
    ----------
    single_report : pandas.Series
        The main issue report that contains all the schedule strings to be processed

    Returns
    -------
    single_report : pandas.Series
    basal_rate_24hr_schedule : pandas.DataFrame
    isf_24hr_schedule : pandas.DataFrame
    carb_ratio_24hr_schedule : pandas.DataFrame
    correction_range_24hr_schedule : pandas.DataFrame

    """
    basal_rate_schedule_string = single_report["basal_rate_schedule"]
    basal_rate_24hr_schedule = pd.DataFrame()
    if pd.notnull(basal_rate_schedule_string):
        (
            basal_rate_schedule_count,
            basal_rate_median,
            basal_rate_geomean,
            basal_rate_24hr_schedule,
            scheduled_basal_total_daily_insulin_expected,
        ) = process_basal_rate_schedule(basal_rate_schedule_string)

        single_report["scheduled_basal_rate_schedule_count"] = basal_rate_schedule_count
        single_report["scheduled_basal_rate_median"] = basal_rate_median
        single_report["scheduled_basal_rate_geomean"] = basal_rate_geomean
        single_report["scheduled_basal_to_max_basal_ratio"] = (
            single_report["maximum_basal_rate"] / single_report["scheduled_basal_rate_median"]
        )
        single_report["scheduled_basal_total_daily_insulin_expected"] = scheduled_basal_total_daily_insulin_expected

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

        single_report = check_settings_units(single_report)

    return (
        single_report,
        basal_rate_24hr_schedule,
        isf_24hr_schedule,
        carb_ratio_24hr_schedule,
        correction_range_24hr_schedule,
    )
