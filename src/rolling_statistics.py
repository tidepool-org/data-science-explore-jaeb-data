import pandas as pd
import numpy as np
import src.risk_metrics as risk_metrics

# %%


def rle(inarray):
    """
    Run length encoding. Partial credit to R rle function

    Parameters
    ----------
    inarray : np.array

    Returns
    -------
        run_lengths : np.array
        run_start_positions : np.array
        run_values : np.array
    """

    inarray = np.asarray(inarray)  # force numpy
    n = len(inarray)
    if n == 0:
        return (None, None, None)
    else:
        value_change_bool = np.array(inarray[1:] != inarray[:-1])  # pairwise unequal (string safe)
        value_change_indices = np.append(np.where(value_change_bool), n - 1)  # must include last element position
        run_lengths = np.diff(np.append(-1, value_change_indices))  # run lengths
        run_start_positions = np.cumsum(np.append(0, run_lengths))[:-1]  # positions
        run_values = inarray[value_change_indices]
        return (run_lengths, run_start_positions, run_values)


def get_columns_to_shift(hour_value, event_ranges, cgm_ranges):

    columns_to_shift = []

    general_columns = [
        "{}hr_n_cgm_points",
        "{}hr_perc_cgm_available",
        "{}hr_cgm_mean",
        "{}hr_cgm_std",
        "{}hr_cgm_median",
        "{}hr_cgm_geomean",
        "{}hr_cgm_geostd",
        "{}hr_LBGI",
        "{}hr_HBGI",
        "{}hr_BGRI",
        "{}hr_LBGI_RS",
        "{}hr_HBGI_RS",
        "{}hr_DKAI",
        "{}hr_DKAI_RS",
    ]

    event_columns = [
        "{}hr_{}_event_count",
        "{}hr_cgm_{}_avg_event_length_minutes",
    ]

    [columns_to_shift.append(gen_col.format(hour_value)) for gen_col in general_columns]
    [columns_to_shift.append("{}hr_perc_cgm_{}".format(hour_value, range_suffix)) for range_suffix in cgm_ranges]
    for event_suffix in event_ranges:
        [columns_to_shift.append(event_col.format(hour_value, event_suffix)) for event_col in event_columns]

    return columns_to_shift


# %% Rolling Stats Functions
def get_hourly_rolling_stats(combined_5min_ts, hourly_values):
    """

    Parameters
    ----------
    combined_5min_ts : pandas.DataFrame
        A 5 minute timeseries containing cgm, carb, insulin, and settings data
    hourly_values : list
        A list of all the hourly values needed to calculate rolling stats over

    Returns
    -------

    """
    # Run Length Encoding
    # This is used to calculate daily hypo/hyper events and duration
    # Credit to Thomas Browne for the vectorized python format

    """

        LBGI
        HBGI
        BGRI
        LBGI_RS
        HBGI_RS

    Parameters
    ----------
    combined_5min_ts :
    rolling_prefixes :

    Returns
    -------

    """
    # Setup run length encoding for hypo/hyper events

    temp_rolling_df = combined_5min_ts.copy()

    temp_rolling_df["cgm_lt40"] = temp_rolling_df["cgm"] < 40
    temp_rolling_df["cgm_lt54"] = temp_rolling_df["cgm"] < 54
    temp_rolling_df["cgm_54-70"] = (temp_rolling_df["cgm"] >= 54) & (temp_rolling_df["cgm"] <= 70)
    temp_rolling_df["cgm_lt70"] = temp_rolling_df["cgm"] < 70
    temp_rolling_df["cgm_70-140"] = (temp_rolling_df["cgm"] >= 70) & (temp_rolling_df["cgm"] <= 140)
    temp_rolling_df["cgm_70-180"] = (temp_rolling_df["cgm"] >= 70) & (temp_rolling_df["cgm"] <= 180)
    temp_rolling_df["cgm_gt180"] = temp_rolling_df["cgm"] > 180
    temp_rolling_df["cgm_gt250"] = temp_rolling_df["cgm"] > 250

    event_ranges = ["lt40", "lt54", "lt70", "gt180", "gt250"]
    cgm_ranges = event_ranges + ["54-70", "70-140", "70-180"]

    for event_suffix in event_ranges:
        event_col = "cgm_{}".format(event_suffix)
        run_lengths, run_start_positions, run_values = rle(temp_rolling_df[event_col])
        event_filter = np.where((run_values == True) & (run_lengths >= 3))
        event_locations = run_start_positions[event_filter]
        event_length_minutes = 5 * run_lengths[event_filter]

        event_name = event_col + "_event"
        event_length_name = event_name + "_length_minutes"
        temp_rolling_df[event_name] = False
        temp_rolling_df.loc[event_locations, event_name] = True
        temp_rolling_df[event_length_name] = 0
        temp_rolling_df.loc[event_locations, event_length_name] = event_length_minutes

    # Calculate LBGI and HBGI using equation from
    # Clarke, W., & Kovatchev, B. (2009)

    transformed_bg = 1.509 * ((np.log(temp_rolling_df["cgm"]) ** 1.084) - 5.381)
    risk_power = 10 * (transformed_bg) ** 2
    low_risk_bool = transformed_bg < 0
    high_risk_bool = transformed_bg > 0
    temp_rolling_df["low_risk_power"] = risk_power * low_risk_bool
    temp_rolling_df["high_risk_power"] = risk_power * high_risk_bool

    # Prep DKA data
    temp_rolling_df["fifty_percent_steady_state_iob_from_sbr"] = (temp_rolling_df["sbr"] * 2.111517) / 2
    temp_rolling_df["iob_lt50perc_steady_state_iob"] = (
        temp_rolling_df["iob"] < temp_rolling_df["fifty_percent_steady_state_iob_from_sbr"]
    )

    # Set minimum percentage of points required to calculate rolling statistic
    percent_points_required = 0.7
    columns_to_copy = []

    # Loop through rolling stats for each time prefix
    for hour_value in hourly_values:
        # print("{}...".format(hour_value), end="")
        rolling_points = hour_value * 12
        min_points = int(rolling_points * percent_points_required)

        rolling_cgm_window = temp_rolling_df["cgm"].rolling(rolling_points, min_periods=min_points)

        n_cgm_points = rolling_cgm_window.count()

        temp_rolling_df["{}hr_n_cgm_points".format(hour_value)] = n_cgm_points
        temp_rolling_df["{}hr_perc_cgm_available".format(hour_value)] = n_cgm_points / rolling_points
        temp_rolling_df["{}hr_cgm_mean".format(hour_value)] = rolling_cgm_window.mean()
        temp_rolling_df["{}hr_cgm_std".format(hour_value)] = rolling_cgm_window.std()
        temp_rolling_df["{}hr_cgm_median".format(hour_value)] = rolling_cgm_window.median()
        temp_rolling_df["{}hr_cgm_geomean".format(hour_value)] = rolling_cgm_window.apply(
            lambda x: np.exp(np.mean(np.log(x)))
        )
        temp_rolling_df["{}hr_cgm_geostd".format(hour_value)] = rolling_cgm_window.apply(
            lambda x: np.exp(np.std(np.log(x)))
        )

        for range_suffix in cgm_ranges:
            rolling_range = temp_rolling_df["cgm_{}".format(range_suffix)].rolling(
                rolling_points, min_periods=min_points
            )
            temp_rolling_df["{}hr_perc_cgm_{}".format(hour_value, range_suffix)] = rolling_range.sum() / n_cgm_points

        for event_suffix in event_ranges:
            rolling_event_counts = (
                temp_rolling_df["cgm_{}_event".format(event_suffix)]
                .rolling(rolling_points, min_periods=min_points)
                .sum()
            )
            temp_rolling_df["{}hr_{}_event_count".format(hour_value, event_suffix)] = rolling_event_counts

            rolling_event_lengths = temp_rolling_df["cgm_{}_event_length_minutes".format(event_suffix)].rolling(
                rolling_points, min_periods=min_points
            )

            total_event_durations = rolling_event_lengths.sum()
            temp_rolling_df["{}hr_cgm_{}_avg_event_length_minutes".format(hour_value, event_suffix)] = (
                total_event_durations / rolling_event_counts
            )

        temp_rolling_df["{}hr_LBGI".format(hour_value)] = (
            temp_rolling_df["low_risk_power"].rolling(rolling_points, min_periods=min_points).mean()
        )
        temp_rolling_df["{}hr_HBGI".format(hour_value)] = (
            temp_rolling_df["high_risk_power"].rolling(rolling_points, min_periods=min_points).mean()
        )
        temp_rolling_df["{}hr_BGRI".format(hour_value)] = (
            temp_rolling_df["{}hr_LBGI".format(hour_value)] + temp_rolling_df["{}hr_HBGI".format(hour_value)]
        )

        temp_rolling_df["{}hr_LBGI_RS".format(hour_value)] = temp_rolling_df["{}hr_LBGI".format(hour_value)].apply(
            lambda x: risk_metrics.lbgi_risk_score(x)
        )
        temp_rolling_df["{}hr_HBGI_RS".format(hour_value)] = temp_rolling_df["{}hr_HBGI".format(hour_value)].apply(
            lambda x: risk_metrics.hbgi_risk_score(x)
        )

        # DKA Metrics
        rolling_dka_window = temp_rolling_df["iob_lt50perc_steady_state_iob"].rolling(
            rolling_points, min_periods=min_points
        )
        temp_rolling_df["{}hr_DKAI".format(hour_value)] = rolling_dka_window.sum() / 12
        temp_rolling_df["{}hr_DKAI_RS".format(hour_value)] = temp_rolling_df["{}hr_DKAI".format(hour_value)].apply(
            lambda x: risk_metrics.dka_risk_score(x)
        )

        # .rolling() function calculates on [t-(rolling_points+1):t]
        # Shift data to fit rolling calculations into [t:t+rolling_points] for each time point
        columns_to_shift = get_columns_to_shift(hour_value, event_ranges, cgm_ranges)
        temp_rolling_df[columns_to_shift] = temp_rolling_df[columns_to_shift].shift(-(rolling_points - 1))
        columns_to_copy += columns_to_shift

    combined_5min_ts[columns_to_copy] = temp_rolling_df[columns_to_copy]

    return combined_5min_ts
