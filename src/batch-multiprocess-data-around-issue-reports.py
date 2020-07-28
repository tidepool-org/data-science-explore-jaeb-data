"""
batch-multiprocess-data-around-issue-reports.py

Author: Jason Meno

Description:
    Processes issue report schedules and gets ± 1 week of data surrounding issue reports and calculates summary
    statistics and creates 5-min timer series datasets

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
import time
import pandas as pd
import numpy as np
import subprocess as sub
from multiprocessing import Pool, cpu_count

# %% Functions


def data_around_issue_report_subprocessor(
    loop_id_index, unique_loop_ids, dataset_list, individual_report_results_save_path, time_series_data_save_path, time_series_with_stats_data_save_path
):

    if loop_id_index % 10 == 0:
        print("Starting: " + str(loop_id_index))

    loop_id = unique_loop_ids[loop_id_index]

    matched_dataset_index = [i for i, s in enumerate(dataset_list) if loop_id in s]
    matched_dataset_count = len(matched_dataset_index)

    if matched_dataset_count == 0:
        # print("SKIPPING {} - No data".format(loop_id))
        pass

    elif matched_dataset_count > 1:
        print("SKIPPING {} - Multiple datasets".format(loop_id))
    else:
        dataset_name = dataset_list[matched_dataset_index[0]]
        dataset_path = os.path.join(dataset_location, dataset_name)

    # Set the python unbuffered state to 1 to allow stdout buffer access
    # This allows continuous reading of subprocess output
    os.environ["PYTHONUNBUFFERED"] = "1"
    p = sub.Popen(
        [
            "python",
            "src/process_data_around_issue_report.py",
            "-loop_id",
            loop_id,
            "-dataset_path",
            dataset_path,
            "-individual_report_results_save_path",
            individual_report_results_save_path,
            "-time_series_data_save_path",
            time_series_data_save_path,
            "-time_series_with_stats_data_save_path",
            time_series_with_stats_data_save_path
        ],
        stdout=sub.PIPE,
        stderr=sub.PIPE,
    )

    # Continuous write out stdout output
    # for line in iter(p.stdout.readline, b''):
    #    sys.stdout.write(line.decode(sys.stdout.encoding))
    for line in iter(p.stdout.readline, b""):
        sys.stdout.write(line.decode("utf-8"))

    output, errors = p.communicate()
    output = output.decode("utf-8")
    errors = errors.decode("utf-8")

    if errors != "":
        print(errors)

    print("COMPLETED: " + str(loop_id_index))

    return


def combine_all_results(individual_report_results_path):
    results_filenames = pd.Series(os.listdir(individual_report_results_path))
    all_results = results_filenames.apply(lambda x: pd.read_csv(os.path.join(individual_report_results_path, x)))
    all_results = pd.concat(all_results.values).reset_index(drop=True)

    return all_results


def find_closest_issue_report_to_event(loop_id, all_results, event_date):
    reports_from_one_id = all_results[all_results["loop_id"] == loop_id].copy()
    reports_from_one_id["report_timestamp"] = pd.to_datetime(reports_from_one_id["report_timestamp"], utc=True)
    event_date = pd.to_datetime(event_date, utc=True)
    reports_from_one_id["report_days_away_from_event"] = abs(
        reports_from_one_id["report_timestamp"] - event_date
    ).dt.days
    closest_index = reports_from_one_id["report_days_away_from_event"].idxmin()
    report_days_away_from_event = reports_from_one_id.loc[closest_index, "report_days_away_from_event"]

    return closest_index, report_days_away_from_event


def add_sh_event_data_to_results(all_results, sh_data_location):
    sh_data = pd.read_csv(sh_data_location)
    sh_data.rename(columns={"Participant ID": "loop_id"}, inplace=True)

    # Replace unknown event dates with survey date - 1 day
    missing_dates = sh_data["reported  SH date"] == "Participant doesn't know"
    sh_data.loc[missing_dates, "reported  SH date"] = pd.to_datetime(
        sh_data.loc[missing_dates, "survey date"]
    ) - datetime.timedelta(days=1)

    sh_events_with_dates = sh_data[sh_data["reported  SH date"].notnull()]
    confirmed_sh_events = sh_events_with_dates[
        sh_events_with_dates["confirmed SH event"].str.lower() == "yes"
    ].reset_index(drop=True)

    all_results["confirmed_sh_event"] = 0
    all_results["report_days_away_from_sh_event"] = np.nan

    for sh_event_idx in range(len(confirmed_sh_events)):
        sh_event = confirmed_sh_events.loc[sh_event_idx]
        reported_sh_date = sh_event["reported  SH date"]
        loop_id = sh_event["loop_id"]
        if loop_id in all_results["loop_id"].values:
            closest_index, report_days_away_from_event = find_closest_issue_report_to_event(
                loop_id, all_results, reported_sh_date
            )
            all_results.loc[closest_index, "confirmed_sh_event"] = 1
            all_results.loc[closest_index, "report_days_away_from_sh_event"] = report_days_away_from_event

    return all_results


def add_dka_event_data_to_results(all_results, dka_data_location):
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

    all_results["confirmed_dka_event"] = 0
    all_results["report_days_away_from_dka_event"] = np.nan

    for dka_event_idx in range(len(confirmed_dka_events)):
        dka_event = confirmed_dka_events.loc[dka_event_idx]
        reported_dka_date = dka_event["reported DKA date"]
        loop_id = dka_event["loop_id"]
        if loop_id in all_results["loop_id"].values:
            closest_index, report_days_away_from_event = find_closest_issue_report_to_event(
                loop_id, all_results, reported_dka_date
            )
            all_results.loc[closest_index, "confirmed_dka_event"] = 1
            all_results.loc[closest_index, "report_days_away_from_dka_event"] = report_days_away_from_event

    return all_results
# %%
if __name__ == "__main__":
    today_timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    start_time = time.time()

    dataset_location = "data/processed/PHI-compressed-data/"
    individual_report_results_save_path = "data/processed/individual-report-results-{}/".format(today_timestamp)
    time_series_data_save_path = "data/processed/time-series-data-around-issue-reports-{}/".format(
        today_timestamp
    )
    time_series_with_stats_data_save_path = "data/processed/time-series-data-with-stats-around-issue-reports-{}/".format(
        today_timestamp
    )
    save_dirs = [individual_report_results_save_path, time_series_data_save_path, time_series_with_stats_data_save_path]

    for dir in save_dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    dataset_list = os.listdir(dataset_location)
    issue_reports = pd.read_csv("data/PHI-parsed-loop-settings-from-issue-reports-2020-07-14.csv")
    issue_reports["report_timestamp"] = pd.to_datetime(issue_reports["report_timestamp"], utc=True)
    unique_loop_ids = issue_reports["loop_id"].unique()

    # Startup CPU multiprocessing pool
    pool = Pool(int(cpu_count()))

    pool_array = [
        pool.apply_async(
            data_around_issue_report_subprocessor,
            args=[
                loop_id_index,
                unique_loop_ids,
                dataset_list,
                individual_report_results_save_path,
                time_series_data_save_path,
                time_series_with_stats_data_save_path,
            ],
        )
        for loop_id_index in range(len(unique_loop_ids))
    ]

    pool.close()
    pool.join()

    # Combine all results with Adverse Event Data

    sh_data_location = "data/Loop SH Review.csv"
    dka_data_location = "data/Loop DKA Review.csv"

    all_results = combine_all_results(individual_report_results_save_path)
    all_results.sort_values(by="loop_id", inplace=True)

    all_results = add_sh_event_data_to_results(all_results, sh_data_location)
    all_results = add_dka_event_data_to_results(all_results, dka_data_location)

    today_timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    all_results.to_csv(
        "data/processed/PHI-issue-reports-with-surrounding-2week-data-summary-stats-{}.csv".format(today_timestamp),
        index=False,
    )

    end_time = time.time()
    elapsed_minutes = round((end_time - start_time) / 60, 4)
    elapsed_time_message = str(len(all_results)) + " processed in: " + str(elapsed_minutes) + " minutes\n"
    print(elapsed_time_message)
