import glob
import os
import pandas as pd
import numpy as np
import tidepool_data_science_metrics as metrics

# %% CONSTANTS
MGDL_PER_MMOLL = 18.01559


# %% GLOBAL FUNCTIONS
def getStartAndEndTimes(df, dateTimeField):
    dfBeginDate = df[dateTimeField].min()
    dfEndDate = df[dateTimeField].max()

    return dfBeginDate, dfEndDate


def removeDuplicates(df, criteriaDF):
    nBefore = len(df)
    df = df.loc[~(df[criteriaDF].duplicated())]
    df = df.reset_index(drop=True)
    nDuplicatesRemoved = nBefore - len(df)

    return df, nDuplicatesRemoved


def round_time(
    df,
    timeIntervalMinutes=5,
    timeField="time",
    roundedTimeFieldName="roundedTime",
    startWithFirstRecord=True,
    verbose=False,
):
    """
    A general purpose round time function that rounds the "time"
    field to nearest <timeIntervalMinutes> minutes
    INPUTS:
        * a dataframe (df) that contains a time field that you want to round
        * timeIntervalMinutes (defaults to 5 minutes given that most cgms output every 5 minutes)
        * timeField to round (defaults to the UTC time "time" field)
        * roundedTimeFieldName is a user specified column name (defaults to roundedTime)
        * startWithFirstRecord starts the rounding with the first record if True,
        and the last record if False (defaults to True)
        * verbose specifies whether the extra columns used to make calculations are returned
    """

    df.sort_values(by=timeField, ascending=startWithFirstRecord, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # make sure the time field is in the right form
    t = pd.to_datetime(df[timeField])

    # calculate the time between consecutive records
    t_shift = pd.to_datetime(df[timeField].shift(1))
    df["timeBetweenRecords"] = (
        round(
            (t - t_shift).dt.days * (86400 / (60 * timeIntervalMinutes))
            + (t - t_shift).dt.seconds / (60 * timeIntervalMinutes)
        )
        * timeIntervalMinutes
    )

    # separate the data into chunks if timeBetweenRecords is greater than
    # 2 times the <timeIntervalMinutes> minutes so the rounding process starts over
    largeGaps = list(df.query("abs(timeBetweenRecords) > " + str(timeIntervalMinutes * 2)).index)
    largeGaps.insert(0, 0)
    largeGaps.append(len(df))

    for gIndex in range(0, len(largeGaps) - 1):
        chunk = t[largeGaps[gIndex] : largeGaps[gIndex + 1]]
        firstRecordChunk = t[largeGaps[gIndex]]

        # calculate the time difference between each time record and the first record
        df.loc[largeGaps[gIndex] : largeGaps[gIndex + 1], "minutesFromFirstRecord"] = (
            chunk - firstRecordChunk
        ).dt.days * (86400 / (60)) + (chunk - firstRecordChunk).dt.seconds / (60)

        # then round to the nearest X Minutes
        # NOTE: the ".000001" ensures that mulitples of 2:30 always rounds up.
        df.loc[largeGaps[gIndex] : largeGaps[gIndex + 1], "roundedMinutesFromFirstRecord"] = round(
            (df.loc[largeGaps[gIndex] : largeGaps[gIndex + 1], "minutesFromFirstRecord"] / timeIntervalMinutes)
            + 0.000001
        ) * (timeIntervalMinutes)

        roundedFirstRecord = (firstRecordChunk + pd.Timedelta("1microseconds")).round(str(timeIntervalMinutes) + "min")
        df.loc[largeGaps[gIndex] : largeGaps[gIndex + 1], roundedTimeFieldName] = roundedFirstRecord + pd.to_timedelta(
            df.loc[largeGaps[gIndex] : largeGaps[gIndex + 1], "roundedMinutesFromFirstRecord"], unit="m",
        )

    # sort by time and drop fieldsfields
    df.sort_values(by=timeField, ascending=startWithFirstRecord, inplace=True)
    df.reset_index(drop=True, inplace=True)
    if verbose is False:
        df.drop(
            columns=["timeBetweenRecords", "minutesFromFirstRecord", "roundedMinutesFromFirstRecord"], inplace=True,
        )

    df[roundedTimeFieldName] = df[roundedTimeFieldName].astype("datetime64")

    return df


def add_uploadDateTime(df):
    if "upload" in df.type.unique():
        uploadTimes = pd.DataFrame(df[df.type == "upload"].groupby("uploadId").time.describe()["top"])
    else:
        uploadTimes = pd.DataFrame(columns=["top"])
    # if an upload does not have an upload date, then add one
    # NOTE: this is a new fix introduced with healthkit data...we now have
    # data that does not have an upload record
    unique_uploadIds = set(df["uploadId"].unique())
    unique_uploadRecords = set(df.loc[df["type"] == "upload", "uploadId"].unique())
    uploadIds_missing_uploadRecords = unique_uploadIds - unique_uploadRecords

    for upId in uploadIds_missing_uploadRecords:
        last_upload_time = df.loc[df["uploadId"] == upId, "time"].max()
        uploadTimes.loc[upId, "top"] = last_upload_time

    uploadTimes.reset_index(inplace=True)
    uploadTimes.rename(columns={"top": "uploadTime", "index": "uploadId"}, inplace=True)
    df = pd.merge(df, uploadTimes, how="left", on="uploadId")
    df["uploadTime"] = pd.to_datetime(df["uploadTime"])

    return df


def removeNegativeDurations(df):
    if "duration" in list(df):

        nNegativeDurations = sum(df.duration < 0)
        if nNegativeDurations > 0:
            df = df[~(df.duration < 0)]
    else:
        nNegativeDurations = np.nan

    return df, nNegativeDurations


def removeInvalidCgmValues(df):
    nBefore = len(df)
    # remove values < 38 and > 402 mg/dL
    df = df.drop(df[((df.type == "cbg") & (df.value < 2.109284236597303))].index)
    df = df.drop(df[((df.type == "cbg") & (df.value > 22.314006924003046))].index)
    nRemoved = nBefore - len(df)

    return df, nRemoved


def removeCgmDuplicates(df, timeCriterion):
    if timeCriterion in df:
        df.sort_values(
            by=[timeCriterion, "uploadTime"], ascending=[False, False], inplace=True,
        )
        dfIsNull = df[df[timeCriterion].isnull()]
        dfNotNull = df[df[timeCriterion].notnull()]
        dfNotNull, nDuplicatesRemoved = removeDuplicates(dfNotNull, [timeCriterion, "value"])
        df = pd.concat([dfIsNull, dfNotNull])
        df.sort_values(
            by=[timeCriterion, "uploadTime"], ascending=[False, False], inplace=True,
        )
    else:
        nDuplicatesRemoved = 0

    return df, nDuplicatesRemoved


def mmolL_to_mgdL(mmolL):
    return mmolL * MGDL_PER_MMOLL


# %% START OF CODE
CGM_WINDOW_HOURS = 4

# load in survey data to get start and end time for each study participant
survey_and_study_df = pd.read_csv(
    os.path.join(
        "..",
        "..",
        "..",
        "projects",
        "data-science--explore--jaeb_settings",
        "data",
        "PHI",
        "phi-cleaned-uniq_set-3hr_hyst-with_survey-2020_08_31_22-v0_1_develop-28c3b55.csv",
    ),
    low_memory=False,
)

# fill in missing issue report data if min and max of study dates
inferred_study_start_date = survey_and_study_df.start_date.min()
inferred_study_end_date = survey_and_study_df.end_date.max()

all_jos_files = glob.glob(os.path.join("..", "data", "compressed_and_zipped", "*LOOP*"))

data_summary = pd.DataFrame(index=range(0, len(all_jos_files)))
data_summary["missing_issue_report"] = False
data_summary["missing_bolus_data"] = False

sorted_jos_files = sorted(all_jos_files)
f_start = 0
f_end = len(sorted_jos_files)

total_possible_open_loop = 0

for i in range(f_start, f_end):  # enumerate(sorted(all_jos_files)):
    f = sorted_jos_files[i]
    loop_id = f[-12:-3]

    data_summary.loc[i, "loop_id"] = loop_id
    study_start_date = survey_and_study_df.loc[survey_and_study_df["loop_id"] == loop_id, "start_date"].min()
    study_end_date = survey_and_study_df.loc[survey_and_study_df["loop_id"] == loop_id, "end_date"].max()

    if pd.isnull(study_start_date):
        data_summary.loc[i, "missing_issue_report"] = True
        data_summary.loc[i, "study_start_date"] = inferred_study_start_date
        data_summary.loc[i, "study_end_date"] = inferred_study_end_date
    else:
        data_summary.loc[i, "study_start_date"] = study_start_date
        data_summary.loc[i, "study_end_date"] = study_end_date

    # load in all of the subject's data
    temp_all_df = pd.read_csv(f, sep="\t", compression="gzip", low_memory=False, index_col=[0])
    temp_all_df.drop(columns="column_combination", inplace=True)

    print("starting to process {}, {}".format(loop_id, i))
    study_df = (
        temp_all_df[
            (
                (temp_all_df["time"] >= data_summary.loc[i, "study_start_date"])
                & (temp_all_df["time"] <= data_summary.loc[i, "study_end_date"])
            )
        ]
        .copy()
        .dropna(axis=1, how="all")
        .reset_index(drop=True)
    )

    # find vignettes or episodes of missing cgm data
    study_df["time"] = pd.to_datetime(study_df["time"], utc=True)

    # remove negative durations
    if "duration" in study_df.columns:
        study_df["duration"] = study_df["duration"].astype(float)
        study_df, nNegativeDurations = removeNegativeDurations(study_df)
        data_summary.loc[i, "nNegativeDurations"] = nNegativeDurations

    # get rid of cgm values too low/high (< 38 & > 402 mg/dL)
    study_df, nInvalidCgmValues = removeInvalidCgmValues(study_df)
    data_summary.loc[i, "nInvalidCgmValues"] = nInvalidCgmValues

    # get list of unique data types
    unique_data_types = study_df["type"].unique()

    # if there is no bolus data then do not process data
    if "bolus" in unique_data_types:

        # add in smbg mg/dL info to study data (if it exists)
        if "smbg" in unique_data_types:
            smbg_mask = study_df["type"] == "smbg"
            study_df.loc[smbg_mask, "smbg.mg_dL"] = mmolL_to_mgdL(study_df.loc[smbg_mask, "value"]).astype(int)

        # TODO: can add in local time later
        study_df = round_time(
            study_df,
            timeIntervalMinutes=5,
            timeField="time",
            roundedTimeFieldName="rounded_time",
            startWithFirstRecord=True,
            verbose=False,
        )

        # group data by type
        groupedData = study_df.groupby(by="type")

        # PROCESS CGM DATA
        # filter by cgm and sort by uploadTime
        cgmData = groupedData.get_group("cbg").dropna(axis=1, how="all")
        cgmData = add_uploadDateTime(cgmData)

        cgmData = round_time(
            cgmData,
            timeIntervalMinutes=5,
            timeField="time",
            roundedTimeFieldName="rounded_time",
            startWithFirstRecord=True,
            verbose=True,
        )

        # get rid of duplicates that have the same ["deviceTime", "value"]
        cgmData, nCgmDuplicatesRemovedDeviceTime = removeCgmDuplicates(cgmData, "deviceTime")
        data_summary.loc[i, "nCgmDuplicatesRemovedDeviceTime"] = nCgmDuplicatesRemovedDeviceTime

        # get rid of duplicates that have the same ["time", "value"]
        cgmData, nCgmDuplicatesRemovedUtcTime = removeCgmDuplicates(cgmData, "time")
        data_summary.loc[i, "cnCgmDuplicatesRemovedUtcTime"] = nCgmDuplicatesRemovedUtcTime

        # get rid of duplicates that have the same "roundedTime"
        cgmData, nCgmDuplicatesRemovedRoundedTime = removeDuplicates(cgmData, "rounded_time")
        data_summary.loc[i, "nCgmDuplicatesRemovedRoundedTime"] = nCgmDuplicatesRemovedRoundedTime

        # get start and end times
        cgmBeginDate, cgmEndDate = getStartAndEndTimes(cgmData, "rounded_time")
        data_summary.loc[i, "cgm.beginDate"] = cgmBeginDate
        data_summary.loc[i, "cgm.endDate"] = cgmEndDate

        # get data in mg/dL units
        cgmData["mg_dL"] = mmolL_to_mgdL(cgmData["value"]).astype(int)

        # create a contiguous time series
        timeIntervalMinutes = 5
        rng = pd.date_range(cgmBeginDate, cgmEndDate, freq="{}min".format(timeIntervalMinutes))
        contiguousData = pd.DataFrame(rng, columns=["cDateTime"])

        # merge data
        contig_df = pd.merge(
            contiguousData,
            cgmData[["rounded_time", "mg_dL", "timeBetweenRecords"]],
            left_on="cDateTime",
            right_on="rounded_time",
            how="left",
        )

        # capture stats on the cgm gaps
        gap_threshold = 30
        gap_end_locations = contig_df["timeBetweenRecords"] > gap_threshold
        n_gaps = np.sum(gap_end_locations)
        data_summary.loc[i, "cgm.gaps_gt_{}_min".format(gap_threshold)] = n_gaps

        # capture each gap and the next CGM_WINDOW_HOURS hours of cgm data
        has_cgm_stats_df = pd.DataFrame()
        open_loop_start = contig_df.loc[0, "cDateTime"]
        for g, gap_end_index in enumerate(contig_df[gap_end_locations].index):
            size_of_gap = contig_df.loc[gap_end_index, "timeBetweenRecords"] - 10
            gap_end = contig_df.loc[gap_end_index - 1, "cDateTime"]
            gap_start = gap_end - pd.Timedelta("{}min".format(size_of_gap - 5))

            if (
                len(study_df[((study_df["rounded_time"] >= open_loop_start) & (study_df["rounded_time"] < gap_start))])
                > 12
            ):

                has_cgm_df = (
                    study_df[((study_df["rounded_time"] >= open_loop_start) & (study_df["rounded_time"] < gap_start))]
                    .copy()
                    .dropna(axis=1, how="all")
                    .sort_values(["rounded_time", "time"])
                    .reset_index()
                )

                open_loop_start = contig_df.loc[gap_end_index, "cDateTime"]

                has_cgm_stats_df.loc[g, "loop_id"] = loop_id
                has_cgm_stats_df.loc[g, "episode_id"] = g

                episode_start = has_cgm_df["rounded_time"].min()
                episode_end = has_cgm_df["rounded_time"].max()
                has_cgm_stats_df.loc[g, "episode_start"] = episode_start
                has_cgm_stats_df.loc[g, "episode_end"] = episode_end

                size_of_episode_days = (episode_end - episode_start).total_seconds() / (60 * 60 * 24)
                has_cgm_stats_df.loc[g, "size_of_episode_days"] = size_of_episode_days

                # capture the number of boluses, temp_basals, and smbgs
                if ("payload.HKInsulinDeliveryReason" in has_cgm_df.columns) and (
                    "payload.HasLoopKitOrigin" in has_cgm_df.columns
                ):
                    bolus_mask = (has_cgm_df["payload.HKInsulinDeliveryReason"] == 2) & (
                        has_cgm_df["payload.HasLoopKitOrigin"] == 1
                    )
                    n_boluses_delivered = np.sum(bolus_mask)
                    has_cgm_stats_df.loc[g, "n_boluses_delivered"] = n_boluses_delivered

                    basal_mask = (has_cgm_df["payload.HKInsulinDeliveryReason"] == 1) & (
                        has_cgm_df["payload.HasLoopKitOrigin"] == 1
                    )
                    n_basals_delivered = np.sum(basal_mask)
                    has_cgm_stats_df.loc[g, "n_basals_delivered"] = n_basals_delivered

                    # calculate average basals delivered per day
                    avg_basals_per_day = n_basals_delivered / size_of_episode_days
                    has_cgm_stats_df.loc[g, "avg_basals_per_day"] = avg_basals_per_day

                    # if number of avg_basals_per_day is < 24 and there is more than one bolus, could be open loop mode
                    if (avg_basals_per_day < 24) and (n_boluses_delivered > 0) and (avg_basals_per_day >= 1):
                        possible_open_loop = 1
                        has_cgm_stats_df.loc[g, "possible_open_loop_episode"] = possible_open_loop
                        total_possible_open_loop = total_possible_open_loop + 1
                        print("possible open loop count = {}, {}, {}".format(total_possible_open_loop, loop_id, g))

                    if "nutrition.carbohydrate.net" in has_cgm_df.columns:
                        n_carbs_entered = np.sum(has_cgm_df["nutrition.carbohydrate.net"].notnull())
                        has_cgm_stats_df.loc[g, "n_carbs_entered"] = n_carbs_entered

                has_cgm_df.to_csv(
                    os.path.join("..", "data", "has_cgm", "{}-id_{}-open_{}.csv".format(loop_id, g, possible_open_loop))
                )

        has_cgm_stats_df.to_csv(os.path.join("..", "data", "has_cgm_summary", "{}-has_cgm_summary.csv".format(loop_id)))

    else:
        print("no bolus data, skipping {}, {}".format(loop_id, i))
        data_summary.loc[i, "missing_bolus_data"] = True

data_summary.to_csv(os.path.join("..", "data", "data_cleaning_summary_has_cgm_analysis.csv"))

# compile results
all_has_cgm_df = pd.DataFrame()
all_has_cgm_files = glob.glob(os.path.join("..", "data", "has_cgm_summary", "*.csv"))
for h in all_has_cgm_files:
    h_df = pd.read_csv(h, low_memory=True, index_col=[0])
    all_has_cgm_df = pd.concat([all_has_cgm_df, h_df], ignore_index=True)

    # plot the possilbe open look datasets
    if "possible_open_loop_episode" in h_df.columns:
        pos_open_index = h_df[h_df["possible_open_loop_episode"].notnull()].index
        for p in pos_open_index:

            find_file = glob.glob(
                os.path.join(
                    "..",
                    "data",
                    "has_cgm",
                    "*{}-has_cgm_df_{}*".format(h_df.loc[p, "loop_id"], h_df.loc[p, "episode_id"].astype(int)),
                )
            )

            open_df = pd.read_csv(find_file[0], index_col=[0])
            # NEXT ACTION IS TO PLOT OUT THIS EPISODE



all_has_cgm_df.to_csv(os.path.join("..", "data", "has_cgm_analysis_summary.csv"))
