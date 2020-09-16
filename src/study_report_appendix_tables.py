# %% REQUIRED LIBRARIES
import os
import numpy as np
import pandas as pd
import datetime as dt
import glob
import os
import tidepool_data_science_metrics as metrics


utc_string = dt.datetime.utcnow().strftime("%Y-%m-%d-%H-%m-%S")
# TODO: automatically grab the code version to add to the figures generated
code_version = "v0-1-0"


## BASELINE DEMOGRAPHICS TABLES

# Import the data - may need to update this
jaeb_data_location = os.path.join("..", "data", "PHI")
jaeb_data_file = (
    "phi-cleaned-uniq_set-3hr_hyst-with_survey-2020_09_15_13-v0_1_develop-cfb2713.csv"
)
jaeb_datapath = os.path.join(jaeb_data_location, jaeb_data_file)
jaeb_data_df = pd.read_csv(jaeb_datapath, index_col=[0])


def make_baseline_demographics_table(
    jaeb_df=jaeb_data_df, save_csv_path=os.path.join("..", "reports", "figures")
):

    for cohort in jaeb_df["PtCohort"].unique():

        # Filter for the particular cohort
        df = jaeb_df[jaeb_df["PtCohort"] == cohort]

        # Filter out columns needed
        df = df[
            [
                "loop_id",
                "ageAtBaseline",
                "duration",
                "bmi_at_baseline",
                "gender",
                "race",
                "ethnicity",
                "a1cBase",
                "months_hypo_events",
            ]
        ]

        # Rename the columns
        df = df.rename(
            columns={
                "loop_id": "Participant ID",
                "ageAtBaseline": "Age (Years)",
                "duration": "Diabetes Duration (Years)",
                "bmi_at_baseline": "BMI",
                "gender": "Gender",
                "race": "Race",
                "ethnicity": "Ethnicity",
                "a1cBase": "HbA1c",
                "months_hypo_events": "Severe Hypoglycemia Events Reported in 3 Months Prior to Enrollment",
            }
        )

        # Race and ethnicity mapping and gender mapping (from codebook)
        gender_dict = {1: "Male", 2: "Female", 3: "Non-Binary"}
        race_dict = {
            1: "White",
            2: "Black / African - American",
            3: "Asian",
            4: "Native Hawaiian / Other Pacific Islander",
            5: "American Indian / Alaskan Native",
            6: "Prefer not to answer",
            7: "More than onerace",
        }
        ethnicity_dict = {
            1: "Hispanic or Latino",
            2: "Not Hispanic or Latino",
            3: "Don not wish to answer",
        }

        df = df.replace({"Race": race_dict})
        df = df.replace({"Ethnicity": ethnicity_dict})
        df = df.replace({"Gender": gender_dict})

        df.fillna("", inplace=True)
        df = df.drop_duplicates()

        file_name = "{}-{}_{}_{}".format(
            "jaeb_data",
            "cohort_" + str(cohort) + "_baseline_demographics_table",
            utc_string,
            code_version,
        )

        df.to_csv(os.path.join(save_csv_path, file_name + ".csv"))


make_baseline_demographics_table(
    jaeb_df=jaeb_data_df,
    save_csv_path=os.path.join(
        "..", "reports", "figures", "PHI_jaeb_data_tables_nogit"
    ),
)


## Glycemic Endpoints Files

# %% CONSTANTS
MGDL_PER_MMOLL = 18.01559


## These functions were copied from tlbrt_find_open_loop.py

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
    largeGaps = list(
        df.query("abs(timeBetweenRecords) > " + str(timeIntervalMinutes * 2)).index
    )
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
        df.loc[
            largeGaps[gIndex] : largeGaps[gIndex + 1], "roundedMinutesFromFirstRecord"
        ] = round(
            (
                df.loc[
                    largeGaps[gIndex] : largeGaps[gIndex + 1], "minutesFromFirstRecord"
                ]
                / timeIntervalMinutes
            )
            + 0.000001
        ) * (
            timeIntervalMinutes
        )

        roundedFirstRecord = (firstRecordChunk + pd.Timedelta("1microseconds")).round(
            str(timeIntervalMinutes) + "min"
        )
        df.loc[largeGaps[gIndex] : largeGaps[gIndex + 1], roundedTimeFieldName] = (
            roundedFirstRecord
            + pd.to_timedelta(
                df.loc[
                    largeGaps[gIndex] : largeGaps[gIndex + 1],
                    "roundedMinutesFromFirstRecord",
                ],
                unit="m",
            )
        )

    # sort by time and drop fieldsfields
    df.sort_values(by=timeField, ascending=startWithFirstRecord, inplace=True)
    df.reset_index(drop=True, inplace=True)
    if verbose is False:
        df.drop(
            columns=[
                "timeBetweenRecords",
                "minutesFromFirstRecord",
                "roundedMinutesFromFirstRecord",
            ],
            inplace=True,
        )

    df[roundedTimeFieldName] = df[roundedTimeFieldName].astype("datetime64")

    return df


def add_uploadDateTime(df):
    if "upload" in df.type.unique():
        uploadTimes = pd.DataFrame(
            df[df.type == "upload"].groupby("uploadId").time.describe()["top"]
        )
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
        dfNotNull, nDuplicatesRemoved = removeDuplicates(
            dfNotNull, [timeCriterion, "value"]
        )
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
        "data",
        "PHI",
        "phi-cleaned-uniq_set-3hr_hyst-with_survey-2020_09_15_13-v0_1_develop-cfb2713.csv",
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


# Create data frame to store tables in
column_names = [
    "Participant ID/nAge/nBMI",
    "Exposure (Time in Study)",
    "Time in Range (70-180 mg/dL)",
    "HbA1c",
    "Time <70",
    "Time <54",
    "Time >180",
    "Mean Glucose",
]

cohort_a_df = pd.DataFrame(columns=column_names)
cohort_b_df = pd.DataFrame(columns=column_names)


for i in range(f_start, f_end):
    f = sorted_jos_files[i]
    loop_id = f[-12:-3]

    data_summary.loc[i, "loop_id"] = loop_id
    study_start_date = survey_and_study_df.loc[
        survey_and_study_df["loop_id"] == loop_id, "start_date"
    ].min()
    study_end_date = survey_and_study_df.loc[
        survey_and_study_df["loop_id"] == loop_id, "end_date"
    ].max()

    if pd.isnull(study_start_date):
        data_summary.loc[i, "missing_issue_report"] = True
        data_summary.loc[i, "study_start_date"] = inferred_study_start_date
        data_summary.loc[i, "study_end_date"] = inferred_study_end_date
    else:
        data_summary.loc[i, "study_start_date"] = study_start_date
        data_summary.loc[i, "study_end_date"] = study_end_date

    # load in all of the subject's data
    temp_all_df = pd.read_csv(
        f, sep="\t", compression="gzip", low_memory=False, index_col=[0]
    )
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
    cgmData, nCgmDuplicatesRemovedDeviceTime = removeCgmDuplicates(
        cgmData, "deviceTime"
    )
    data_summary.loc[
        i, "nCgmDuplicatesRemovedDeviceTime"
    ] = nCgmDuplicatesRemovedDeviceTime

    # get rid of duplicates that have the same ["time", "value"]
    cgmData, nCgmDuplicatesRemovedUtcTime = removeCgmDuplicates(cgmData, "time")
    data_summary.loc[i, "cnCgmDuplicatesRemovedUtcTime"] = nCgmDuplicatesRemovedUtcTime

    # get rid of duplicates that have the same "roundedTime"
    cgmData, nCgmDuplicatesRemovedRoundedTime = removeDuplicates(
        cgmData, "rounded_time"
    )
    data_summary.loc[
        i, "nCgmDuplicatesRemovedRoundedTime"
    ] = nCgmDuplicatesRemovedRoundedTime

    # get start and end times
    cgmBeginDate, cgmEndDate = getStartAndEndTimes(cgmData, "rounded_time")
    data_summary.loc[i, "cgm.beginDate"] = cgmBeginDate
    data_summary.loc[i, "cgm.endDate"] = cgmEndDate

    # get data in mg/dL units
    cgmData["mg_dL"] = mmolL_to_mgdL(cgmData["value"]).astype(int)

    # create a contiguous time series
    timeIntervalMinutes = 5
    rng = pd.date_range(
        cgmBeginDate, cgmEndDate, freq="{}min".format(timeIntervalMinutes)
    )
    contiguousData = pd.DataFrame(rng, columns=["cDateTime"])

    # merge data
    contig_df = pd.merge(
        contiguousData,
        cgmData[["rounded_time", "mg_dL", "timeBetweenRecords"]],
        left_on="cDateTime",
        right_on="rounded_time",
        how="left",
    )

    print(contig_df)

    """
    values = get_glycemic_endpoints_data(contig_df)
    individual_participant_df = pd.DataFrame(values, columns=column_names)

    if cohort = A:
        cohort_a_df = cohort_a_df.append(individual_participant_df, ignore_index=True)
    elif cohort = B:
        cohort_b_df = cohort_b_df.append(individual_participant_df, ignore_index=True)

    """


def get_glycemic_endpoints_data(df):
    # Participant, ID, Age, BMI
    # Exposure (Time in Study)
    # Time in Range (70-180 mg/dL)
    # HbA1c
    # Time <70
    # Time <54
    # Time >180
    # Mean Glucose
    return
