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
jaeb_data_location = os.path.join("data-science-explore-jaeb-data","data", "PHI")
jaeb_data_file = (
    "PHI Tidepool Survey Data 08-19-2020-cleaned-2020_09_15_13-v0_1_develop-cfb2713.csv"
)
jaeb_datapath = "/Users/anneevered/Desktop/Tidepool 2020/Tidepool Repositories/data-science-explore-jaeb-data/data/PHI/PHI Tidepool Survey Data 08-19-2020-cleaned-2020_09_15_13-v0_1_develop-cfb2713.csv"
jaeb_data_df = pd.read_csv(jaeb_datapath, index_col=[0])

def make_baseline_demographics_table(
    jaeb_df=jaeb_data_df, save_csv_path=os.path.join("..", "reports", "figures")
):

    for cohort in jaeb_df["PtCohort"].unique():

        # Filter for the particular cohort
        df = jaeb_df[jaeb_df["PtCohort"] == cohort]

        df["height_total_inches"] = 12*df["height_feet"]+df["height_inches"]
        df["BMI"] = round(df["weight"]/(df["height_total_inches"]*df["height_total_inches"])*703, 2)
        df["HbA1c"] = df["a1cBase"].fillna(df["hba1c_level"])

        # Filter out columns needed
        df = df.loc[:,
             ("loop_id",
                "ageAtBaseline",
                "duration",
                "BMI",
                "gender",
                "race",
                "ethnicity",
                "HbA1c",
                "months_hypo_events",
             )
        ]

        # Rename the columns
        df = df.rename(
            columns={
                "loop_id": "Participant ID",
                "ageAtBaseline": "Age (Years)",
                "duration": "Diabetes Duration (Years)",
                "gender": "Gender",
                "race": "Race",
                "ethnicity": "Ethnicity",
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
            3: "Do not wish to answer",
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