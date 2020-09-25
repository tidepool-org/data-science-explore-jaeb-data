import pandas as pd
import utils

input_file_name = "phi-uniq_set-3hr_hyst-2020_08_29_23-v0_1_develop-12c5af2"
data_path = utils.find_full_path(input_file_name, ".csv")
df = pd.read_csv(data_path)
result_cols = [
    "jaeb_aic",
    "traditional_aic",
    "jaeb_rmse",
    "traditional_rmse",
    "jaeb_sse",
    "traditional_sse",
    "jaeb_bic",
    "traditional_bic",
]
output_df = pd.DataFrame(columns=result_cols)
analysis_name = "evaluate-equations"


# Keys for working with Jaeb exports
tdd_key = "total_daily_dose_avg"
basal_key = "total_daily_basal_insulin_avg"  # Total daily basal
carb_key = "total_daily_carb_avg"  # Total daily CHO
bmi_key = "bmi_at_baseline"
bmi_percentile = "bmi_perc_at_baseline"
isf_key = "isf"
icr_key = "carb_ratio"
age_key = "age_at_baseline"
tir_key = "percent_70_180_2week"

percent_cgm_available = "percent_cgm_available_2week"
below_40 = "percent_below_40_2week"
below_54 = "percent_below_54_2week"
percent_70_180 = "percent_70_180_2week"
days_insulin = "days_with_insulin"

# Filter to make sure basals are reasonable
df = df[(df[basal_key] > 1)]

# Filter for aspirational
# df = df[
#     (df[basal_key] > 1)
#     # Normal weight
#     & (df[bmi_key] < 25)
#     & (df[bmi_key] >= 18.5)
#     # Enough data to evaluate
#     & (df[percent_cgm_available] >= 90)
#     & (df[days_insulin] >= 14)
#     # Good CGM distributions
#     & (df[below_40] == 0)
#     & (df[below_54] < 1)
#     & (df[percent_70_180] >= 70)
# ]

# Non-Aspirational
# df = df[
#     (df[basal_key] > 1)
#     # Normal weight
#     | (df[bmi_key] >= 25)
#     | (df[bmi_key] < 18.5)
#     # Enough data to evaluate
#     | (df[percent_cgm_available] >= 90)
#     | (df[days_insulin] >= 14)
#     # Good CGM distributions
#     | (df[below_40] != 0)
#     | (df[below_54] >= 1)
#     | (df[percent_70_180] <= 70)
# ]

""" Basal Analysis """
df["jaeb_predicted_basals"] = df.apply(
    lambda x: utils.jaeb_basal_equation(x[tdd_key], x[carb_key]), axis=1
)
df["traditional_predicted_basals"] = df.apply(
    lambda x: utils.traditional_basal_equation(x[tdd_key]), axis=1
)

df["jaeb_basal_residual"] = df[basal_key] - df["jaeb_predicted_basals"]
df["traditional_basal_residual"] = df[basal_key] - df["traditional_predicted_basals"]

jaeb_basal_sum_squared_errors = sum(df["jaeb_basal_residual"] ** 2)
traditional_basal_sum_squared_errors = sum(df["traditional_basal_residual"] ** 2)

jaeb_basal_aic, jaeb_basal_bic = utils.aic_bic(
    df.shape[0], 2, jaeb_basal_sum_squared_errors
)
traditional_basal_aic, traditional_basal_bic = utils.aic_bic(
    df.shape[0], 1, traditional_basal_sum_squared_errors
)

output_df.loc["Basal"] = [
    jaeb_basal_aic,
    traditional_basal_aic,
    (jaeb_basal_sum_squared_errors / df.shape[0]) ** 0.5,
    (traditional_basal_sum_squared_errors / df.shape[0]) ** 0.5,
    jaeb_basal_sum_squared_errors,
    traditional_basal_sum_squared_errors,
    jaeb_basal_bic,
    traditional_basal_bic,
]

""" ISF Analysis """
df["jaeb_predicted_isf"] = df.apply(
    lambda x: utils.jaeb_isf_equation(x[tdd_key], x[bmi_key]), axis=1
)
df["traditional_predicted_isf"] = df.apply(
    lambda x: utils.traditional_isf_equation(x[tdd_key]), axis=1
)
df = df.dropna(subset=["jaeb_predicted_isf", "traditional_predicted_isf"])

df["jaeb_isf_residual"] = df[isf_key] - df["jaeb_predicted_isf"]
df["traditional_isf_residual"] = df[isf_key] - df["traditional_predicted_isf"]

jaeb_isf_sum_squared_errors = sum(df["jaeb_isf_residual"] ** 2)
traditional_isf_sum_squared_errors = sum(df["traditional_isf_residual"] ** 2)

jaeb_isf_aic, jaeb_isf_bic = utils.aic_bic(df.shape[0], 2, jaeb_isf_sum_squared_errors)
traditional_isf_aic, traditional_isf_bic = utils.aic_bic(
    df.shape[0], 1, traditional_isf_sum_squared_errors
)

output_df.loc["ISF"] = [
    jaeb_isf_aic,
    traditional_isf_aic,
    (jaeb_isf_sum_squared_errors / df.shape[0]) ** 0.5,
    (traditional_isf_sum_squared_errors / df.shape[0]) ** 0.5,
    jaeb_isf_sum_squared_errors,
    traditional_isf_sum_squared_errors,
    jaeb_isf_bic,
    traditional_isf_bic,
]

""" ICR Analysis """
df["jaeb_predicted_icr"] = df.apply(
    lambda x: utils.jaeb_icr_equation(x[tdd_key], x[carb_key]), axis=1
)
df["traditional_predicted_icr"] = df.apply(
    lambda x: utils.traditional_icr_equation(x[tdd_key]), axis=1
)

df["jaeb_icr_residual"] = df[icr_key] - df["jaeb_predicted_icr"]
df["traditional_icr_residual"] = df[icr_key] - df["traditional_predicted_icr"]

jaeb_icr_sum_squared_errors = sum(df["jaeb_icr_residual"] ** 2)
traditional_icr_sum_squared_errors = sum(df["traditional_icr_residual"] ** 2)

jaeb_icr_aic, jaeb_icr_bic = utils.aic_bic(df.shape[0], 2, jaeb_icr_sum_squared_errors)
traditional_icr_aic, traditional_icr_bic = utils.aic_bic(
    df.shape[0], 1, traditional_icr_sum_squared_errors
)

output_df.loc["ICR"] = [
    jaeb_icr_aic,
    traditional_icr_aic,
    (jaeb_icr_sum_squared_errors / df.shape[0]) ** 0.5,
    (traditional_icr_sum_squared_errors / df.shape[0]) ** 0.5,
    jaeb_icr_sum_squared_errors,
    traditional_icr_sum_squared_errors,
    jaeb_icr_bic,
    traditional_icr_bic,
]

output_df.to_csv(
    utils.get_save_path_with_file(
        input_file_name, analysis_name, "equation_errors.csv", "data-processing"
    )
)
df.to_csv(
    utils.get_save_path_with_file(
        input_file_name,
        analysis_name,
        "data_with_equation_predictions.csv",
        "data-processing",
    )
)
