import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error
import math
import os
from pathlib import Path
import subprocess
import datetime as dt


def rmse(y, y_predict):
    """ RMSE function, as Python doesn't have a library func for it """
    return ((y - y_predict) ** 2).mean() ** 0.5


def filter_data_for_equations_adult(df):
    return df[
        (df.total_daily_basal_insulin_avg > 1)
        # Normal weight
        & (df.bmi_at_baseline_at_baseline < 25)
        & (df.bmi_at_baseline_at_baseline >= 18.5)
        # Enough data to evaluate
        & (df.percent_cgm_available_2week >= 90)
        & (df.days_with_insulin >= 14)
        # Good CGM distributions
        & (df.percent_below_40_2week == 0)
        & (df.percent_below_54_2week < 1)
        & (df.percent_70_180_2week >= 70)
        & (df.percent_above_250_2week < 5)
    ]


def filter_data_for_equations_peds(df):
    # TODO: use BMI percentiles
    return df[
        (df.total_daily_basal_insulin_avg > 1)
        # Normal weight
        & (df.bmi_at_baseline_at_baseline < 25)
        & (df.bmi_at_baseline_at_baseline >= 18.5)
        # Enough data to evaluate
        & (df.percent_cgm_available_2week >= 90)
        & (df.days_with_insulin >= 14)
        # Good CGM distributions
        & (df.percent_below_40_2week == 0)
        & (df.percent_below_54_2week < 1)
        & (df.percent_70_180_2week >= 70)
        & (df.percent_above_250_2week < 5)
    ]


def filter_data_for_equation_verification(df):
    """
    Filter to get data that's not ideal but close to it,
    for the purpose of testing the equations
    """
    return df[
        (df.age_at_baseline >= 18)
        & (df.bmi_at_baseline > 18.5)
        & (df.bmi_at_baseline < 25)
        & (df.percent_cgm_available_2week >= 90)
        & (df.percent_70_180_2week >= 70)
        & (df.percent_below_54_2week > 1)
        & (df.percent_below_54_2week < 1.5)
        & (df.percent_below_40_2week == 0)
        & (df.days_with_insulin >= 14)
    ]


def filter_data_for_peds_equation_verification(df):
    """
    Filter to get pediatric data that's not ideal but close to it,
    for the purpose of testing the equations
    """
    df = df.dropna(subset=["basal_rate_schedule"])
    return df[
        (df.age_at_baseline < 18)
        & (df.percent_cgm_available_2week >= 90)
        & (df.percent_70_180_2week >= 70)
        & (df.percent_below_54_2week > 1)
        & (df.percent_below_54_2week < 1.5)
        & (df.percent_below_40_2week == 0)
        & (df.days_with_insulin >= 14)
    ]


def filter_data_for_non_ideal_settings(df):
    """ Filter to get settings that likely aren't correct """
    return df[
        (df.age_at_baseline >= 18)
        & (df.percent_cgm_available_2week >= 90)
        & (df.percent_below_54_2week > 5)
        & (df.days_with_insulin >= 14)
    ]


def three_dimension_plot(x, y, z, labels=["", "", ""], title=""):
    """
    Function to plot a 3D graph of data, with optional labels & a title
    """
    assert len(labels) == 3

    fig = plt.figure(1, figsize=(7, 7))
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    ax.scatter(x, y, z, edgecolor="k", s=50)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

    plt.title(title, fontsize=30)
    plt.show()


def two_dimension_plot(x, y, labels=["", ""], title="", ylim=None):
    """
    Function to plot a 2D graph of data, with optional labels & a title
    """
    assert len(labels) == 2

    fig = plt.figure(1, figsize=(7, 7))
    plt.scatter(x, y, edgecolor="k", s=50)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    axes = plt.gca()
    if ylim:
        axes.set_ylim(ylim)

    plt.title(title, fontsize=30)
    plt.show()


def log_likelihood(n, k, sum_squared_errors):
    """ 
    Find the maximum log likelihood for a *normal* distribution 
    Note: formula is from
    https://www.projectrhea.org/rhea/index.php/Maximum_Likelihood_
    Estimation_Analysis_for_various_Probability_Distributions
    """
    ll = -(n / 2) * (1 + np.log(2 * np.pi)) - (n / 2) * np.log(sum_squared_errors / n)
    return ll


def aic_bic(n, k, sum_squared_errors):
    """ 
    Compute Akaike Information Criterion (AIC) & 
    Bayesian Information Criterion (BIC)
    """
    max_log_likelihood = log_likelihood(n, k, sum_squared_errors)
    aic = (2 * k) - 2 * max_log_likelihood
    bic = np.log(n) * k - 2 * max_log_likelihood
    return (aic, bic)


def jaeb_basal_equation(tdd, carbs):
    """ Basal equation fitted from Jaeb data """
    return 0.6507 * tdd * math.exp(-0.001498 * carbs)


def traditional_basal_equation(tdd):
    """ Traditional basal equation """
    return 0.5 * tdd


def jaeb_isf_equation(tdd, bmi):
    """ ISF equation fitted from Jaeb data """
    return 40250 / (tdd * bmi)


def traditional_isf_equation(tdd):
    """ Traditional ISF equation """
    return 1500 / tdd


def jaeb_icr_equation(tdd, carbs):
    """ ICR equation fitted from Jaeb data """
    return (1.31 * carbs + 136.3) / tdd


def traditional_icr_equation(tdd):
    """ Traditional ICR equation """
    return 500 / tdd


def find_full_path(resource_name, extension):
    """ Find file path, given name and extension
        example: "/home/pi/Media/tidepool_demo.json"

        This will return the *first* instance of the file

    Arguments:
    resource_name -- name of file without the extension
    extension -- ending of file (ex: ".json")

    Output:
    path to file
    """
    search_dir = Path(__file__).parent.parent
    for root, dirs, files in os.walk(search_dir):  # pylint: disable=W0612
        for name in files:
            (base, ext) = os.path.splitext(name)
            if base == resource_name and extension == ext:
                return os.path.join(root, name)

    raise Exception("No file found for specified resource name & extension")


def get_file_stamps():
    """
    Get context for information generated at runtime.
    """
    current_commit = (
        subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")
    )
    utc_string = dt.datetime.utcnow().strftime("%Y_%m_%d_%H")
    code_description = "v0_1"
    date_version_name = "{}-{}-{}".format(utc_string, code_description, current_commit)

    return date_version_name, utc_string, code_description, current_commit


def make_dir_if_it_doesnt_exist(dir_):
    if not os.path.isdir(dir_):
        os.makedirs(dir_)


def get_save_path(
    dataset_name, full_analysis_name, report_type="figures", root_dir=".reports"
):
    output_path = os.path.join(root_dir, dataset_name)
    date_version_name, _, _, _ = get_file_stamps()
    save_path = os.path.join(
        output_path, "{}-{}".format(full_analysis_name, date_version_name), report_type
    )

    make_dir_if_it_doesnt_exist(save_path)

    return save_path


def get_save_path_with_file(
    dataset_name,
    full_analysis_name,
    file_name,
    report_type="figures",
    root_dir=".reports",
):
    return os.path.join(
        get_save_path(dataset_name, full_analysis_name, report_type, root_dir),
        file_name,
    )


def get_demographic_export_path(demographic, dataset, analysis_name):
    """ Get file path for export of filtered demographic datasets """
    assert isinstance(demographic, DemographicSelection)
    file_name = demographic.name.lower() + "_" + dataset + ".csv"
    return get_save_path_with_file(dataset, analysis_name, file_name, "data-processing")


def get_figure_export_path(dataset, plot_title, analysis_name):
    """ Get file path for export of filtered demographic datasets """
    short_dataset = dataset[:20] if len(dataset) >= 20 else dataset
    file_name = plot_title + "_" + short_dataset + ".png"
    return get_save_path_with_file(dataset, analysis_name, file_name, "plots")
