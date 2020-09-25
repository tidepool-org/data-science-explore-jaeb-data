import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from enum import Enum
import subprocess
import datetime as dt


class DemographicSelection(Enum):
    OVERALL = 1
    PEDIATRIC = 2
    ADULT = 3
    ASPIRATIONAL = 4
    NON_ASPIRATIONAL = 5


def extract_bmi_percentile(s):
    """
    Extract a bmi percentile from a string.
    Precondition: string must be in format 'number%' (ex: '20%')
    """
    return int(s[:-1])


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


def box_plot(
    data_to_plot,
    group_labels=None,
    data_axis_labels=["", ""],
    title="",
    should_be_vertical=False,
    should_export=False,
):
    """ 
    Plot a box plot 
    data_to_plot - data that should be plotted
    group_labels - if data is in particular groupings, these would be their labels
    data_axis_labels - x/y axis labels
    title - title of plot
    should_be_vertical - orientation of plot
    """
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.boxplot(data_to_plot, vert=should_be_vertical)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if group_labels:
        assert len(group_labels) == len(data_to_plot)
        ax.set_xticklabels(group_labels)

    plt.xlabel(data_axis_labels[0])
    plt.ylabel(data_axis_labels[1])
    plt.title(title, fontsize=30)

    if should_export:
        file_name = title + ".png"
        plt.savefig(get_save_path(file_name, ["results", "figures"]))
    else:
        plt.show()


def generate_boxplot_data(df, y_data_key, range, x_data_key=None, interval=1):
    """
    df - dataframe with Jaeb data
    y_data_key - string of column that should have the boxplot generated
    range - range iterable to use to determine windows
    x_data_key - string of column that have boxplots generated over specific 
    ranges of this column
    interval - interval to make windows
    """
    """
    If we only want to plot 1 datatype on the box plot 
    (like just BMI box plot instead of a series of BMI 
    plots based on particular ages), then x_data_key should
    be same as y_data_key
    """
    if not x_data_key:
        x_data_key = y_data_key

    boxplot_data = []
    for val in range:
        filtered = df[(df[x_data_key] >= val) & (df[x_data_key] < val + interval)]
        boxplot_data.append(filtered[y_data_key].tolist())

    ticks = [str(val) for val in range]
    return (boxplot_data, ticks)


def plot_by_frequency(
    df, column_key, title="", x_axis_label="", x_lim=None, bins=10, export_path=""
):
    """
    df - dataframe containing column titled 'column_key'
    column_key - title of column to plot frequency
    title - title of plot
    x_axis_label - label of x axis
    x_lim - list of x limits in form [left, right]
    bins - number of bins to group data into
    should_export - set true to save the plot to png without plotting it
    """
    plt.hist(df[column_key], bins=bins)
    plt.title(title, fontsize=30)
    plt.xlabel(x_axis_label)
    plt.ylabel("Count of Occurrences")
    if x_lim:
        plt.xlim(x_lim[0], x_lim[1])
    if len(export_path) > 0:
        plt.savefig(export_path)
        plt.clf()
    else:
        plt.show()


def find_bmi(row):
    """
    Assumes height is cm & in first column, and weight is in pounds & in second column
    """
    kgs = row[1]
    meters = row[0] / 100
    return kgs / (meters * meters)


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
    dataset_name, full_analysis_name, report_type="figures", root_dir="results"
):
    output_path = os.path.join("..", root_dir, dataset_name)
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
    root_dir="results",
):
    return os.path.join(
        get_save_path(dataset_name, full_analysis_name, report_type, root_dir),
        file_name,
    )


def save_df(df_results, analysis_name, save_dir, save_type="tsv"):
    utc_string = dt.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S")
    filename = "{}-created_{}".format(analysis_name, utc_string)
    path = os.path.join(save_dir, filename)
    if "tsv" in save_type:
        df_results.to_csv("{}.tsv".format(path), sep="\t")
    else:
        df_results.to_csv("{}.csv".format(path))


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
