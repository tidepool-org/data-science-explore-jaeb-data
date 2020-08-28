import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

def plot_by_frequency(df, column_key, title="", x_axis_label="", bins=10):
    plt.hist(df[column_key], bins=bins)
    plt.title(title, fontsize=30)
    plt.xlabel(x_axis_label)
    plt.ylabel("Count of Occurrences")
    plt.show()

def find_bmi(row):
    """
    Assumes height is cm & in first column, and weight is in pounds & in second column
    """
    kgs = row[1]
    meters = row[0] / 100
    print(row)
    print(row[0], kgs, kgs / (meters * meters))
    print()
    return kgs / (meters * meters)