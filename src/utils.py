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


def box_plot(y, x_tick_labels=None, axis_labels=["", ""], title=""):
    """ Plot a box plot """
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.boxplot(y)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if x_tick_labels:
        assert len(x_tick_labels) == len(y)
        ax.set_xticklabels(x_tick_labels)

    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.title(title, fontsize=30)

    plt.show()

def generate_boxplot_data(df, y_data_key, range, x_data_key=None):
    """
    df - dataframe with Jaeb data
    y_data_key - string of column that should have the boxplot generated
    range - range iterable to use to determine windows
    x_data_key - string of column that have boxplots generated over specific 
    ranges of this column
    """
    '''
    If we only want to plot 1 datatype on the box plot 
    (like just BMI box plot instead of a series of BMI 
    plots based on particular ages), then x_data_key should
    be same as y_data_key
    '''
    if not x_data_key:
        x_data_key = y_data_key

    boxplot_data = []
    for val in range:
        filtered = df[(df[x_data_key] >= val) & (df[x_data_key] < val + 1)]
        boxplot_data.append(filtered[y_data_key].tolist())
    
    ticks = [str(val) for val in range]
    return (boxplot_data, ticks)
