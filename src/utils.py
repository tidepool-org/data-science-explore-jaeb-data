import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def filter_data_for_equations(df):
    df = df.dropna(subset=["basal_rate_schedule"])
    return df[
        (df.ageAtBaseline >= 18) 
        & (df.bmi > 18.5) 
        & (df.bmi < 25)
        & (df.percent_cgm_available >= 90)
        & (df.percent_70_180 >= 70)
        & (df.percent_below_54 < 1)
        & (df.percent_below_40 == 0)
        & (df.days_with_carbs >= 14)
        & (df.days_with_insulin >= 14)
        & (df.days_with_basals >= 14)
    ]

def three_dimension_plot(x, y, z, labels=["", "", ""], title=""):
    """
    Function to plot a 3D graph of data, with optional labels & a title
    """
    assert len(labels) == 3

    fig = plt.figure(1, figsize=(7,7))
    ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
    ax.scatter(x, y, z, edgecolor="k", s=50)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

    plt.title(title, fontsize=14)
    plt.show()

def two_dimension_plot(x, y, labels=["", ""], title=""):
    """
    Function to plot a 2D graph of data, with optional labels & a title
    """
    assert len(labels) == 2

    fig = plt.figure(1, figsize=(7,7))
    plt.scatter(x, y, edgecolor="k", s=50)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    plt.title(title, fontsize=14)
    plt.show()

