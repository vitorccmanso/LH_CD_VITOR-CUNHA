import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from math import radians, sin, cos, sqrt, asin
from scipy.stats import skew, mode

class Plots:
    """
    A class for plotting data distributions and correlations

    Attributes:
    - data (DataFrame): The dataset to be visualized

    Methods:
    - __init__: Initialize the Plots object with a DataFrame
    - plot_transformed_distributions: Plot distributions of a numeric column and its transformed versions
    - plot_corr: Plot a heatmap of the correlation matrix for numerical columns in the DataFrame
    """
    def __init__(self, data):
        """
        Initialize the Plots object with a DataFrame

        Parameters:
        - data: DataFrame containing the data
        """
        self.data = data

    def plot_transformed_distributions(self, col):
        """
        Plot distributions of a numeric column and its transformed versions

        Parameters:
        - col (str): Name of the column to plot
        """
        data = self.data[col]
        fig, ax = plt.subplots(2, 2, figsize=(16, 8))
        ax = ax.ravel()

        # Plot original distribution
        sns.histplot(data, kde=True, ax=ax[0])
        ax[0].set_title(f"Original Distribution\n Skewness:{round(skew(data), 2)}")
        self.plot_legends(ax, 0, data)

        # Apply log transformation
        log_transformed = np.log1p(data)
        sns.histplot(log_transformed, kde=True, ax=ax[1])
        ax[1].set_title(f"Log Transformation\n Skewness: {round(skew(log_transformed), 2)}")
        self.plot_legends(ax, 1, log_transformed)

        # Apply square root transformation
        sqrt_transformed = np.cbrt(data)
        sns.histplot(sqrt_transformed, kde=True, ax=ax[2])
        ax[2].set_title(f"Cubic Transformation\n Skewness: {round(skew(sqrt_transformed), 2)}")
        self.plot_legends(ax, 2, sqrt_transformed)

        skewness_values = [skew(data), skew(log_transformed), skew(sqrt_transformed)]
        min_skew_idx = np.argmin(skewness_values)

        # Plot boxplot of the transformation with the lowest skewness
        if min_skew_idx == 0:
            sns.boxplot(x=data, ax=ax[3])
        elif min_skew_idx == 1:
            sns.boxplot(x=log_transformed, ax=ax[3])
        else:
            sns.boxplot(x=sqrt_transformed, ax=ax[3])
        ax[3].set_title(f"Boxplot: Best Distribution")
        plt.tight_layout()
        plt.show()

    def plot_legends(self, ax, pos, data):
        """
        Plot legends on a given axis for a specific plot position

        Parameters:
        - ax: Axis object to plot on
        - pos (int): Position in the subplot grid
        - data: Data for which legends are plotted
        """
        ax[pos].axvline(data.mean(), color='r', linestyle='--', label='Mean: {:.2f}'.format(data.mean()))
        ax[pos].axvline(mode(data)[0], color='g', linestyle='--', label='Mode: {:.2f}'.format(mode(data)[0]))
        ax[pos].axvline(data.median(), color='b', linestyle='--', label='Median: {:.2f}'.format(data.median()))
        ax[pos].legend()

    def plot_corr(self, method):
        """
        Plots a heatmap of the correlation matrix for numerical columns in the DataFrame

        Parameters:
        - method: Correlation method to use ('pearson', 'kendall', or 'spearman')
        """
        sns.heatmap(self.data.select_dtypes(include="number").corr(method=method), annot=True, fmt=".2f", cmap="RdYlGn")
        plt.show()

def calculate_distance(lat2, lon2):
    """
    Calculates the distance between two points on Earth using the Haversine formula

    Parameters:
    - lat2: Latitude of the second point in degrees
    - lon2: Longitude of the second point in degrees

    Returns:
    - The distance between the two points in kilometers
    """ 
    city_center_lat = radians(40.71427)
    city_center_lon = radians(-74.00597)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    earth_radius = 6371
    difference_lat = lat2 - city_center_lat
    difference_lon = lon2 - city_center_lon

    a = sin(difference_lat / 2) * sin(difference_lat / 2) + cos(city_center_lat) * cos(lat2) * sin(difference_lon / 2) * sin(difference_lon / 2)
    central_angle = 2 * asin(sqrt(a))

    distance = earth_radius * central_angle

    return distance

def saving_dataset(data, save_folder, save_filename):
    """
    Saves the dataset, and creates the specified folder if it doesn't exist

    Parameters:
    - data: DataFrame containing the original dataset
    - save_folder: Folder path where the datasets will be saved
    - save_filename: Base filename for the saved datasets
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    original_save_path = os.path.join(save_folder, f"{save_filename}.csv")
    data.to_csv(original_save_path, index=False)