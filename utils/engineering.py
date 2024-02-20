import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import radians, sin, cos, sqrt, asin

def plot_corr(data, method):
    """
    Plots a heatmap of the correlation matrix for numerical columns in the DataFrame

    Parameters:
    - data: DataFrame for correlation analysis
    - method: Correlation method to use ('pearson', 'kendall', or 'spearman')
    """
    sns.heatmap(data.select_dtypes(include="number").corr(method=method), annot=True, fmt=".2f",cmap="RdYlGn")
    plt.show()

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