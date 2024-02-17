import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualization:
    def __init__(self, data):
        self.data = data

    def create_subplots(self, rows, columns, figsize=(18,12)):
        """
        Creates a figure and subplots with common settings

        Parameters:
        - rows: Number of rows in the subplot grid
        - columns: Number of columns in the subplot grid
        - figsize: Tuple specifying the width and height of the figure (default is (18, 12))
        
        Returns:
        - fig: The Matplotlib figure object
        - ax: A 1D NumPy array of Matplotlib axes
        """
        fig, ax = plt.subplots(rows, columns, figsize=figsize)
        ax = ax.ravel()
        return fig, ax

    def plot_columns(self, cols, plot_func, ax, title_prefix="", target=None, x=None):
        """
        Plots specified graphs using the given plotting function

        Parameters:
        - data: DataFrame containing the data to be plotted
        - cols: List of column names to be plotted
        - plot_func: The Seaborn plotting function to be used
        - ax: Matplotlib axes to plot on
        - title_prefix: Prefix to be added to the title of each subplot (default is an empty string)
        - x: Optional parameter for the x-axis when plotting scatter plots (default is None)
        """
        for i, col in enumerate(cols):
            if plot_func == sns.boxplot and target is not None:
                value_counts = self.data[col].value_counts()
                if len(value_counts) > 10:
                    top_values = value_counts.head(10)
                    plot_func(x=self.data[col], y=self.data[target], order=top_values.index, ax=ax[i])
                    ax[i].tick_params(axis='x', rotation=45)
                else:
                    plot_func(x=self.data[col], y=self.data[target], ax=ax[i])
            elif plot_func == sns.histplot:
                plot_func(self.data[col], ax=ax[i], kde=True)
            elif plot_func == sns.scatterplot and target is not None:
                plot_func(x=self.data[col], y=self.data[target], ax=ax[i])
            elif plot_func == sns.countplot:
                value_counts = self.data[col].value_counts()
                if len(value_counts) > 10:
                    top_values = value_counts.head(10)
                    plot_func(x=self.data[col], order=top_values.index, ax=ax[i])
                    ax[i].tick_params(axis='x', rotation=90)
                else:
                    plot_func(x=self.data[col], ax=ax[i])
            else:
                plot_func(x=self.data[col], ax=ax[i])
            ax[i].set_title(f"{title_prefix}{col.capitalize()}")

    def remove_unused_axes(self, fig, ax, num_plots):
        """
        Removes unused axes from a figure

        Parameters:
        - fig: The Matplotlib figure object
        - ax: A 1D NumPy array of Matplotlib axes
        - num_plots: Number of subplots to keep; remove the rest
        """
        total_axes = len(ax)
        for j in range(num_plots, total_axes):
            fig.delaxes(ax[j])

    def numerical_univariate_analysis(self, rows, columns):
        """
        Performs univariate analysis on numerical columns

        Parameters:
        - data: DataFrame containing numerical data for analysis
        - rows: Number of rows in the subplot grid
        - columns: Number of columns in the subplot grid

        Effect:
        - Plots the distribution of numerical columns using seaborn's histplot
        """
        fig, ax = self.create_subplots(rows, columns)
        cols = self.data.select_dtypes(include="number")
        self.plot_columns(cols, sns.histplot, ax, title_prefix="Distribution of ")
        self.remove_unused_axes(fig, ax, cols.shape[1])
        plt.tight_layout()
        plt.show()


    def categorical_univariate_analysis(self, rows, columns):
        """
        Performs univariate analysis on categorical columns

        Parameters:
        - data: DataFrame containing categorical data for analysis
        - rows: Number of rows in the subplot grid
        - columns: Number of columns in the subplot grid

        Effect:
        - Plots countplots for each categorical column
        """
        fig, ax = self.create_subplots(rows, columns)
        cols = self.data.select_dtypes("object")
        self.plot_columns(cols, sns.countplot, ax)
        self.remove_unused_axes(fig, ax, cols.shape[1])
        plt.tight_layout()
        plt.show()

    def num_features_vs_target(self, rows, columns, target):
        """
        Plots numerical features against the target variable

        Parameters:
        - data: DataFrame containing both features and target variable
        - rows: Number of rows in the subplot grid
        - columns: Number of columns in the subplot grid

        Effect:
        - Plots boxplots for numerical features against the target variable
        - Plots a countplot for the target variable against the 'type' column
        """
        fig, ax = self.create_subplots(rows, columns, figsize=(18, 12))
        cols = self.data.drop(columns=target).select_dtypes(include="number")
        self.plot_columns(cols, sns.scatterplot, ax, "Price x ", target=target)
        self.remove_unused_axes(fig, ax, cols.shape[1])
        plt.tight_layout()
        plt.show()

    def facegrid_hist_target(self, facecol, target):
        """
        Generates FacetGrid histograms for numerical columns based on target values

        Parameters:
        - df: DataFrame containing data for analysis
        - facecol: Column for creating facets in the FacetGrid
        - color: Color for the histograms

        Effect:
        - Creates a FacetGrid for each column based on the unique values in the specified facecol
        - Filters the data to include only rows where the "target" column is equal to 1
        - Shows the resulting FacetGrids with histograms
        """
        for col in self.data.drop(columns=[target]).select_dtypes(include="number"):
            g = sns.FacetGrid(self.data, col=facecol)
            g.map(sns.scatterplot, col, target)
            plt.show()

    def cat_features_vs_target(self, rows, columns, target, cols, figsize):
        """
        Plots scatter plots of numerical columns against x column for target value 1

        Parameters:
        - data: DataFrame containing data for analysis
        - rows: Number of rows in the subplot grid
        - columns: Number of columns in the subplot grid
        - x: Column for the x-axis in scatter plots

        Effect:
        - Plots scatter plots using seaborn's scatterplot for each numerical column against x column
        """
        fig, ax = self.create_subplots(rows, columns, figsize=figsize)
        cols = self.data[cols]
        self.plot_columns(cols, sns.boxplot, ax, "Price x ", target=target)
        self.remove_unused_axes(fig, ax, cols.shape[1])
        plt.tight_layout()
        plt.show()

    def features_boxplots(self, rows, columns, cols, figsize):
        fig, ax = self.create_subplots(rows, columns, figsize=figsize)
        cols = self.data[cols]
        self.plot_columns(cols, sns.boxplot, ax, "Boxplot ")
        self.remove_unused_axes(fig, ax, cols.shape[1])
        plt.tight_layout()
        plt.show()
        self.calculate_whiskers(cols)

    def calculate_whiskers(self, cols):
        upper_whiskers = {}
        for col in cols:
            q1 = np.percentile(self.data[col], 25)
            q3 = np.percentile(self.data[col], 75)
            iqr = q3 - q1
            upper_whisker = q3 + 1.5 * iqr
            upper_whiskers[col] = upper_whisker
        print("Upper whiskers:", upper_whiskers)

    def plot_scatter_numericals_target(self, rows, columns, target, x):
        """
        Plots scatter plots of numerical columns against x column for target value 1

        Parameters:
        - data: DataFrame containing data for analysis
        - rows: Number of rows in the subplot grid
        - columns: Number of columns in the subplot grid
        - x: Column for the x-axis in scatter plots

        Effect:
        - Plots scatter plots using seaborn's scatterplot for each numerical column against x column
        """
        fig, ax = self.create_subplots(rows, columns, figsize=(18, 12))
        cols = self.data.drop(columns=[target, x]).select_dtypes(include='number')
        for i, col in enumerate(cols):
            im = ax[i].scatter(y=self.data[col], x=self.data[x], c=self.data[target], cmap='tab20c', label='price', s=10)
            cbar = fig.colorbar(im, ax=ax[i], label='Price')
            ax[i].set_xlabel(x)
            ax[i].set_ylabel(col)
            ax[i].set_title(f"{x.capitalize()} x {col.capitalize()}")
        self.remove_unused_axes(fig, ax, cols.shape[1])
        plt.tight_layout()
        plt.show()
