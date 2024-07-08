import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats
import pickle

mlflow_tracking_username = os.environ.get("MLFLOW_TRACKING_USERNAME")
mlflow_tracking_password = os.environ.get("MLFLOW_TRACKING_PASSWORD")

class DataPreprocess:
    """
    A class for preprocessing data including feature engineering, transformation, and splitting into train-test sets

    Methods:
    - __init__: Initializes the DataPreprocess object
    - save_preprocessor: Saves the preprocessor object to a file
    - get_feature_names: Retrieves the feature names after preprocessing
    - preprocessor: Creates and returns a preprocessor pipeline for data preprocessing
    - preprocess_data: Preprocesses the input data, including feature engineering, transformation, and splitting into train-test sets
    """
    def __init__(self):
        pass

    def save_preprocessor(self, preprocessor):
        """
        Saves the preprocessor object to a file

        Parameters:
        - preprocessor: The preprocessor object to be saved
        """
        if not os.path.exists("../artifacts"):
            os.makedirs("../artifacts")
        with open('../artifacts/preprocessor.pkl', 'wb') as f:
            pickle.dump(preprocessor, f)
    
    def get_feature_names(self, preprocessor, coordinate_cols, log_cols, cbrt_cols, cat_cols):
        """
        Retrieves the feature names after preprocessing

        Parameters:
        - preprocessor: The preprocessor object
        - coordinate_cols: List of column names representing coordinate features
        - log_cols: List of column names for which log transformation is applied
        - cbrt_cols: List of column names for which cubic root transformation is applied
        - cat_cols: List of column names representing categorical features

        Returns:
        - feature_names: List of feature names after preprocessing
        """
        numeric_features = coordinate_cols + log_cols + cbrt_cols
        categorical_features = list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols))
        feature_names = numeric_features + categorical_features
        return feature_names

    def preprocessor(self, coordinate_cols, log_cols, cbrt_cols, cat_cols):
        """
        Creates and returns a preprocessor pipeline for data preprocessing

        Parameters:
        - coordinate_cols: List of column names representing coordinate features
        - log_cols: List of column names for which log transformation is applied
        - cbrt_cols: List of column names for which cubic root transformation is applied
        - cat_cols: List of column names representing categorical features

        Returns:
        - preprocessor: Preprocessor pipeline for data preprocessing
        """
        # Define transformers for numeric columns
        log_transformer = Pipeline(steps=[
            ('log_transformation', FunctionTransformer(np.log1p, validate=True)),
            ("scaler", RobustScaler())
        ])
        cubic_transformer = Pipeline(steps=[
            ('sqrt_transformation', FunctionTransformer(np.cbrt, validate=True)),
            ("scaler", RobustScaler())
        ])

        #Define transformer for categorical columns
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder())
        ])

        # Combine transformers for numeric and categorical columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('coordinates', RobustScaler(), coordinate_cols),
                ('num_log', log_transformer, log_cols),
                ('num_sqrt', cubic_transformer, cbrt_cols), 
                ('cat', categorical_transformer, cat_cols)
            ], verbose_feature_names_out=False)
        return preprocessor

    def preprocess_data(self, data, test_size, target_name):
        """
        Preprocesses the input data, including feature engineering, transformation, and splitting into train-test sets

        Parameters:
        - data: Input DataFrame containing the raw data
        - test_size: The proportion of the dataset to include in the test split
        - target_name: Name of the target variable

        Returns:
        - X_train: Features of the training set
        - X_test: Features of the testing set
        - y_train: Target labels of the training set
        - y_test: Target labels of the testing set
        """
        data_process = data.drop(columns=[target_name])

        # Specify columns needing log transformation, square root transformation
        log_cols = ["minimo_noites", "numero_de_reviews", "calculado_host_listings_count"]
        cbrt_cols = ["reviews_por_mes", "distance_to_city_center"]
        coordinate_cols = ["latitude", "longitude"]
        cat_cols = data_process.select_dtypes("object").columns

        # Build preprocessor
        preprocessor = self.preprocessor(coordinate_cols, log_cols, cbrt_cols, cat_cols)

        # Fit and transform data
        data_preprocessed = preprocessor.fit_transform(data_process)
        data_preprocessed = data_preprocessed.toarray()

        feature_names = self.get_feature_names(preprocessor, coordinate_cols, log_cols, cbrt_cols, cat_cols)
        data_preprocessed = pd.DataFrame(data_preprocessed, columns=feature_names)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(data_preprocessed, data[target_name], test_size=test_size, shuffle=True, random_state=42)

        # Save preprocessor if not already saved
        if not os.path.exists("../artifacts/preprocessor.pkl"):
            self.save_preprocessor(preprocessor)

        return X_train, X_test, y_train, y_test

class ModelTraining:
    """
    A class for training machine learning models and evaluating their performance

    Methods:
    - __init__: Initializes the ModelTraining object
    - initiate_model_trainer: Initiates the model training process
    - evaluate_models: Evaluates multiple models using random search cross-validation and logs the results with MLflow
    """
    def __init__(self):
        pass

    def initiate_model_trainer(self, train_test, experiment_name):
        """
        Initiates the model training process

        Parameters:
        - train_test: A tuple containing the train-test split data in the format (X_train, y_train, X_test, y_test)
        - experiment_name: Name of the MLflow experiment where the results will be logged
        
        Returns:
        - dict: A dictionary containing the evaluation report for each model
        """
        mlflow.set_tracking_uri("https://dagshub.com/vitorccmanso/Rent-Price-Prediction.mlflow")
        X_train, y_train, X_test, y_test = train_test
        
        models = {
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor()
        }
        
        params = {
            "Ridge": {
                'alpha':[0.1, 0.5, 1, 10, 100], 
                'max_iter':[1000, 3000, 5000], 
                'tol': [0.0001, 0.001, 0.01, 0.1],
                "random_state": [42]
            },
            "Lasso":{
                'alpha':[0.1, 0.5, 1, 10, 100], 
                'max_iter':[1000, 3000, 5000], 
                'tol': [0.0001, 0.001, 0.01, 0.1],
                "random_state": [42]
            },
            "Random Forest":{
                "criterion":["squared_error", "absolute_error", "poisson"],
                "max_features":["sqrt","log2"],
                "n_estimators": [5,10,25,50,100],
                "max_depth": [5, 10, 20, 30],
                "random_state": [42]
            },
            "Gradient Boosting":{
                "loss":["squared_error", "absolute_error", "quantile"],
                "max_features":["sqrt","log2"],
                "n_estimators": [5,10,25,50,100],
                "max_depth": [5, 10, 20, 30],
                "learning_rate": [0.001, 0.01, 0.1],
                "random_state": [42]
            },
        }
        
        model_report = self.evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                           models=models, params=params, experiment_name=experiment_name)
        
        return model_report

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params, experiment_name):
        """
        Evaluates multiple models using random search cross-validation and logs the results with MLflow

        Parameters:
        - X_train: Features of the training data
        - y_train: Target labels of the training data
        - X_test: Features of the testing data
        - y_test: Target labels of the testing data
        - models: A dictionary containing the models to be evaluated
        - params: A dictionary containing the hyperparameter grids for each model
        - experiment_name: Name of the MLflow experiment where the results will be logged
        
        Returns:
        - dict: A dictionary containing the evaluation report for each model
        """
        mlflow.set_experiment(experiment_name)
        report = {}
        for model_name, model in models.items():
            with mlflow.start_run(run_name=model_name):
                param = params[model_name]
                rs = RandomizedSearchCV(model, param, cv=5, scoring=["neg_mean_absolute_error", "r2"], refit="neg_mean_absolute_error", random_state=42)
                search_result = rs.fit(X_train, y_train)
                model = search_result.best_estimator_
                y_pred = model.predict(X_test)

                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = root_mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Log metrics to MLflow
                mlflow.log_params(search_result.best_params_)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.sklearn.log_model(model, model_name, registered_model_name=f"{model_name} - {experiment_name}")
                
                # Store the model for visualization
                report[model_name] = {"model": model, "y_pred": y_pred, "mae": mae, "rmse": rmse, "r2": r2}        
        return report


class MetricsVisualizations:
    """
    A class for visualizing model evaluation metrics and results

    Attributes:
    - models: A dictionary containing the trained models

    Methods:
    - __init__: Initializes the MetricsVisualizations object with a dictionary of models
    - plot_pred_x_real: Plots predicted vs real values for each model
    - plot_feature_importance: Plots feature importance for each model
    - plot_residuals: Plots residuals, residual autocorrelation, and residual distribution for each model
    """
    def __init__(self, models):
        """
        Initializes the MetricsVisualizations object with a dictionary of models

        Parameters:
        - models: A dictionary containing the trained models
        """
        self.models = models

    def create_subplots(self, rows, columns, figsize=(18,12)):
        """
        Creates a figure and subplots with common settings

        Parameters:
        - rows: Number of rows for subplots grid
        - columns: Number of columns for subplots grid
        - figsize: Figure size. Default is (18, 12)
        
        Returns:
        - fig: The figure object
        - ax: Array of axes objects
        """
        fig, ax = plt.subplots(rows, columns, figsize=figsize)
        ax = ax.ravel()
        return fig, ax

    def plot_pred_x_real(self, y_test, rows, columns):
        """
        Plots predicted vs real values for each model

        Parameters:
        - y_test: True labels of the test data
        - rows: Number of rows for subplots
        - columns: Number of columns for subplots
        """
        fig, ax = self.create_subplots(rows, columns)
        for i, (model_name, model_data) in enumerate(self.models.items()):
            # Get predicted values and make a copy of y_test
            y_pred = pd.Series(model_data["y_pred"])
            y_test_copy = y_test.copy()

            # Reset indices for easy plotting
            y_pred.reset_index(drop=True, inplace=True)
            y_test_copy.reset_index(drop=True, inplace=True)

            # Create DataFrame for plotting
            df_plot = pd.DataFrame({'Predicted Values': y_pred.values, 'Real Values': y_test_copy.values})

            # Plot scatter plot and regression line
            sns.scatterplot(data=df_plot, x="Predicted Values", y="Real Values", ax=ax[i])
            sns.regplot(data=df_plot, x="Predicted Values", y="Real Values", ax=ax[i], scatter=False, color='red', line_kws={"linewidth": 2})
            ax[i].set_title(f"Predicted x Real Values: {model_name}")
            ax[i].set_xlabel("Predicted Values")
            ax[i].set_ylabel("Real Values")

        fig.tight_layout()
        plt.show()

    def plot_feature_importance(self, y_test, X_test, metric, rows, columns):
        """
        Plots feature importance for each model

        Parameters:
        - y_test: True labels of the test data
        - X_test: Features of the test data
        - metric: Metric used for evaluating feature importance
        - rows: Number of rows for subplots
        - columns: Number of columns for subplots
        """
        fig, ax = self.create_subplots(rows, columns)
        for i, (model_name, model_data) in enumerate(self.models.items()):
            # Calculate and sort permutation importances
            result = permutation_importance(model_data["model"], X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring=metric)
            sorted_importances_idx = result['importances_mean'].argsort()[::-1]

            # Select top 5 features
            top_features_idx = sorted_importances_idx[:5]
            top_features = X_test.columns[top_features_idx]
            importances = pd.DataFrame(result.importances[top_features_idx].T, columns=top_features)

            # Plot boxplot of feature importances
            box = importances.plot.box(vert=False, whis=10, ax=ax[i])
            box.set_title(f"Top 5 Feature Importance - {model_name}")
            box.axvline(x=0, color="k", linestyle="--")
            box.set_xlabel(f"Increase in MAE")
            box.figure.tight_layout()

        fig.tight_layout()
        plt.show()

    def plot_residuals(self, y_test, rows, columns):
        """
        Plots residuals, residual autocorrelation, and residual distribution for each model

        Parameters:
        - y_test: True labels of the test data
        - rows: Number of rows for subplots
        - columns: Number of columns for subplots
        """
        fig, ax = self.create_subplots(rows, columns)
        for i, (model_name, model_data) in enumerate(self.models.items()):
            # Calculate residuals
            y_pred = model_data["y_pred"]
            residuals = y_test - y_pred

            # Plot residual plot
            sns.scatterplot(x=y_pred, y=residuals, ax=ax[i * columns])
            ax[i * columns].set_xlabel('Predicted Values')
            ax[i * columns].set_ylabel('Residuals')
            ax[i * columns].axhline(y=0, color='r', linestyle='--')
            ax[i * columns].set_title(f'Residual Plot - {model_name}')

            # Plot residual autocorrelation
            plot_acf(residuals, lags=40, ax=ax[i * columns + 1])
            ax[i * columns + 1].set_title(f"Residual Autocorrelation - {model_name}")
            ax[i * columns + 1].set_xlabel("Lags")
            ax[i * columns + 1].set_ylabel("Autocorrelation")

            # Plot residual distribution
            stats.probplot(residuals, dist="norm", plot=ax[i * columns + 2])
            ax[i * columns + 2].set_title(f'Residual Distribution - {model_name}')

        fig.tight_layout()
        plt.show()