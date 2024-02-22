import pandas as pd
import pickle
from math import radians, sin, cos, sqrt, asin

class PredictPipeline:
    """
    A class for predicting listing rent prices using a pre-trained model and preprocessing pipeline

    Methods:
    - __init__: Initializes the PredictPipeline object by loading the preprocessor and model from .pkl files
    - process_dataset: Processes the input dataset, ensuring it contains the required columns and reordering them if necessary
    - calculate_distance: Calculates the distance between two points on Earth using the Haversine formula
    - get_feature_names: Retrieves the feature names after preprocessing
    - preprocess_data: Preprocesses the input data, including feature engineering and transformation
    - predict: Predicts rent prices based on the input data
    """
    def __init__(self):
        """
        Initializes the PredictPipeline object by loading the preprocessor and model from .pkl files
        """
        # Load preprocessor
        preprocessor_path = 'app/artifacts/preprocessor.pkl'
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)

        # Load model
        model_path = 'app/artifacts/model.pkl'
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def process_dataset(self, input_data):
        """
        Processes the input dataset, ensuring it contains the required columns and reordering them if necessary

        Parameters:
        - input_data: The input dataset to be processed

        Returns:
        - pandas.DataFrame: The processed dataset
        """
        columns = ['bairro_group', 'bairro', 'latitude', 'longitude', 'room_type',
                    'minimo_noites', 'numero_de_reviews', 'reviews_por_mes',
                    'calculado_host_listings_count']
        input_data.columns = input_data.columns.str.lower().str.replace(r'\[.*\]', '', regex=True).str.rstrip().str.replace(' ', '_')

        # Check if the uploaded dataset contains all required columns
        if not set(columns).issubset(input_data.columns):
            raise ValueError("Dataset must contain all the columns listed above")

        filtered_data = input_data[columns]
        reordered_data = filtered_data.reindex(columns=columns)
        return reordered_data

    def calculate_distance(self, lat2, lon2):
        """
        Calculates the distance between two points on Earth using the Haversine formula

        Parameters:
        - lat2: Latitude of the second point in degrees
        - lon2: Longitude of the second point in degrees

        Returns:
        - The distance between the two points in kilometers
        """ 
        # Convert coordinates into radians
        city_center_lat = radians(40.71427)
        city_center_lon = radians(-74.00597)
        lat2 = radians(lat2)
        lon2 = radians(lon2)

        earth_radius = 6371

        # Calculate the difference in latitude and longitude
        difference_lat = lat2 - city_center_lat
        difference_lon = lon2 - city_center_lon

        # Haversine formula
        a = sin(difference_lat / 2) * sin(difference_lat / 2) + cos(city_center_lat) * cos(lat2) * sin(difference_lon / 2) * sin(difference_lon / 2)
        central_angle = 2 * asin(sqrt(a))
        distance = earth_radius * central_angle

        return distance

    def get_feature_names(self, cat_cols):
        """
        Retrieves the feature names after preprocessing

        Parameters:
        - cat_cols: Categorical columns in the dataset

        Returns:
        - list: List of feature names
        """
        log_cols = ["minimo_noites", "numero_de_reviews", "calculado_host_listings_count"]
        cbrt_cols = ["reviews_por_mes", "distance_to_city_center"]
        coordinate_cols = ["latitude", "longitude"]
        numeric_features = coordinate_cols + log_cols + cbrt_cols

        # Get the categorical features names
        categorical_features = list(self.preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(cat_cols))

        feature_names = numeric_features + categorical_features
        return feature_names

    def preprocess_data(self, data):
        """
        Preprocesses the input data, including feature engineering and transformation

        Parameters:
        - data: The input data to be preprocessed

        Returns:
        - pandas.DataFrame: The preprocessed data
        """
        # Create the distance_to_city_center column and select the categorical columns
        data['distance_to_city_center'] = data.apply(lambda row: self.calculate_distance(row['latitude'], row['longitude']), axis=1)
        cat_cols = data.select_dtypes("object").columns

        # Transform the data with the preprocessor
        data = self.preprocessor.transform(data)
        data = data.toarray()
        feature_names = self.get_feature_names(cat_cols)

        # Create a DataFrame with the transformed data and feature names
        new_data = pd.DataFrame(data, columns=feature_names)
        return new_data

    def predict(self, data, manual=False):
        """
        Predicts rent prices based on the input data

        Parameters:
        - data: The input data for prediction
        - manual: A boolean indicating whether it's a manual prediction or not. Default is False

        Returns:
        - float or list: The predicted rent price(s)
        """
        prediction = self.model.predict(self.preprocess_data(data))
        if manual:
            return prediction[0]

        # If predicting from a dataset
        predictions = [pred for pred in prediction]
        return predictions

class CustomData:
    """
    A class representing custom datasets

    Attributes:
    - bairro_group: The neighborhood group of the ad
    - bairro: The neighborhood of the ad
    - latitude: The latitude coordinate of the ad
    - longitude: The longitude coordinate of the ad
    - room_type: The type of room in the ad
    - minimo_noites: The minimum number of nights required for booking the ad
    - numero_de_reviews: The number of reviews for the ad
    - reviews_por_mes: The number of reviews per month for the ad
    - calculado_host_listings_count: The calculated host listings count for the ad
    """
    def __init__(self, bairro_group: str,
                    bairro: str,
                    latitude: float,
                    longitude: float,
                    room_type: str,
                    minimo_noites: int,
                    numero_de_reviews: int,
                    reviews_por_mes: float,
                    calculado_host_listings_count: int):
        """
        Initializes the CustomData object with the provided attributes
        """
        self.bairro_group = bairro_group
        self.bairro = bairro
        self.latitude = latitude
        self.longitude = longitude
        self.room_type = room_type
        self.minimo_noites = minimo_noites
        self.numero_de_reviews = numero_de_reviews
        self.reviews_por_mes = reviews_por_mes
        self.calculado_host_listings_count = calculado_host_listings_count

    def get_data_as_dataframe(self):
        """
        Converts the CustomData object into a pandas DataFrame

        Returns:
        - pd.DataFrame: The CustomData object as a DataFrame
        """
        custom_data_input_dict = {
            "bairro_group": [self.bairro_group],
            "bairro": [self.bairro],
            "latitude": [self.latitude],
            "longitude": [self.longitude],
            "room_type": [self.room_type],
            "minimo_noites": [self.minimo_noites],
            "numero_de_reviews": [self.numero_de_reviews],
            "reviews_por_mes": [self.reviews_por_mes],
            "calculado_host_listings_count": [self.calculado_host_listings_count]
        }
        return pd.DataFrame(custom_data_input_dict)