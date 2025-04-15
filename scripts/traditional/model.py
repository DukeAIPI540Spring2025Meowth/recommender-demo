import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

from scripts.etl.etl import extract, get_train_test_splits
from scripts.util.model import Model


class TraditionalModel(Model):
    """
    TraditionalModel: A KNN-based regression model for predicting user-recipe ratings.
    This implementation uses GridSearchCV to optimize hyperparameters and includes
    feature engineering based on user and recipe average ratings.
    """

    def __init__(self):
        """
        Initializing the TraditionalModel by loading and preparing the data,
        initializing encoders and scalers.
        """
        self.model = None
        self.scaler = StandardScaler()
        self.user_encoder = LabelEncoder()
        self.recipe_encoder = LabelEncoder()
        self.recipes_df, self.reviews_df = extract()

    def __train(self):
        """
        Training the model using GridSearchCV on KNeighborsRegressor.
        Includes encoding and scaling, and adds additional features such as
        user and recipe average ratings.
        """
        df = self.reviews_df.copy()

        # Feature engineering
        user_avg_rating = df.groupby('user_id')['rating'].mean().rename('user_avg_rating')
        recipe_avg_rating = df.groupby('recipe_id')['rating'].mean().rename('recipe_avg_rating')
        df = df.merge(user_avg_rating, on='user_id', how='left')
        df = df.merge(recipe_avg_rating, on='recipe_id', how='left')

        # Encoding user and recipe IDs
        df['user_id_enc'] = self.user_encoder.fit_transform(df['user_id'])
        df['recipe_id_enc'] = self.recipe_encoder.fit_transform(df['recipe_id'])

        # Preparing features and target
        features = ['user_id_enc', 'recipe_id_enc', 'user_avg_rating', 'recipe_avg_rating']
        X = df[features]
        y = df['rating']

        # Splitting into train, validation, test
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Scaling features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Defining model and parameter grid
        knn = KNeighborsRegressor()
        param_grid = {
            'n_neighbors': range(3, 21),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }

        # Hyperparameter tuning
        grid_search = GridSearchCV(
            knn,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train_scaled, y_train)
        self.model = grid_search.best_estimator_

        # Evaluating on test set
        preds = self.model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"TraditionalModel (GridSearch KNN) Test RMSE: {rmse:.4f}")

        self.__save()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicting ratings for given user-recipe pairs using the trained model.

        Parameters:
            X (pd.DataFrame): DataFrame with 'user_id' and 'recipe_id'

        Returns:
            np.ndarray: Predicted ratings
        """
        df = X.copy()

        # Generating average rating features on the fly
        df['user_avg_rating'] = df['user_id'].map(
            self.reviews_df.groupby('user_id')['rating'].mean()
        )
        df['recipe_avg_rating'] = df['recipe_id'].map(
            self.reviews_df.groupby('recipe_id')['rating'].mean()
        )

        # Filling missing values if any
        df['user_avg_rating'] = df['user_avg_rating'].fillna(4.57)
        df['recipe_avg_rating'] = df['recipe_avg_rating'].fillna(4.57)

        # Encoing categorical IDs
        df['user_id_enc'] = self.user_encoder.transform(df['user_id'])
        df['recipe_id_enc'] = self.recipe_encoder.transform(df['recipe_id'])

        # Scaling features
        features = ['user_id_enc', 'recipe_id_enc', 'user_avg_rating', 'recipe_avg_rating']
        X_scaled = self.scaler.transform(df[features])
        return self.model.predict(X_scaled)

    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> float:
        """
        Evaluating the model on a given dataset.

        Parameters:
            X (pd.DataFrame): Features (must contain user_id, recipe_id)
            y_true (pd.Series): True ratings

        Returns:
            float: Root Mean Squared Error (RMSE)
        """
        y_pred = self.predict(X)
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def __save(self):
        """
        Saving the trained model, encoders, and scaler to disk.
        """
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, "models/traditional_model.pkl")
        joblib.dump(self.scaler, "models/scaler.pkl")
        joblib.dump(self.user_encoder, "models/user_encoder.pkl")
        joblib.dump(self.recipe_encoder, "models/recipe_encoder.pkl")
        print("Model, scaler, and encoders saved.")

    @staticmethod
    def __load():
        """
        Loading the saved TraditionalModel from disk.

        Returns:
            TraditionalModel: A loaded model instance
        """
        model = TraditionalModel()
        model.model = joblib.load("models/traditional_model.pkl")
        model.scaler = joblib.load("models/scaler.pkl")
        model.user_encoder = joblib.load("models/user_encoder.pkl")
        model.recipe_encoder = joblib.load("models/recipe_encoder.pkl")
        print("TraditionalModel loaded from disk.")
        return model

    @staticmethod
    def __create():
        """
        Training a new TraditionalModel from scratch.

        Returns:
            TraditionalModel: A trained model instance
        """
        model = TraditionalModel()
        model.__train()
        return model

    @staticmethod
    def get_instance():
        """
        Loading an existing model if available; otherwise, train a new one.

        Returns:
            TraditionalModel: A ready-to-use model instance
        """
        try:
            return TraditionalModel.__load()
        except Exception as e:
            print(f"Could not load TraditionalModel. Reason: {e}")
            print("Training new TraditionalModel...")
            return TraditionalModel.__create()
