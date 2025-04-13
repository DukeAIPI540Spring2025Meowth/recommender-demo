import pandas as pd
import numpy as np
import joblib
import os
import sys
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Adding project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from scripts.util.model import Model
from scripts.etl.etl import extract, get_train_test_splits

class TraditionalModel(Model):
    """
    TraditionalModel implements a user-recipe rating predictor using XGBoost.
    It encodes user and recipe IDs, trains a regression model, and supports saving/loading.
    """

    def __init__(self):
        """
        Initialize the model and load raw data.

        Attributes:
            model: Trained XGBoostRegressor instance
            user_encoder: LabelEncoder for user_id
            item_encoder: LabelEncoder for recipe_id
            recipes_df: DataFrame of all recipes
            full_reviews_df: DataFrame of all interactions
        """
        self.model = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.recipes_df, self.full_reviews_df = extract()

    def __train(self):
        """
        Train the XGBoost regression model on encoded user-recipe pairs.
        Saves the trained model and encoders to disk.
        """
        X_train, X_test, y_train, y_test = get_train_test_splits()

        # Fitting encoders on the entire dataset to prevent unseen labels
        self.user_encoder.fit(self.full_reviews_df['user_id'])
        self.item_encoder.fit(self.full_reviews_df['recipe_id'])

        # Encoding user_id and recipe_id for training
        X_train_encoded = pd.DataFrame({
            'User': self.user_encoder.transform(X_train['user_id']),
            'Item': self.item_encoder.transform(X_train['recipe_id'])
        })
        X_test_encoded = pd.DataFrame({
            'User': self.user_encoder.transform(X_test['user_id']),
            'Item': self.item_encoder.transform(X_test['recipe_id'])
        })

        # Initializing and train XGBoost Regressor
        self.model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_encoded, y_train)

        # Evaluating the model performance
        preds = self.model.predict(X_test_encoded)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        # print(f"TraditionalModel trained. RMSE on test set: {rmse:.4f}")

        self.__save()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict ratings for a given set of user-recipe pairs.

        Args:
            X (pd.DataFrame): A DataFrame with two columns:
                - 'user_id': The user identifier
                - 'recipe_id': The recipe identifier

        Returns:
            np.ndarray: An array of predicted ratings.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        # Encoding user and recipe IDs before prediction
        X_encoded = pd.DataFrame({
            'User': self.user_encoder.transform(X['user_id']),
            'Item': self.item_encoder.transform(X['recipe_id'])
        })

        return self.model.predict(X_encoded)

    def __save(self):
        """
        Save the trained model and encoders to disk under scripts/models/
        """
        os.makedirs("scripts/models", exist_ok=True)
        joblib.dump(self.model, "scripts/models/traditional_model.pkl")
        joblib.dump(self.user_encoder, "scripts/models/user_encoder.pkl")
        joblib.dump(self.item_encoder, "scripts/models/item_encoder.pkl")
        print(" TraditionalModel and encoders saved to scripts/models/")

    @staticmethod
    def __load():
        """
        Load a previously saved model and encoders from scripts/models/

        Returns:
            TraditionalModel: A model instance with loaded state.
        """
        model = TraditionalModel()
        model.model = joblib.load("scripts/models/traditional_model.pkl")
        model.user_encoder = joblib.load("scripts/models/user_encoder.pkl")
        model.item_encoder = joblib.load("scripts/models/item_encoder.pkl")
        print(" TraditionalModel loaded from scripts/models/")
        return model

    @staticmethod
    def __create():
        """
        Create and train a new TraditionalModel instance.

        Returns:
            TraditionalModel: A newly trained model instance.
        """
        model = TraditionalModel()
        model.__train()
        return model

    @staticmethod
    def get_instance():
        """
        Load a saved model if it exists; otherwise, train a new model.

        Returns:
            TraditionalModel: A ready-to-use model instance.
        """
        try:
            return TraditionalModel.__load()
        except Exception as e:
            print(f" Could not load TraditionalModel. Reason: {e}")
            print(" Training new TraditionalModel...")
            return TraditionalModel.__create()