import pandas as pd
import numpy as np
from ..util.model import Model
from ..etl.etl import extract

class NaiveModel(Model):
    '''
    A simple model that predicts the rating of a recipe based on the average rating of all recipes
    '''
    def __init__(self):
        '''
        Initialize the model
        '''
        self.recipes_df, self.reviews_df = extract()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        '''
        Run inference on the model

        Parameters:
            X (pd.DataFrame): dataframe of user and recipe ids for which to predict ratings

        Returns:
            np.ndarray: The predicted ratings
        '''
        return pd.merge(X, self.recipes_df[['RecipeId', 'AggregatedRating']], on='RecipeId', how='left')['AggregatedRating'].to_numpy()
    
    def __save(self):
        '''
        Save the model
        '''
        pass

    def __train(self):
        '''
        Train the model
        '''
        pass

    @staticmethod
    def get_instance():
        '''
        Get the instance of the model
        '''
        return NaiveModel.__create()

    @staticmethod
    def __load():
        '''
        Load the model
        '''
        raise NotImplementedError("Naive model does not support loading")

    @staticmethod
    def __create():
        '''
        Create the model
        '''
        return NaiveModel()
