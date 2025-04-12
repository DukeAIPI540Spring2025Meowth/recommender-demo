import numpy as np
import pandas as pd

class Model:
    '''
    Abstract class for all models
    '''
    def __init__(self):
        '''
        Initialize the model
        '''
        self.__train()
        self.__save()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        '''
        Run inference on the model

        Parameters:
            X (pd.DataFrame): dataframe of user and recipe ids for which to predict ratings

        Returns:
            np.ndarray: The predicted ratings
        '''
        raise NotImplementedError("Model is an abstract class")

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> float:
        '''
        Evaluate the model

        Parameters:
            X (pd.DataFrame): dataframe of user and recipe ids for which to predict ratings
            y (pd.DataFrame): dataframe of ground truth ratings

        Returns:
            float: The RMSE of the model
        '''
        return np.sqrt(np.mean((self.predict(X) - y) ** 2))
        
    def __save(self):
        '''
        Save the model
        '''
        raise NotImplementedError("Model is an abstract class")
    
    def __train(self):
        '''
        Train the model
        '''
        raise NotImplementedError("Model is an abstract class")

    @staticmethod
    def get_instance():
        '''
        Get the instance of the model
        '''
        raise NotImplementedError("Model is an abstract class")

    @staticmethod
    def __load():
        '''
        Load the model
        '''
        raise NotImplementedError("Model is an abstract class")

    @staticmethod
    def __create():
        '''
        Create the model
        '''
        raise NotImplementedError("Model is an abstract class")
