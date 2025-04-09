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

    def predict(self, X):
        '''
        Run inference on the model
        '''
        raise NotImplementedError("Model is an abstract class")

    def evaluate(self, X, y):
        '''
        Evaluate the model
        '''
        raise NotImplementedError("Model is an abstract class")
        
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
