from ..util.model import Model

class NaiveModel(Model):
    def __init__(self):
        '''
        Initialize the model
        '''
        super().__init__()

    def predict(self, X):
        '''
        Run inference on the model
        '''
        pass    
    
    def evaluate(self, X, y):
        '''
        Evaluate the model
        '''
        pass
    
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
        try:
            print(f"{NaiveModel.__name__} found, loading instance")
            return NaiveModel.__load()
        except:
            print(f"{NaiveModel.__name__} not found, creating new instance")
            return NaiveModel.__create()

    @staticmethod
    def __load():
        '''
        Load the model
        '''
        pass

    @staticmethod
    def __create():
        '''
        Create the model
        '''
        return NaiveModel()
