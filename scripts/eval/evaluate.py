import pandas as pd
from ..etl.etl import extract
from ..util.model import Model
from ..etl.etl import get_train_test_splits
from ..naive.model import NaiveModel
from ..traditional.model import TraditionalModel
from ..ncf.model import DeepModel

def evaluate_model(model: Model, model_name: str, X_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
    '''
    Evaluate the model
    '''
    rmse = model.evaluate(X_test, y_test)
    print(f"{model_name} RMSE: {rmse}")

def main():
    '''
    Evaluate the three modeling approaches
    '''
    recipes_df, reviews_df = extract()
    X_train, X_test, y_train, y_test = get_train_test_splits()
    print("Evaluating...")
    models = [
        NaiveModel.get_instance(),
        TraditionalModel.get_instance(),
        DeepModel.get_instance(),
    ]
    for model in models:
        evaluate_model(model, model.__class__.__name__, X_test, y_test)

if __name__ == "__main__":
    main()