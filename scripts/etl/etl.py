import kagglehub
import pandas as pd
import os
def extract() -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Extract the data from the source. First checks if data exists in data/ directory,
    if not downloads from Kaggle and saves to data/ directory.
    '''
    data_dir = 'data'
    recipes_path = f'{data_dir}/recipes.csv'
    reviews_path = f'{data_dir}/reviews.csv'
    reviews_train_path = f'{data_dir}/reviews_train.csv'
    reviews_test_path = f'{data_dir}/reviews_test.csv'

    # Check if files already exist in data directory
    if os.path.exists(recipes_path) and os.path.exists(reviews_path):
        recipes_df = pd.read_csv(recipes_path)
        reviews_df = pd.read_csv(reviews_path)
    else:
        print("Downloading data from Kaggle...")
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Load from Kaggle
        kaggle_path = kagglehub.dataset_download("shuyangli94/food-com-recipes-and-user-interactions")
        recipes_df = pd.read_csv(f'{kaggle_path}/RAW_recipes.csv')
        reviews_df = pd.read_csv(f'{kaggle_path}/RAW_interactions.csv')
        reviews_train_df = pd.read_csv(f'{kaggle_path}/interactions_train.csv')
        reviews_test_df = pd.read_csv(f'{kaggle_path}/interactions_test.csv')
        
        # Save to data directory
        recipes_df.to_csv(recipes_path, index=False)
        reviews_df.to_csv(reviews_path, index=False)
        reviews_train_df.to_csv(reviews_train_path, index=False)
        reviews_test_df.to_csv(reviews_test_path, index=False)

    return recipes_df, reviews_df

def get_train_test_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Split the dataframe into train and test sets

    Parameters:
        reviews_df (pd.DataFrame): dataframe of reviews
        test_size (float): size of the test set

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: train and test sets
    '''
    data_dir = 'data'
    reviews_train_path = f'{data_dir}/reviews_train.csv'
    reviews_test_path = f'{data_dir}/reviews_test.csv'
    reviews_train, reviews_test =pd.read_csv(reviews_train_path), pd.read_csv(reviews_test_path)
    X_train = reviews_train[['user_id', 'recipe_id']]
    y_train = reviews_train['rating']
    X_test = reviews_test[['user_id', 'recipe_id']]
    y_test = reviews_test['rating']
    return X_train, X_test, y_train, y_test
