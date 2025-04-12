import kagglehub
import pandas as pd
from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split
load_dotenv()

def extract() -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Extract the data from the source. First checks if data exists in data/ directory,
    if not downloads from Kaggle and saves to data/ directory.
    '''
    data_dir = 'data'
    recipes_path = f'{data_dir}/recipes.csv'
    reviews_path = f'{data_dir}/reviews.csv'

    # Check if files already exist in data directory
    if os.path.exists(recipes_path) and os.path.exists(reviews_path):
        recipes_df = pd.read_csv(recipes_path)
        reviews_df = pd.read_csv(reviews_path)
    else:
        print("Downloading data from Kaggle...")
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Load from Kaggle
        kaggle_path = kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews")
        recipes_df = pd.read_csv(f'{kaggle_path}/recipes.csv')
        reviews_df = pd.read_csv(f'{kaggle_path}/reviews.csv')
        
        # Save to data directory
        recipes_df.to_csv(recipes_path, index=False)
        reviews_df.to_csv(reviews_path, index=False)

    return recipes_df, reviews_df

def train_test_split_reviews(reviews_df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Split the dataframe into train and test sets

    Parameters:
        reviews_df (pd.DataFrame): dataframe of reviews
        test_size (float): size of the test set

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: train and test sets
    '''
    X = reviews_df[['AuthorId', 'RecipeId']]
    y = reviews_df['Rating']
    return train_test_split(X, y, test_size=test_size, random_state=42)