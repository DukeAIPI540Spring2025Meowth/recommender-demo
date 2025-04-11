import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

def extract() -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Extract the data from the source
    '''
    print(f"Extracting data from Kaggle...")

    # Load the latest version
    path = kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews")

    recipes_df = pd.read_csv(f'{path}/recipes.csv')
    reviews_df = pd.read_csv(f'{path}/reviews.csv')

    print("First 5 records:", recipes_df.head())

    return recipes_df, reviews_df
