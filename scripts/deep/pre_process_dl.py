import pandas as pd
import ast  

def load_datasets(inter_path, recipe_path, ingr_map_path):
    # read in all three datasets
    interactions = pd.read_csv(inter_path)
    recipes = pd.read_csv(recipe_path)
    ingr_map_df = pd.read_pickle(ingr_map_path)

    # fix columns that are stored as strings instead of actual lists
    recipes['tags'] = recipes['tags'].apply(ast.literal_eval)
    recipes['nutrition'] = recipes['nutrition'].apply(ast.literal_eval)
    recipes['ingredients'] = recipes['ingredients'].apply(ast.literal_eval)
    recipes['steps'] = recipes['steps'].apply(ast.literal_eval)

    return interactions, recipes, ingr_map_df

# load everything
interactions, recipes, ingr_map = load_datasets(
    'RAW_interactions.csv',
    'RAW_recipes.csv',
    'ingr_map.pkl'
)

