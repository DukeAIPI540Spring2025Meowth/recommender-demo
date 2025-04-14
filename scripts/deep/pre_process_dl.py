import pandas as pd
import ast  

def load_datasets(inter_path, recipe_path, ingr_map_path):
    """
        Loads and cleans the interaction, recipe, and ingredient mapping datasets.

        Parameters:
        -----------
        inter_path : str
            Path to the CSV file with user-recipe interactions (ratings, reviews, etc.).
        recipe_path : str
            Path to the CSV file containing recipe details like ingredients, tags, and nutrition.
        ingr_map_path : str
            Path to the pickle file with processed ingredient mappings.

        Returns:
        --------
        interactions : pd.DataFrame
            User interactions with recipes — includes things like ratings, reviews, and timestamps.
        
        recipes : pd.DataFrame
            Recipe metadata with proper Python lists for fields like tags, nutrition info, ingredients, and steps.

        ingr_map_df : pd.DataFrame
            A mapping of cleaned ingredient names to unique IDs for building the graph.

        Notes:
        ------
        Some columns in the recipe dataset (like 'tags' and 'ingredients') are stored as strings 
        that look like lists. This function converts them back to actual Python lists so they’re 
        easier to work with later on.
    """

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
