import torch
import pandas as pd
import pickle
from tqdm import tqdm
from build_graph import build_enhanced_graph
from pre_process_dl import load_datasets

def main():
    # This script samples 100,000 user-recipe interactions and filters out inactive users and unrelated recipes to 
    # focus on a cleaner, more relevant subset of the data. Only the ingredients actually used in these recipes are retained to reduce noise. 
    # A compact heterogeneous graph is then built, capturing relationships between users, recipes, ingredients, tags, and review text. 
    # This graph, along with user and recipe ID mappings, is saved for training a GNN-based recommender system. 
    # The purpose of this process is to enable faster, more efficient model development by working with a smaller, 
    # information-rich dataset—ideal for experimentation, prototyping, and limited compute environments.

    # Load Datasets
    interactions, recipes, ingr_map_df = load_datasets(
        'RAW_interactions.csv',
        'RAW_recipes.csv',
        'ingr_map.pkl'
    )

    # ====== Parameters ======
    INTERACTION_SAMPLE_SIZE = 100_000
    MIN_USER_INTERACTIONS = 5

    # ====== Sample & Filter Interactions ======
    print(f"Sampling {INTERACTION_SAMPLE_SIZE} interactions...")
    interactions_subset = interactions.sample(n=INTERACTION_SAMPLE_SIZE, random_state=42)

    # Keep only recipes and users involved in these interactions
    print("Filtering matching recipes and active users...")
    recipes_subset = recipes[recipes['id'].isin(interactions_subset['recipe_id'].unique())]

    user_counts = interactions_subset['user_id'].value_counts()
    active_users = user_counts[user_counts >= MIN_USER_INTERACTIONS].index
    interactions_subset = interactions_subset[interactions_subset['user_id'].isin(active_users)]

    # Re-filter recipes to match cleaned interactions
    recipes_subset = recipes_subset[recipes_subset['id'].isin(interactions_subset['recipe_id'].unique())]

    # ====== Filter Ingredients ======
    print("Filtering used ingredients...")
    used_ingredients = set(
        ing.lower()
        for ingredients in tqdm(recipes_subset['ingredients'], desc="Scanning ingredients")
        for ing in ingredients
    )
    ingr_map_df = ingr_map_df[ingr_map_df['processed'].isin(used_ingredients)]

    # ====== Summary ======
    print(f"\nSubset ready:")
    print(f"• Interactions:  {len(interactions_subset)}")
    print(f"• Recipes:       {len(recipes_subset)}")
    print(f"• Users:         {interactions_subset['user_id'].nunique()}")
    print(f"• Ingredients:   {len(ingr_map_df)}")

    # ====== Build & Save Graph ======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nBuilding graph on {device}...")

    data, user_map, recipe_map = build_enhanced_graph(
        interactions_subset,
        recipes_subset,
        ingr_map_df,
        device=device
    )

    # Save graph
    print("Saving graph to recipe_graph_100k.pt...")
    torch.save(data, "recipe_graph_100k.pt")

    # Save user and recipe ID maps
    print("Saving ID maps...")
    with open("user_map_100k.pkl", "wb") as f:
        pickle.dump(user_map, f)
    with open("recipe_map_100k.pkl", "wb") as f:
        pickle.dump(recipe_map, f)

    print("\nDone! Graph and ID maps saved successfully.")

if __name__ == "__main__":
    main()
