# This script is designed to build a heterogeneous graph for recipe recommendation using data from user-recipe interactions, 
# recipe content, and ingredient mappings. The primary goal is to represent various types of information—such as ratings, text, 
# ingredients, and metadata—as a structured graph that later is be used to train a graph neural network (GNN) for personalized recommendations.


import pandas as pd
import numpy as np
import torch
import re
from collections import defaultdict, Counter
from torch_geometric.data import HeteroData
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from pre_process_dl import load_datasets

nltk.download('vader_lexicon')

# Load Datasets
interactions, recipes, ingr_map = load_datasets(
    'RAW_interactions.csv',
    'RAW_recipes.csv',
    'ingr_map.pkl'
)

# The process begins by enhancing the interaction data with sentiment scores. 

# Each review left by a user is analyzed using the VADER sentiment analysis tool, and the resulting sentiment score (ranging from negative to positive) 
# is added as a new feature to the dataset. 


def enhance_interactions_with_sentiment(interactions):
    # Adds a sentiment score and converts dates to timestamps
    sia = SentimentIntensityAnalyzer()
    sentiments, timestamps = [], []

    for _, row in interactions.iterrows():
        review = str(row['review'])
        sentiment = sia.polarity_scores(review)['compound']  # value between -1 (negative) and 1 (positive)
        sentiments.append(sentiment)

        # Additionally, the original review timestamps are converted to numerical UNIX timestamps to allow for temporal modeling. 
        # These steps help enrich the user-recipe interactions with more contextual signals.

        # Convert review date to a numeric timestamp (or 0 if invalid)
        date = pd.to_datetime(row['date'], errors='coerce')
        timestamp = int(date.timestamp()) if pd.notnull(date) else 0
        timestamps.append(timestamp)

    interactions['sentiment'] = sentiments
    interactions['timestamp'] = timestamps
    return interactions

# Next, the script performs basic text processing to clean and vectorize the user reviews using the TF-IDF method. 
# This allows us to identify meaningful words that users frequently use in their reviews. The TF-IDF matrix and word vocabulary are then used to connect 
# users to word nodes in the graph, with the edge weights reflecting the importance of each word to each user.


def compute_tfidf(interactions):
    # Clean up text and generate a TF-IDF matrix for user reviews
    interactions['cleaned_review'] = interactions['review'].fillna('').apply(
        lambda r: ' '.join(re.findall(r'\b[a-z]{3,}\b', r.lower()))
    )
    tfidf = TfidfVectorizer(min_df=5)  # Ignore very rare words
    tfidf_matrix = tfidf.fit_transform(interactions['cleaned_review'])
    word_map = {word: i for i, word in enumerate(tfidf.get_feature_names_out())}
    return tfidf_matrix, word_map, tfidf


# The graph is built using HeteroData, with nodes for users, recipes, ingredients, tags, and words. 
# Edges capture interactions and content links, enriched with features like ratings, sentiment, timestamps, and TF-IDF scores.


def build_enhanced_graph(interactions, recipes, ingr_map_df, device="cpu"):
    data = HeteroData()

    # Preprocess interactions: add sentiment and timestamp info
    interactions = enhance_interactions_with_sentiment(interactions)

    # Convert review text into TF-IDF matrix and word index
    tfidf_matrix, tfidf_word_map, tfidf_model = compute_tfidf(interactions)

    # Create mapping from user/recipe/ingredient/tag to node index
    user_ids = interactions['user_id'].unique()
    recipe_ids = recipes['id'].unique()
    ingr_map = {row['processed']: row['id'] for _, row in ingr_map_df.iterrows()}

    # Internal ID maps are used for users, recipes, tags, and ingredients to track edges with indexed tensors. 
    # Word usage is captured via TF-IDF-weighted edges and raw counts to model language patterns in reviews.


    user_map = {uid: i for i, uid in enumerate(user_ids)}
    recipe_map = {rid: i for i, rid in enumerate(recipe_ids)}

    ingr_index_map = {}
    ingr_counter = 0
    tag_map = {}
    tag_counter = 0
    word_map = tfidf_word_map  # Reuse TF-IDF vocab for word indexing

    # === USER -> RECIPE edge construction ===
    user_idx, recipe_idx = [], []
    rating_attr, sentiment_attr, time_attr = [], [], []

    for _, row in interactions.iterrows():
        if row['user_id'] in user_map and row['recipe_id'] in recipe_map:
            user_idx.append(user_map[row['user_id']])
            recipe_idx.append(recipe_map[row['recipe_id']])
            rating_attr.append(row['rating'])
            sentiment_attr.append(row['sentiment'])
            time_attr.append(row['timestamp'])

    data['user'].num_nodes = len(user_map)
    data['recipe'].num_nodes = len(recipe_map)

    data['user', 'rates', 'recipe'].edge_index = torch.tensor(
        [user_idx, recipe_idx], dtype=torch.long, device=device
    )
    data['user', 'rates', 'recipe'].edge_attr = torch.tensor(
        np.stack([rating_attr, sentiment_attr, time_attr], axis=1), dtype=torch.float, device=device
    )

    # === USER -> WORD (TF-IDF) ===
    # Connect users to words they use, weighted by TF-IDF
    tfidf_csr = tfidf_matrix.tocoo()
    user_word_src, user_word_dst, tfidf_weights = [], [], []

    for u_idx, w_idx, val in zip(tfidf_csr.row, tfidf_csr.col, tfidf_csr.data):
        uid = interactions.iloc[u_idx]['user_id']
        if uid not in user_map:
            continue
        user_word_src.append(user_map[uid])
        user_word_dst.append(w_idx)
        tfidf_weights.append(val)

    data['word'].num_nodes = len(word_map)
    data['user', 'uses_word', 'word'].edge_index = torch.tensor(
        [user_word_src, user_word_dst], dtype=torch.long, device=device
    )
    data['user', 'uses_word', 'word'].edge_attr = torch.tensor(
        tfidf_weights, dtype=torch.float, device=device
    )

    # === RECIPE -> INGREDIENT ===
    # Link each recipe to its matched ingredients (using processed names)
    ingr_src, ingr_dst = [], []

    for _, row in recipes.iterrows():
        r_id = row['id']
        if r_id not in recipe_map:
            continue
        r_idx = recipe_map[r_id]
        for raw_ingr in row['ingredients']:
            cleaned = raw_ingr.lower().strip()
            for match in ingr_map:
                if match in cleaned:
                    ingr_id = ingr_map[match]
                    if ingr_id not in ingr_index_map:
                        ingr_index_map[ingr_id] = ingr_counter
                        ingr_counter += 1
                    i_idx = ingr_index_map[ingr_id]
                    ingr_src.append(r_idx)
                    ingr_dst.append(i_idx)
                    break

    data['ingredient'].num_nodes = len(ingr_index_map)
    data['recipe', 'has_ingredient', 'ingredient'].edge_index = torch.tensor(
        [ingr_src, ingr_dst], dtype=torch.long, device=device
    )

    # === RECIPE -> TAG ===
    # Tag each recipe with its associated labels
    tag_src, tag_dst = [], []

    for _, row in recipes.iterrows():
        r_id = row['id']
        if r_id not in recipe_map:
            continue
        r_idx = recipe_map[r_id]
        for tag in row['tags']:
            if tag not in tag_map:
                tag_map[tag] = tag_counter
                tag_counter += 1
            t_idx = tag_map[tag]
            tag_src.append(r_idx)
            tag_dst.append(t_idx)

    data['tag'].num_nodes = len(tag_map)
    data['recipe', 'has_tag', 'tag'].edge_index = torch.tensor(
        [tag_src, tag_dst], dtype=torch.long, device=device
    )

    # === USER / RECIPE -> WORD (raw token usage) ===
    # Link users and recipes to words they mention in reviews (not weighted)
    word_src_u, word_dst_u, word_src_r, word_dst_r = [], [], [], []
    word_freq_u = defaultdict(Counter)
    word_freq_r = defaultdict(Counter)
    tokenizer = lambda txt: re.findall(r'\b[a-z]{3,}\b', str(txt).lower())

    for _, row in interactions.iterrows():
        uid, rid = row['user_id'], row['recipe_id']
        if uid not in user_map or rid not in recipe_map:
            continue
        u_idx, r_idx = user_map[uid], recipe_map[rid]
        words = tokenizer(row['review'])
        for word in words:
            if word in word_map:
                w_idx = word_map[word]
                word_freq_u[u_idx][w_idx] += 1
                word_freq_r[r_idx][w_idx] += 1

    for u_idx in word_freq_u:
        for w_idx in word_freq_u[u_idx]:
            word_src_u.append(u_idx)
            word_dst_u.append(w_idx)

    for r_idx in word_freq_r:
        for w_idx in word_freq_r[r_idx]:
            word_src_r.append(r_idx)
            word_dst_r.append(w_idx)

    data['user', 'raw_uses', 'word'].edge_index = torch.tensor(
        [word_src_u, word_dst_u], dtype=torch.long, device=device
    )
    data['recipe', 'mentioned_with', 'word'].edge_index = torch.tensor(
        [word_src_r, word_dst_r], dtype=torch.long, device=device
    )

    # === RECIPE node features ===
    # Recipe node features combine nutrition, cook time, and ingredient count into a vector used by the model. 
    # The final HeteroData graph is ready for training a GNN for content- and behavior-aware recommendations.

    feats = []
    for _, row in recipes.iterrows():
        try:
            feats.append(row['nutrition'] + [row['minutes'], row['n_ingredients']])
        except:
            feats.append([0.0] * 9)

    data['recipe'].x = torch.tensor(feats, dtype=torch.float, device=device)

    return data, user_map, recipe_map
