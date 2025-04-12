
import streamlit as st
import pandas as pd
from streamlit_star_rating import st_star_rating
from scripts.naive.model import NaiveModel

# # Sample data for recipes
# recipes_data = {
#     'recipe': ['Vegan Salad', 'Gluten-free Pasta', 'Vegetarian Stir-fry'],
#     'ingredients': ['Lettuce, Tomato, Cucumber', 'Rice Flour, Eggs', 'Broccoli, Carrots, Tofu'],
#     'time_to_cook': ['10 mins', '20 mins', '15 mins'],
#     'average_rating': [4.5, 4.0, 4.2],
# }

# # Create a DataFrame
# recipes_df = pd.DataFrame(recipes_data)

# Load the recipes and reviews data
recipes_df = pd.read_csv('data/recipes.csv')  # Update the path as necessary
reviews_df = pd.read_csv('data/reviews.csv')  # Update the path as necessary

# Initialize the NaiveModel
model = NaiveModel()

mean_ratings = reviews_df.groupby('recipe_id')['rating'].mean().reset_index()  # Assuming 'Rating' is the column for ratings
mean_ratings.rename(columns={'rating': 'mean_rating'}, inplace=True)

recipes_df = recipes_df.merge(mean_ratings, left_on='id', right_on='recipe_id', how='left')  # Merge mean ratings into recipes_df

user_id = 'test_user'
input_data = pd.DataFrame({
    'recipe_id': recipes_df['id'],  # Use RecipeId from recipes_df
    'user_id': user_id  # Set user_id to 'test_user' for all rows
})

predicted = model.predict(input_data)

recipes_df['predicted_rating'] = predicted
recipes_df = recipes_df.sort_values(by='predicted_rating', ascending=False)  # Sort in descending order

# Function to filter recipes based on dietary restrictions
# def filter_recipes(dietary_restriction):

#     return recipes_df  # Placeholder

# Initialize the number of entries to display in session state
if 'num_entries' not in st.session_state:
    st.session_state.num_entries = 10  # Default to 10 entries

# Function to load more entries
def load_more():
    st.session_state.num_entries += 10  # Increase the number of entries by 10

# ------------------------ Custom CSS ------------------------
st.markdown("""
    <style>
    .recipe-card {
        background-color: #1e1e1e;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    .recipe-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .ingredients {
        color: #bbbbbb;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }
    .time {
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
        color: #999;
    }
    .stButton>button {
        background-color: #313131;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        margin-top: 0.75rem;
    }
    .comment-box textarea {
        border-radius: 10px !important;
        border: 1px solid #555 !important;
        background-color: #2b2b2b !important;
        color: white !important;
    }
    /* Smaller star size */
    .st-star-rating svg {
        width: 18px !important;
        height: 18px !important;
    }
    .st-star-rating label {
        font-size: 0.85rem;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------ App UI ------------------------
st.title("🥗 RecipeMe – Smart Recipe Recommender")

# # User input for dietary restrictions
# dietary_restriction = st.selectbox(
#     "Filter by dietary preference:",
#     ["Vegan", "Vegetarian", "Gluten-free", "Diabetic-friendly", "Dairy-free"]
# )

# Filter recipes based on user input
# filtered_recipes = filter_recipes(dietary_restriction)

# Display recipes
st.subheader("🍽️ Recommended Recipes")



for index, row in recipes_df.head(st.session_state.num_entries).iterrows():
    recipe_id = str(row['id'])
    recipe_name = row['name']
    ingredients = row['ingredients']
    time_to_cook = row['minutes']
    average_rating = row['mean_rating']
    user_rating = st.session_state.get(f"userRating_{recipe_id}", 0)  # Get user rating from session state

    # Update reviews_df with the new user rating
    if user_rating > 0:  # Only save if the user has rated
        new_review = pd.DataFrame({
            'id': [recipe_id],
            'user_id': [user_id],
            'rating': [user_rating]
        })
        reviews_df = pd.concat([reviews_df, new_review], ignore_index=True)

        # Recalculate mean ratings after user input
        mean_ratings = reviews_df.groupby('id')['rating'].mean().reset_index()  # Group by 'id' to get mean ratings
        mean_ratings.rename(columns={'rating': 'mean_rating'}, inplace=True)

        # Merge updated mean ratings into recipes_df
        recipes_df = recipes_df.merge(mean_ratings, left_on='id', right_on='id', how='left')  # Merge mean ratings into recipes_df

        # Sort recipes_df by the new mean_rating
        recipes_df = recipes_df.sort_values(by='mean_rating', ascending=False)  # Sort in descending order

    with st.container():
        #st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
        
        st.markdown(f'<div class="recipe-title">{recipe_name}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ingredients"><strong>Ingredients:</strong> {ingredients}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="time">⏱️ {time_to_cook} min</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st_star_rating(label="⭐ Average Rating", maxValue=5, defaultValue=average_rating, key=recipe_id, read_only=True)
        with col2:
            st_star_rating(label="🤔 Your Experience", maxValue=5, defaultValue=0, key=recipe_id + '_userRating')
        
        # Comment box
        comment = st.text_area("💬 Leave a comment (optional):", placeholder="What did you like or dislike?", key=f"comment_{index}", label_visibility="collapsed")
        st.markdown('<div class="comment-box"></div>', unsafe_allow_html=True)

        if st.button(f"View Details for {recipe_name}", key=f"details_{index}"):
            st.info(f"More details for *{recipe_name}* will appear here soon!")

        st.markdown('</div>', unsafe_allow_html=True)

# Load more button
if st.button("Load more"):
    load_more()  # Call the function to load more entries