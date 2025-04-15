import streamlit as st
import pandas as pd
from streamlit_star_rating import st_star_rating
from scripts.traditional.model import TraditionalModel

# Initialize session state for dataframes if not already done
if 'recipes_df' not in st.session_state:
    # Load the recipes and reviews data
    recipes_df = pd.read_csv('data/recipes.csv')
    reviews_df = pd.read_csv('data/reviews.csv')
    
    # Sample 20 recipes randomly
    recipes_df = recipes_df.sample(n=50, random_state=42)
    
    # Filter reviews to only include reviews for the sampled recipes
    reviews_df = reviews_df[reviews_df['recipe_id'].isin(recipes_df['id'])]
    
    # Initialize the NaiveModel
    model = TraditionalModel.get_instance()
    
    # Calculate mean ratings
    mean_ratings = reviews_df.groupby('recipe_id')['rating'].mean().reset_index() 
    mean_ratings.rename(columns={'rating': 'mean_rating'}, inplace=True)
    
    # Merge mean ratings into recipes_df
    recipes_df = recipes_df.merge(mean_ratings, left_on='id', right_on='recipe_id', how='left')  
    
    # Initialize user_id
    user_id = 2312
    
    # Create input data for prediction
    input_data = pd.DataFrame({
        'recipe_id': recipes_df['id'],
        'user_id': user_id
    })
    
    # Get predictions from model
    predicted = model.predict(input_data)
    
    # Add predicted ratings to recipes_df
    recipes_df['predicted_rating'] = predicted
    
    # Initialize user_ratings_df
    user_ratings_df = pd.DataFrame(columns=['recipe_id', 'user_id', 'rating'])
    
    # Add final_rating column
    recipes_df['final_rating'] = recipes_df['predicted_rating']
    
    # Sort recipes by predicted rating initially
    recipes_df = recipes_df.sort_values(by='final_rating', ascending=False)
    
    # Store in session state
    st.session_state.recipes_df = recipes_df
    st.session_state.reviews_df = reviews_df
    st.session_state.user_ratings_df = user_ratings_df
    st.session_state.user_id = user_id

# Initialize the number of entries to display in session state
if 'num_entries' not in st.session_state:
    st.session_state.num_entries = 10  # Default to 10 entries

# Function to load more entries
def load_more():
    st.session_state.num_entries += 10  # Increase the number of entries by 10

# Function to update user ratings and sort recipes
def update_rating(recipe_id, rating):
    user_id = st.session_state.user_id
    
    # Update user_ratings_df
    if len(st.session_state.user_ratings_df[(st.session_state.user_ratings_df['recipe_id'] == recipe_id) & 
                                            (st.session_state.user_ratings_df['user_id'] == user_id)]) > 0:
        # Update existing rating
        st.session_state.user_ratings_df.loc[
            (st.session_state.user_ratings_df['recipe_id'] == recipe_id) & 
            (st.session_state.user_ratings_df['user_id'] == user_id), 'rating'] = rating
    else:
        # Add new rating
        new_rating = pd.DataFrame({'recipe_id': [recipe_id], 'user_id': [user_id], 'rating': [rating]})
        st.session_state.user_ratings_df = pd.concat([st.session_state.user_ratings_df, new_rating], 
                                                     ignore_index=True)
    
    # Add to reviews_df as well
    new_review = pd.DataFrame({'id': [recipe_id], 'user_id': [user_id], 'rating': [rating]})
    st.session_state.reviews_df = pd.concat([st.session_state.reviews_df, new_review], ignore_index=True)
    
    # Update final_rating column
    for idx, row in st.session_state.recipes_df.iterrows():
        recipe_id_str = str(row['id'])
        # Check if user has rated this recipe
        user_rating = st.session_state.user_ratings_df[
            (st.session_state.user_ratings_df['recipe_id'] == recipe_id_str) & 
            (st.session_state.user_ratings_df['user_id'] == user_id)
        ]
        
        if len(user_rating) > 0:
            # Use user rating if available
            st.session_state.recipes_df.at[idx, 'final_rating'] = user_rating.iloc[0]['rating']
        else:
            # Use predicted rating otherwise
            st.session_state.recipes_df.at[idx, 'final_rating'] = row['predicted_rating']
    
    # Sort by final_rating
    st.session_state.recipes_df = st.session_state.recipes_df.sort_values(by='final_rating', ascending=False)
    st.rerun()

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

# Add transparency statement
st.markdown("""
<div style='background-color: #2b2b2b; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
    <p style='color: #bbbbbb; margin: 0;'>
        Our recommendations are powered by collaborative filtering, which means we suggest recipes based on patterns in how people like you have rated similar recipes. 
        The more you rate recipes, the better our recommendations become at understanding your preferences!
    </p>
    <p style='color: #bbbbbb; margin: 0; margin-top: 0.5rem; font-size: 0.9em;'>
        Note: While we recommend recipes based on preferences, we don't make health claims or provide medical advice. Please consult healthcare professionals for dietary guidance.
    </p>
</div>
""", unsafe_allow_html=True)

# Display recipes
st.subheader("🍽️ Recommended Recipes")

# Get the recipes dataframe from session state
recipes_df = st.session_state.recipes_df

for index, row in recipes_df.head(st.session_state.num_entries).iterrows():
    recipe_id = str(row['id'])
    recipe_name = row['name']
    
    # Format ingredients nicely
    ingredients_list = row['ingredients']
    if isinstance(ingredients_list, str):
        # If ingredients are stored as a string representation of a list
        try:
            ingredients_list = eval(ingredients_list)
        except:
            ingredients_list = [ingredients_list]
    
    # Create a nicely formatted ingredients string
    if isinstance(ingredients_list, list):
        formatted_ingredients = "<ul style='margin-top: 0; padding-left: 20px;'>"
        for ingredient in ingredients_list:
            formatted_ingredients += f"<li>{ingredient}</li>"
        formatted_ingredients += "</ul>"
    else:
        formatted_ingredients = str(ingredients_list)
    
    time_to_cook = row['minutes']
    average_rating = row['mean_rating']
    
    # Get user rating if it exists
    user_rating_row = st.session_state.user_ratings_df[
        (st.session_state.user_ratings_df['recipe_id'] == recipe_id) & 
        (st.session_state.user_ratings_df['user_id'] == st.session_state.user_id)
    ]
    
    user_rating_default = user_rating_row['rating'].iloc[0] if len(user_rating_row) > 0 else 0

    with st.container():
        st.markdown(f'<div class="recipe-title">{recipe_name}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ingredients"><strong>Ingredients:</strong></div>', unsafe_allow_html=True)
        st.markdown(f'{formatted_ingredients}', unsafe_allow_html=True)
        st.markdown(f'<div class="time">⏱️ {time_to_cook} min</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st_star_rating(label="⭐ Average Rating", maxValue=5, defaultValue=average_rating, key=f"avg_{recipe_id}", read_only=True)
        with col2:
            user_stars = st_star_rating(
                label="🤔 Your Experience", 
                maxValue=5, 
                defaultValue=user_rating_default,
                key=f"user_{recipe_id}"
            )
            # Check if rating has changed and update it
            if user_stars != user_rating_default and user_stars > 0:
                update_rating(recipe_id, user_stars)
    

        if st.button(f"View Details for {recipe_name}", key=f"details_{index}"):
            # Format steps nicely
            steps_list = row['steps']
            if isinstance(steps_list, str):
                # If steps are stored as a string representation of a list
                try:
                    steps_list = eval(steps_list)
                except:
                    steps_list = [steps_list]
            
            # Create a nicely formatted steps string
            if isinstance(steps_list, list):
                formatted_steps = "<ol style='margin-top: 10px; padding-left: 20px;'>"
                for step in steps_list:
                    formatted_steps += f"<li>{step}</li>"
                formatted_steps += "</ol>"
            else:
                formatted_steps = str(steps_list)
                
            st.markdown(f"<h4>Instructions for {recipe_name}:</h4>", unsafe_allow_html=True)
            st.markdown(formatted_steps, unsafe_allow_html=True)

if st.session_state.num_entries < len(recipes_df):
    # Load more button
    if st.button("Load more"):
        load_more()  # Call the function to load more entries
