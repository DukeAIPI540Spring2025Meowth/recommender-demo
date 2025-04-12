
import streamlit as st
import pandas as pd
from streamlit_star_rating import st_star_rating

# Sample data for recipes
recipes_data = {
    'recipe': ['Vegan Salad', 'Gluten-free Pasta', 'Vegetarian Stir-fry'],
    'ingredients': ['Lettuce, Tomato, Cucumber', 'Rice Flour, Eggs', 'Broccoli, Carrots, Tofu'],
    'time_to_cook': ['10 mins', '20 mins', '15 mins'],
    'average_rating': [4.5, 4.0, 4.2],
}

# Create a DataFrame
recipes_df = pd.DataFrame(recipes_data)

# Function to filter recipes based on dietary restrictions
def filter_recipes(dietary_restriction):
    # Here you would implement the actual filtering logic based on dietary restrictions
    return recipes_df  # For now, return all recipes

# Streamlit app layout
st.title("RecipeMe - Recipe Recommendation")

# User input for dietary restrictions
dietary_restriction = st.selectbox("Select your dietary restriction:", 
                                    ["Vegan", "Vegetarian", "Gluten-free", "Diabetic-friendly", "Dairy-free"])

# Filter recipes based on user input
filtered_recipes = filter_recipes(dietary_restriction)

# Display recipes in a table
st.subheader("Recommended Recipes")
for index, row in filtered_recipes.iterrows():
    # recipe_name = row['recipe']
    # st.write(f"**{recipe_name}**")
    # st.write(f"Ingredients: {row['ingredients']}")
    # st.write(f"Time to Cook: {row['time_to_cook']}")
    # st.write(f"Average Rating: {row['average_rating']}")
    
    # # User rating input
    # user_rating = st.slider(f"Rate {recipe_name}:", 1, 5, 3)
    # st.write(f"Your Rating: {user_rating}")
    recipe_name = row['recipe']
    ingredients = row['ingredients']
    time_to_cook = row['time_to_cook']
    average_rating = row['average_rating']
    
    col1, col2, col3, col4, col5 = st.columns([2, 3, 2, 6, 6])
    
    with col1:
        st.write(f"**{recipe_name}**")
    with col2:
        st.write(ingredients)
    with col3:
        st.write(time_to_cook)
    with col4:
       
        st_star_rating(label = "Average Rating", maxValue = 5, defaultValue = average_rating, key = recipe_name, read_only = True )
    with col5:
        st_star_rating(label = "Please rate you experience", maxValue = 5, defaultValue = 0, key = recipe_name + '_userRating')
        #user_rating = st_star_rating(0, label="Your Rating")  # User rating input

    # Link to more details (placeholder)
    if st.button(f"View Details for {recipe_name}"):
        st.write(f"Details for {recipe_name} will be displayed here.")
