# recommender-demo
Recommender System demo for food.com recipes

## Setup Instructions

1. Create a Python virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- On macOS/Linux:
```bash
source venv/bin/activate
```
- On Windows:
```bash
.\venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Run the setup script to load and transform data:
```bash
python setup.py
```

5. To deactivate the virtual environment when you're done:
```bash
deactivate
```

## Running the Evaluation

To evaluate the recommender system, run:
```bash
python -m scripts.eval.evaluate
```

This will run the evaluation module and output the performance metrics of the recommender system.

## Problem

Recommender systems are essential for helping users navigate large content libraries. On platforms like [food.com](https://www.food.com/), with tens of thousands of recipes, it can be difficult for users to find meals that match their preferences, dietary needs, and skill levels.

This project tackles the problem of personalized recipe recommendation - predicting how much a user will like a given recipe based on their past reviews.

**Challenges include:**
- Sparse user-recipe interaction data
- Diverse recipe features (ingredients, prep time, nutrition)
- Capturing non-linear, personalized preferences

**Why build this?**
Existing approaches like popularity-based methods or ingredient filtering often fail to capture the complexity of human taste. Our goal is to build a flexible, accurate, and extensible recommendation engine that adapts to user preferences and scales well.


## Modeling Approaches

This project implements four different approaches to recipe recommendation:

1. **Naive Approach**: A simple baseline that recommends the most popular recipes based on average ratings and number of reviews.

2. **Deep Learning Approach**: Uses a graph neural network to learn user and recipe embeddings, trained on a 100K sample of the dataset. This approach leverages modern deep learning techniques to capture complex patterns in user-recipe interactions.

3. **Neural Collaborative Filtering (NCF)**: A deep learning approach that combines matrix factorization with neural networks. The model:
   - Uses user and item embeddings to capture latent features
   - Implements a multi-layer perceptron (MLP) to learn complex interactions
   - Adds user and item bias terms to model global rating tendencies
   - Supports GPU/MPS acceleration for faster training
   - Currently achieves an RMSE of 0.55 after 10 epochs of training
   - Trains on the full training set, unlike the previous deep learning approach
   - Supports Optuna-based hyperparameter tuning for embedding size, MLP depth, and learning rate

4. **Traditional Approach**: Implements a collaborative filtering system using K-Nearest Neighbors (KNN) regression with feature engineering. The model:
   - Uses average user and recipe ratings as features
   - Implements KNN regression with optimized hyperparameters
   - Scales and encodes features for better performance
   - Achieves an RMSE of 0.58
   - Can handle any user ID or recipe ID in the dataset
  
The Traditional collaborative filtering model is more economical and sustainable to run as a production application. Unlike deep learning models, it requires significantly less computational power, making it faster to train and more efficient at inference time. This allows it to handle real-time user requests without the need for GPUs or high-memory instances, reducing infrastructure costs and energy consumption—making it an ideal choice for scalable, long-term deployment.

## Evaluation

The system is evaluated using Root Mean Square Error (RMSE) as the primary metric. RMSE is chosen because:
- It penalizes larger errors more heavily than smaller ones
- It's in the same units as the original ratings (1-5 scale)
- It's widely used in recommender systems evaluation
- It provides an intuitive measure of prediction accuracy

### Evaluation Results
**Naive Approach:** 0.70 RMSE

**Deep Learning Approach:** 0.53 RMSE

**Neural Collaborative Filtering:** 0.55 RMSE

**Traditional Approach:** 0.58 RMSE

Note: The original GNN-based deep learning approach currently has some data leakage in its implementation that would need to be addressed in future revisions to ensure proper evaluation.

### Approach Selection
The traditional collaborative filtering approach was selected for the production application because:
- It can gracefully handle any user ID or recipe ID in the dataset
- It provides consistent performance across the entire dataset
- It's more computationally efficient than the deep learning approach
- It doesn't suffer from the data leakage issues present in the current deep learning implementation

## Data Source

The project uses the Food.com Recipes and User Interactions dataset from Kaggle:
https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

The dataset includes:
- Raw recipe reviews and user interactions
- Pre-defined train/test splits
- Recipe metadata including ingredients, instructions, and nutritional information
- User interaction data including ratings and reviews

## Previous Work

This project builds upon several previous approaches to recipe recommendation:

1. [Recipe Recommendation System](https://www.kaggle.com/code/engyyyy/recipe-recommendation-system)
2. [Item-Based Recommender](https://www.kaggle.com/code/ipekgamzeucal/item-based-recommender)
3. [Recipe Recommendation Based on Ingredients](https://www.kaggle.com/code/ipekgamzeucal/recipe-recommendation-based-on-ingredients)
4. [Food Recommendation System](https://www.kaggle.com/code/capgos/food-recommendation-system-assignment-1)

## Streamlit Application

A user-friendly web interface for the recommender system is available at:
https://recommender-demo-518487429487.us-central1.run.app/

The Streamlit application allows users to:
- Search for recipes
- Get personalized recommendations
- View recipe details and ratings
- Interact with the recommender system in real-time

Note that with the current hardware configuration in GCP (2GB memory, 1 CPU), the app takes up to a minute to load.

## Ethics Statement

This food.com recipe recommender system is designed with several ethical considerations in mind:

1. **Health and Nutrition**: While the system recommends recipes based on user preferences, it does not make health claims or provide medical advice. Users should consult healthcare professionals for dietary guidance.

2. **Data Privacy**: User data is handled with strict privacy considerations. Personal information is anonymized and used only for improving recommendation quality.

3. **Transparency**: The system is transparent about its recommendation process and the factors that influence its suggestions.
