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

## Modeling Approaches

This project implements four different approaches to recipe recommendation:

1. **Naive Approach**: A simple baseline that recommends the most popular recipes based on average ratings and number of reviews.

2. **Deep Learning Approach**: Uses a graph neural network to learn user and recipe embeddings, trained on a 100K sample of the dataset. This approach leverages modern deep learning techniques to capture complex patterns in user-recipe interactions.

3. **Neural Collaborative Filtering (NCF)**: A deep learning approach that combines matrix factorization with neural networks. The model:
   - Uses user and item embeddings to capture latent features
   - Implements a multi-layer perceptron (MLP) to learn complex interactions
   - Combines dot product and MLP outputs for final predictions
   - Supports GPU/MPS acceleration for faster training
   - Currently achieves an RMSE of 6.67 after 100 epochs of training
   - Trains on the full training set, unlike the previous deep learning approach.

4. **Traditional Approach**: Implements a collaborative filtering system using K-Nearest Neighbors (KNN) regression with feature engineering. The model:
   - Uses average user and recipe ratings as features
   - Implements KNN regression with optimized hyperparameters
   - Scales and encodes features for better performance
   - Achieves an RMSE of 0.58
   - Can handle any user ID or recipe ID in the dataset

## Evaluation

The system is evaluated using Root Mean Square Error (RMSE) as the primary metric. RMSE is chosen because:
- It penalizes larger errors more heavily than smaller ones
- It's in the same units as the original ratings (1-5 scale)
- It's widely used in recommender systems evaluation
- It provides an intuitive measure of prediction accuracy

### Evaluation Results
**Naive Approach:** 0.70 RMSE

**Deep Learning Approach:** 0.53 RMSE

**Neural Collaborative Filtering:** 6.67 RMSE (needs improvement)

**Traditional Approach:** 0.58 RMSE

Note: The neural collaborative filtering approach currently has higher error rates than expected. Potential improvements could include:
- Hyperparameter tuning (learning rate, embedding dimensions, hidden layer sizes)
- Adjusting the model architecture
- Implementing better regularization techniques
- Using a more sophisticated training strategy

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
