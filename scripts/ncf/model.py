"""
Enhanced Neural Collaborative Filtering (NCF) for Recipe Recommendations

This script builds, trains, and saves a deep learning model that predicts how much a user will like a recipe based on past reviews. 
It uses user and recipe embeddings, a multi-layer perceptron (MLP), and bias terms to learn preferences.

Key features:
- PyTorch model with user/item embeddings and MLP
- Handles model training, saving, loading, and prediction
- Includes Optuna hyperparameter tuning support
- Device-aware (uses GPU/MPS if available)

To train and evaluate the model, just run: `python enhanced_ncf.py`
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os
import optuna

# These two functions come from your project
from ..etl.etl import extract, get_train_test_splits
from ..util.model import Model


def get_best_device():
    """
    Returns the best available device (cuda, mps, or cpu)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EnhancedNCF(nn.Module):
    """
    This is the main PyTorch model.
    It uses embeddings for users and items, adds bias terms, and passes them through a feedforward network (MLP).
    """

    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dims=[128, 64]):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim)
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim)

        # Bias terms help capture the general tendency of a user or item
        self.user_bias = nn.Embedding(num_users + 1, 1)
        self.item_bias = nn.Embedding(num_items + 1, 1)

        # Build the MLP layers
        layers = []
        input_dim = embedding_dim * 2
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))  # Helps avoid overfitting
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))

        self.mlp = nn.Sequential(*layers)
        self.device = get_best_device()
        self.to(self.device)

    def forward(self, user_ids, item_ids):
        """
        Forward pass through the model
        """
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        user_b = self.user_bias(user_ids).squeeze()
        item_b = self.item_bias(item_ids).squeeze()

        # Concatenate embeddings and pass through MLP
        x = torch.cat([user_emb, item_emb], dim=1)
        out = self.mlp(x).squeeze()

        # Final prediction includes biases
        return out + user_b + item_b


class EnhancedNCFModel(Model):
    """
    Wraps the NCF PyTorch model with train, save, load, and predict methods.
    """

    def __init__(self, embedding_dim=64, hidden_dims=[128, 64], lr=0.001, epochs=5):
        recipe_df, review_df = extract()
        self.reviews_df = review_df
        self.epochs = epochs
        self.lr = lr

        # Encode user and item IDs to indices
        self.user2idx = {uid: idx + 1 for idx, uid in enumerate(review_df['user_id'].unique())}
        self.item2idx = {iid: idx + 1 for idx, iid in enumerate(review_df['recipe_id'].unique())}
        self.unknown_user = 0
        self.unknown_item = 0
        self.global_mean = review_df['rating'].mean()

        self.model = EnhancedNCF(len(self.user2idx), len(self.item2idx), embedding_dim, hidden_dims)

    def _encode(self, user_ids, item_ids):
        """
        Turns user_id and recipe_id into integer indices for embedding layers.
        Falls back to 0 if not seen before.
        """
        u = torch.tensor([self.user2idx.get(uid, self.unknown_user) for uid in user_ids], dtype=torch.long)
        i = torch.tensor([self.item2idx.get(iid, self.unknown_item) for iid in item_ids], dtype=torch.long)
        return u, i

    def _train(self):
        """
        Train the NCF model on the training dataset.
        """
        print("Training EnhancedNCF model...")
        X_train, X_test, y_train, y_test = get_train_test_splits()

        # Encode data
        train_users, train_items = self._encode(X_train['user_id'], X_train['recipe_id'])
        test_users, test_items = self._encode(X_test['user_id'], X_test['recipe_id'])

        train_ratings = torch.tensor(y_train.values, dtype=torch.float32)
        test_ratings = torch.tensor(y_test.values, dtype=torch.float32)

        # Create DataLoaders
        train_data = TensorDataset(train_users, train_items, train_ratings)
        test_data = TensorDataset(test_users, test_items, test_ratings)

        train_loader = DataLoader(train_data, batch_size=512, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=512)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for users, items, ratings in train_loader:
                users, items, ratings = users.to(self.model.device), items.to(self.model.device), ratings.to(self.model.device)
                optimizer.zero_grad()
                preds = self.model(users, items)
                loss = criterion(preds, ratings)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for users, items, ratings in test_loader:
                    users, items, ratings = users.to(self.model.device), items.to(self.model.device), ratings.to(self.model.device)
                    preds = self.model(users, items)
                    val_loss += criterion(preds, ratings).item()

            print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(test_loader):.4f}")

        self._save()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts ratings for a batch of user-recipe pairs.
        """
        self.model.eval()
        users, items = self._encode(X['user_id'], X['recipe_id'])
        dataset = TensorDataset(users, items)
        loader = DataLoader(dataset, batch_size=512)

        predictions = []
        with torch.no_grad():
            for batch_users, batch_items in loader:
                batch_users, batch_items = batch_users.to(self.model.device), batch_items.to(self.model.device)
                preds = self.model(batch_users, batch_items).cpu().numpy()
                predictions.extend(preds)

        return np.array(predictions)

    def _save(self):
        """
        Save model and encoders to disk
        """
        os.makedirs("models", exist_ok=True)
        self.model.to("cpu")
        torch.save({
            'model_state': self.model.state_dict(),
            'user2idx': self.user2idx,
            'item2idx': self.item2idx,
            'global_mean': self.global_mean
        }, "models/enhanced_ncf.pth")
        self.model.to(self.model.device)

    @staticmethod
    def _load():
        """
        Load model and encoders from disk
        """
        checkpoint = torch.load("models/enhanced_ncf.pth")
        instance = EnhancedNCFModel()
        instance.model.load_state_dict(checkpoint['model_state'])
        instance.user2idx = checkpoint['user2idx']
        instance.item2idx = checkpoint['item2idx']
        instance.global_mean = checkpoint['global_mean']
        instance.model.to(instance.model.device)
        return instance

    @staticmethod
    def get_instance():
        """
        Try loading model; if not found, train a new one.
        """
        try:
            return EnhancedNCFModel._load()
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Training a new one...")
            model = EnhancedNCFModel()
            model._train()
            return model


def objective(trial):
    """
    Optuna trial function to search over hyperparameters.
    """
    embedding_dim = trial.suggest_categorical("embedding_dim", [32, 64, 128])
    hidden_dims = trial.suggest_categorical("hidden_dims", [[64, 32], [128, 64], [256, 128]])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    model = EnhancedNCFModel(embedding_dim=embedding_dim, hidden_dims=hidden_dims, lr=lr, epochs=10)
    model._train()
    _, X_test, _, y_test = get_train_test_splits()
    return model.evaluate(X_test, y_test)


def main():
    """
    Runs hyperparameter search, trains final model, and reports RMSE.
    """
    print("Running Optuna hyperparameter search...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)
    print("Best trial:", study.best_trial.params)

    # Train best model found
    best = study.best_trial.params
    final_model = EnhancedNCFModel(embedding_dim=best['embedding_dim'],
                                   hidden_dims=best['hidden_dims'],
                                   lr=best['lr'],
                                   epochs=5)
    final_model._train()

    # Evaluate
    _, X_test, _, y_test = get_train_test_splits()
    rmse = final_model.evaluate(X_test, y_test)
    print(f"✅ Final RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
