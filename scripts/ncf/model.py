import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from ..etl.etl import extract, get_train_test_splits
from ..util.model import Model

def get_best_device():
    """
    Returns the best available device (cuda, mps, or cpu)
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

class NeuralCollaborativeFiltering(nn.Module):
    '''
    Neural Collaborative Filtering model
    '''
    def __init__(self, num_users, num_items, embedding_dim, hidden_dim):
        '''
        Initialize the Neural Collaborative Filtering model
        '''
        super(NeuralCollaborativeFiltering, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Move model to best available device
        self.device = get_best_device()
        self.to(self.device)

    def forward(self, user_ids, item_ids):
        user_embeddings = self.user_embedding(user_ids)
        item_embeddings = self.item_embedding(item_ids)
        mlp_output = self.mlp(torch.cat([user_embeddings, item_embeddings], dim=-1))
        # Calculate dot product between user and item embeddings
        dot_product = (user_embeddings * item_embeddings).sum(dim=1, keepdim=True)
        return dot_product + mlp_output
    
class DeepModel(Model):
    def __init__(self):
        recipe_df, review_df = extract()
        # Get unique users and items for embedding dimensions
        self.user_encoder = {user_id: idx for idx, user_id in enumerate(review_df['user_id'].unique())}
        self.item_encoder = {item_id: idx for idx, item_id in enumerate(review_df['recipe_id'].unique())}
        
        num_users = len(self.user_encoder)
        num_items = len(self.item_encoder)
        embedding_dim = 64
        hidden_dim = 32
        
        self.ncf = NeuralCollaborativeFiltering(num_users, num_items, embedding_dim, hidden_dim)

    def __train(self):
        print("Training...")
        # Get train/test splits
        X_train, X_test, y_train, y_test = get_train_test_splits()
        
        # Convert to tensors and move to device
        train_user_ids = torch.tensor([self.user_encoder[uid] for uid in X_train['user_id']], dtype=torch.long).to(self.ncf.device)
        train_item_ids = torch.tensor([self.item_encoder[iid] for iid in X_train['recipe_id']], dtype=torch.long).to(self.ncf.device)
        train_ratings = torch.tensor(y_train.values, dtype=torch.float32).to(self.ncf.device)
        
        test_user_ids = torch.tensor([self.user_encoder[uid] for uid in X_test['user_id']], dtype=torch.long).to(self.ncf.device)
        test_item_ids = torch.tensor([self.item_encoder[iid] for iid in X_test['recipe_id']], dtype=torch.long).to(self.ncf.device)
        test_ratings = torch.tensor(y_test.values, dtype=torch.float32).to(self.ncf.device)
        
        # Create datasets and dataloaders
        train_dataset = TensorDataset(train_user_ids, train_item_ids, train_ratings)
        test_dataset = TensorDataset(test_user_ids, test_item_ids, test_ratings)
        
        batch_size = 1024
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.ncf.parameters(), lr=0.001)
        num_epochs = 100
        
        # Training loop
        for epoch in range(num_epochs):
            self.ncf.train()
            total_loss = 0
            
            for user_ids, item_ids, ratings in train_loader:
                optimizer.zero_grad()
                predictions = self.ncf(user_ids, item_ids).squeeze()
                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            self.ncf.eval()
            val_loss = 0
            with torch.no_grad():
                for user_ids, item_ids, ratings in test_loader:
                    predictions = self.ncf(user_ids, item_ids).squeeze()
                    val_loss += criterion(predictions, ratings).item()
            
            val_loss /= len(test_loader)
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Convert input to tensors and move to device
        user_ids = torch.tensor([self.user_encoder[uid] for uid in X['user_id']], dtype=torch.long).to(self.ncf.device)
        item_ids = torch.tensor([self.item_encoder[iid] for iid in X['recipe_id']], dtype=torch.long).to(self.ncf.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(user_ids, item_ids)
        dataloader = DataLoader(dataset, batch_size=1024)  # Use same batch size as training
        
        # Make predictions
        self.ncf.eval()
        all_predictions = np.array([])
        with torch.no_grad():
            for batch_user_ids, batch_item_ids in dataloader:
                predictions = self.ncf(batch_user_ids, batch_item_ids).squeeze()
                all_predictions = np.concatenate((all_predictions, predictions.cpu().numpy().flatten()))
        
        return all_predictions

    def __save(self):
        # Move model to CPU before saving
        self.ncf.to('cpu')
        torch.save({
            'model_state_dict': self.ncf.state_dict(),
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder
        }, 'models/ncf_model.pth')
        # Move model back to original device
        self.ncf.to(self.ncf.device)

    @staticmethod
    def get_instance():
        """
        Loading an existing model if available; otherwise, train a new one.

        Returns:
            DeepModel: A ready-to-use model instance
        """
        try:
            return DeepModel.__load()
        except Exception as e:
            print(f"Could not load DeepModel. Reason: {e}")
            print("Training new DeepModel...")
            return DeepModel.__create()

    @staticmethod
    def __load():
        checkpoint = torch.load('models/ncf_model.pth', weights_only=False)
        model = DeepModel()
        model.ncf.load_state_dict(checkpoint['model_state_dict'])
        model.user_encoder = checkpoint['user_encoder']
        model.item_encoder = checkpoint['item_encoder']
        # Move model to best available device
        model.ncf.to(get_best_device())
        return model

    @staticmethod
    def __create():
        model = DeepModel()
        model.__train()
        model.__save()
        return model

def main():
    model = DeepModel.get_instance()
    X_train, X_test, y_train, y_test = get_train_test_splits()
    print("Evaluating...")
    rmse = model.evaluate(X_test, y_test)
    print(f"RMSE: {rmse}")

if __name__ == "__main__":
    main()
