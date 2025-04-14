import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
import torch.optim as optim
import time
from tqdm import tqdm
from sklearn.metrics import mean_squared_error


class RecipeRecommenderGNN(nn.Module):
    """
    A Heterogeneous Graph Neural Network for personalized recipe recommendations.
    
    It applies multiple layers of SAGEConv across different edge types
    to learn user and recipe embeddings in a shared space.
    """
    def __init__(self, metadata, hidden_channels=64, out_channels=64, num_layers=2):
        super().__init__()
        self.embeddings = nn.ModuleDict()

        # Initialize learnable embeddings for node types that don't have features (e.g. users, ingredients)
        for node_type in metadata[0]:
            if node_type != 'recipe':
                self.embeddings[node_type] = nn.Embedding(100000, hidden_channels)

        # Project recipe features into hidden embedding space
        self.input_linear = Linear(9, hidden_channels)

        # Build a stack of heterogeneous convolution layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in metadata[1]
            }, aggr='sum')
            self.convs.append(conv)

        # Linear layers to map final embeddings for prediction
        self.lin_user = Linear(hidden_channels, out_channels)
        self.lin_recipe = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, num_nodes_dict=None):
        """
        Perform message passing and return final node embeddings.
        """
        x_dict_out = {}
        for node_type, x in x_dict.items():
            if node_type == 'recipe':
                # Recipes already have features; project them to hidden space
                x_dict_out[node_type] = self.input_linear(x)
            else:
                # Other nodes use learned embeddings
                x_dict_out[node_type] = self.embeddings[node_type](x)

        # Apply multi-layer GNN
        for conv in self.convs:
            prev = x_dict_out
            updated = conv(prev, edge_index_dict)

            # Fallback to previous features if any node type is missing
            for node_type in prev:
                if node_type not in updated or updated[node_type] is None:
                    updated[node_type] = prev[node_type]

            # Apply ReLU to all updated features
            x_dict_out = {k: F.relu(v) for k, v in updated.items()}

        return x_dict_out

    def predict(self, x_user, x_recipe):
        """
        Predict preference score between user and recipe embeddings.
        Cosine similarity is used after L2 normalization.
        """
        u = F.normalize(self.lin_user(x_user), p=2, dim=-1)
        r = F.normalize(self.lin_recipe(x_recipe), p=2, dim=-1)
        return (u * r).sum(dim=-1)


class RecommenderDeployWrapper(torch.nn.Module):
    """
    Wrapper module to allow deployment of the GNN model in TorchScript.
    """
    def __init__(self, model, user_embed, recipe_embed):
        super().__init__()
        self.user_proj = model.lin_user
        self.recipe_proj = model.lin_recipe
        self.user_embed = user_embed
        self.recipe_embed = recipe_embed

    def forward(self, user_id: int):
        """
        Given a user ID, return preference scores for all recipes.
        """
        u = self.user_proj(self.user_embed[user_id])
        r = self.recipe_proj(self.recipe_embed)
        return (u * r).sum(dim=1)


def export_script_model(model, out_dict, path="recipe_gnn_script_100k.pt"):
    """
    Export the trained model using TorchScript for deployment.
    """
    wrapper = RecommenderDeployWrapper(
        model,
        out_dict['user'].detach(),
        out_dict['recipe'].detach()
    )
    wrapper.eval()
    script = torch.jit.script(wrapper)
    script.save(path)
    print(f"TorchScript model saved to {path}")


def get_pos_neg_edges(data, num_neg=1):
    """
    Sample positive and negative edges for training.
    Returns user and recipe indices for both positive and negative interactions.
    """
    pos_u, pos_r = data['user', 'rates', 'recipe'].edge_index
    num_pos = pos_u.size(0)

    # Create negative samples by pairing users with random recipes
    neg_u = pos_u.repeat_interleave(num_neg)
    neg_r = torch.randint(
        0, data['recipe'].num_nodes,
        size=(num_pos * num_neg,),
        device=pos_u.device
    )
    return pos_u, pos_r, neg_u, neg_r


def main():
    """
    Main training loop for the recipe recommender model.
    """
    # === Load data and move to device ===
    data = torch.load("recipe_graph_100k.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    # Add missing edge types with empty edge indices
    for edge_type in data.metadata()[1]:
        if edge_type not in data.edge_index_dict or data.edge_index_dict[edge_type] is None:
            data.edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long, device=device)

    model = RecipeRecommenderGNN(metadata=data.metadata()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 10
    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        start = time.time()

        # Construct input features
        x_dict = {}
        for node_type in data.metadata()[0]:
            if node_type == 'recipe':
                x_dict[node_type] = data[node_type].x
            else:
                x_dict[node_type] = torch.arange(data.num_nodes_dict[node_type], device=device)

        # Forward pass
        out_dict = model(x_dict, data.edge_index_dict, data.num_nodes_dict)

        # Get positive and negative samples
        pos_u, pos_r, neg_u, neg_r = get_pos_neg_edges(data)

        pos_scores = model.predict(out_dict['user'][pos_u], out_dict['recipe'][pos_r])
        neg_scores = model.predict(out_dict['user'][neg_u], out_dict['recipe'][neg_r])

        # Create binary labels and compute loss
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        scores = torch.cat([pos_scores, neg_scores])

        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}/{EPOCHS} — Loss: {loss.item():.4f} — Time: {time.time() - start:.2f}s")

    print("\nTraining complete!")

    # === Export model and embeddings ===
    print("Exporting model...")
    model.eval()
    x_dict = {
        node_type: (
            data[node_type].x if node_type == 'recipe'
            else torch.arange(data.num_nodes_dict[node_type], device=device)
        )
        for node_type in data.metadata()[0]
    }

    final_embs = model(x_dict, data.edge_index_dict, data.num_nodes_dict)
    export_script_model(model, final_embs)

    # Save raw PyTorch checkpoint
    torch.save(model.state_dict(), "recipe_model.pt")


class RecipeGNNModel:
    """
    Wrapper for evaluating a trained GNN model using internal index-based test data.
    """
    def __init__(self, model, data, device):
        self.model = model.eval()
        self.data = data
        self.device = device
        self.out_dict = self._generate_embeddings()

    def _generate_embeddings(self):
        """
        Generate final embeddings for all nodes in the graph.
        """
        x_dict = {
            node_type: (
                self.data[node_type].x if node_type == 'recipe'
                else torch.arange(self.data.num_nodes_dict[node_type], device=self.device)
            )
            for node_type in self.data.metadata()[0]
        }

        return self.model(x_dict, self.data.edge_index_dict, self.data.num_nodes_dict)

    def evaluate_from_index(self, test_df):
        """
        Evaluate model predictions using RMSE against ground truth ratings.
        
        Args:
            test_df (pd.DataFrame): DataFrame with 'u', 'i', and 'rating' columns.
        
        Returns:
            float: Root Mean Squared Error (RMSE)
        """
        # Remove out-of-bound user/recipe indices
        test_df = test_df[
            (test_df['u'] < self.out_dict['user'].shape[0]) &
            (test_df['i'] < self.out_dict['recipe'].shape[0])
        ].copy()

        if test_df.empty:
            raise ValueError("No valid test rows found after filtering index bounds.")

        user_idx = torch.tensor(test_df['u'].values, dtype=torch.long, device=self.device)
        recipe_idx = torch.tensor(test_df['i'].values, dtype=torch.long, device=self.device)
        true_ratings = test_df['rating'].values

        with torch.no_grad():
            preds = self.model.predict(
                self.out_dict['user'][user_idx],
                self.out_dict['recipe'][recipe_idx]
            ).sigmoid() * 5.0  # Scale to match rating range

        rmse = mean_squared_error(true_ratings, preds.cpu().numpy(), squared=False)
        return rmse


if __name__ == "__main__":
    main()
