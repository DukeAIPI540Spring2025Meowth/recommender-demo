import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
import torch.optim as optim
import time
from tqdm import tqdm

# This script defines and trains a graph neural network model for recipe recommendation using our before created heterogeneous graph data. 
# It uses PyTorch Geometric to model users, recipes, and related entities like ingredients, tags, and review words. 
# Non-recipe nodes are initialized with learnable embeddings, while recipe nodes use handcrafted features. 
# The model applies multi-layer SAGE-based convolutions to update node embeddings.


# === GNN Model Definition ===
class RecipeRecommenderGNN(nn.Module):
    def __init__(self, metadata, hidden_channels=64, out_channels=64, num_layers=2):
        super().__init__()
        self.embeddings = nn.ModuleDict()

        # Learnable embeddings for all node types except 'recipe'
        for node_type in metadata[0]:
            if node_type != 'recipe':
                self.embeddings[node_type] = nn.Embedding(100000, hidden_channels)

        # Linear layer to project recipe features into embedding space
        self.input_linear = Linear(9, hidden_channels)

        # Multi-layer heterogeneous GNN with SAGE-style convolution
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in metadata[1]
            }, aggr='sum')
            self.convs.append(conv)

        self.lin_user = Linear(hidden_channels, out_channels)
        self.lin_recipe = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, num_nodes_dict=None):
        # Apply initial embeddings or projections
        x_dict_out = {}
        for node_type in x_dict:
            if node_type == 'recipe':
                x_dict_out[node_type] = self.input_linear(x_dict[node_type])
            else:
                x_dict_out[node_type] = self.embeddings[node_type](x_dict[node_type])

        # Apply multiple layers of graph convolution
        for conv in self.convs:
            prev_dict = x_dict_out
            x_dict_out_new = conv(prev_dict, edge_index_dict)

            # Keep old values if a node type wasn't updated
            for node_type in prev_dict:
                if node_type not in x_dict_out_new or x_dict_out_new[node_type] is None:
                    x_dict_out_new[node_type] = prev_dict[node_type]

            # Apply non-linearity
            x_dict_out = {k: F.relu(v) for k, v in x_dict_out_new.items()}

        return x_dict_out

    def predict(self, x_user, x_recipe):
        # Dot product between user and recipe embeddings
        u = self.lin_user(x_user)
        r = self.lin_recipe(x_recipe)
        return (u * r).sum(dim=-1)


# === TorchScript Wrapper for Exporting ===
class RecommenderDeployWrapper(torch.nn.Module):
    def __init__(self, model, user_embed, recipe_embed):
        super().__init__()
        self.user_proj = model.lin_user
        self.recipe_proj = model.lin_recipe
        self.user_embed = user_embed
        self.recipe_embed = recipe_embed

    def forward(self, user_id: int):
        # Compute scores for all recipes for a given user
        u = self.user_proj(self.user_embed[user_id])
        r = self.recipe_proj(self.recipe_embed)
        scores = (u * r).sum(dim=1)
        return scores


def export_script_model(model, out_dict, path="recipe_gnn_script_100k.pt"):
    wrapper = RecommenderDeployWrapper(
        model,
        out_dict['user'].detach(),
        out_dict['recipe'].detach()
    )
    wrapper.eval()
    script = torch.jit.script(wrapper)
    script.save(path)
    print(f"TorchScript model saved to {path}")


# === Load Data and Setup ===
data = torch.load("recipe_graph_100k.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)

# Ensure all expected edge types are defined in the graph
for edge_type in data.metadata()[1]:
    if edge_type not in data.edge_index_dict or data.edge_index_dict[edge_type] is None:
        data.edge_index_dict[edge_type] = torch.empty((2, 0), dtype=torch.long, device=device)

# === Initialize Model and Optimizer ===
model = RecipeRecommenderGNN(metadata=data.metadata()).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# === Generate Negative Samples ===
def get_pos_neg_edges(data, num_neg=1):
    pos_u, pos_r = data['user', 'rates', 'recipe'].edge_index
    num_pos = pos_u.size(0)
    neg_u = pos_u.repeat_interleave(num_neg)
    neg_r = torch.randint(0, data['recipe'].num_nodes, (num_pos * num_neg,), device=pos_u.device)
    return pos_u, pos_r, neg_u, neg_r


# === Training Loop ===
EPOCHS = 10
print("\nStarting training...\n")

# During training, it uses positive and randomly sampled negative user-recipe pairs to learn preference scores via binary classification. 
# After training, the model is wrapped for deployment using TorchScript, enabling fast, production-ready inference where 
# recommendations are made by computing similarity scores between user and recipe embeddings.


for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    start = time.time()

    # Prepare input features for each node type
    x_dict = {}
    for node_type in data.metadata()[0]:
        if node_type == 'recipe':
            x_dict[node_type] = data[node_type].x
        else:
            if node_type not in data.num_nodes_dict:
                raise ValueError(f"Node type '{node_type}' not found in data.num_nodes_dict.")
            x_dict[node_type] = torch.arange(data.num_nodes_dict[node_type], device=device)

    # Run the model to get embeddings
    out_dict = model(x_dict, data.edge_index_dict, data.num_nodes_dict)

    # Generate positive and negative pairs
    pos_u, pos_r, neg_u, neg_r = get_pos_neg_edges(data)

    pos_scores = model.predict(out_dict['user'][pos_u], out_dict['recipe'][pos_r])
    neg_scores = model.predict(out_dict['user'][neg_u], out_dict['recipe'][neg_r])

    # Binary labels: 1 for positive, 0 for negative
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
    scores = torch.cat([pos_scores, neg_scores])

    # Binary cross-entropy loss on prediction
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}/{EPOCHS} — Loss: {loss.item():.4f} — Time: {time.time() - start:.2f}s")

print("\nTraining complete!")

# === Export Model as TorchScript ===
print("Exporting model...")
model.eval()

# Build full input for export
x_dict = {}
for node_type in data.metadata()[0]:
    if node_type == 'recipe':
        x_dict[node_type] = data[node_type].x
    else:
        x_dict[node_type] = torch.arange(data.num_nodes_dict[node_type], device=device)

final_embs = model(x_dict, data.edge_index_dict, data.num_nodes_dict)
export_script_model(model, final_embs)
