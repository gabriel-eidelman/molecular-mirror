import pandas as pd
import torch
from torch_geometric.data import Data
from graph_autoencoder import GCNEncoder
from torch_geometric.nn import GAE

# 1. Load data
nodes = pd.read_csv('inputs/nodes_191120.csv')
edges = pd.read_csv('inputs/edges_191120.csv')

# 2. Create a mapping for IDs to 0-indexed integers
node_map = {old_id: i for i, old_id in enumerate(nodes['node_id'])}
edge_index = torch.tensor([
    [node_map[s], node_map[t]] for s, t in zip(edges['id_1'], edges['id_2'])
], dtype=torch.long).t().contiguous()

# 3. Features: If you don't have chemical fingerprints yet, use an Identity Matrix
# In an interview, explain: "I used an Identity init, but would upgrade to Morgan Fingerprints."
x = torch.eye(len(nodes)) 
data = Data(x=x, edge_index=edge_index)

# Initialize GAE with our Encoder
model = GAE(GCNEncoder(len(nodes), 64)) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)
    # The GAE loss calculates how well we can predict edges vs non-edges
    loss = model.recon_loss(z, data.edge_index) 
    loss.backward()
    optimizer.step()
    return float(loss)

for epoch in range(100):
    loss = train()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}")


# 1. Save the Model Weights
# This is what you'll load later in your 'inference' or 'agent' script
model_path = 'molecular_mirror_weights.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# 2. Extract the Final Embeddings (The "Latent Space")
model.eval()
with torch.no_grad():
    # 'z' is the N-dimensional vector representation of your ingredients
    z = model.encode(data.x, data.edge_index)
    
# Save embeddings for your AG2 agents to use (avoids re-running the GNN every time)
torch.save(z, 'ingredient_embeddings.pt')
print("Latent embeddings saved to ingredient_embeddings.pt")