import torch
import pandas as pd
from typing import Annotated, Literal

# Load the data you just generated
z = torch.load('ingredient_embeddings.pt')
nodes = pd.read_csv('inputs/nodes_191120.csv')

def get_molecular_mirrors(target_name: Annotated[str, "the name of the food item"], top_k=5):
    # 1. Find the index of your target ingredient
    try:
        target_idx = nodes[nodes['name'].str.contains(target_name, case=False)].index[0]
        actual_name = nodes.iloc[target_idx]['name']
    except IndexError:
        return f"Ingredient '{target_name}' not found in FlavorGraph."

    # 2. Get the latent vector (embedding) for the target
    target_vec = z[target_idx].unsqueeze(0) # Shape [1, 64]

    # 3. Calculate Cosine Similarity against ALL other nodes
    # Cosine similarity is better than Euclidean distance for high-dimensional embeddings
    similarities = torch.nn.functional.cosine_similarity(target_vec, z)

    # 4. Get the top-K matches
    values, indices = torch.topk(similarities, k=top_k + 1)

    print(f"\n--- Molecular Mirrors for: {actual_name} ---")
    results = []
    for i in range(1, len(indices)): # Start at 1 to skip the target itself
        node_name = nodes.iloc[indices[i].item()]['name']
        node_group = nodes.iloc[indices[i].item()]['node_type'] # Check if it's 'ingr'
        score = values[i].item()
        results.append((node_name, score))
        print(f"{i}. {node_name} (Similarity: {score:.4f})")
    
    return results

# Test it out! Try 'beef', 'butter', 'egg', or 'milk'
get_molecular_mirrors("milk")