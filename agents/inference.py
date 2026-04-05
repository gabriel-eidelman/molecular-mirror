from pathlib import Path
import torch
import pandas as pd
from typing import Annotated

ROOT = Path(__file__).parent.parent

z = torch.load(ROOT / 'artifacts/ingredient_embeddings.pt')
nodes = pd.read_csv(ROOT / 'data/nodes_191120.csv')

def get_molecular_mirrors(target_name: Annotated[str, "the name of the food item"], top_k=5):
    # 1. Find the index of your target ingredient
    try:
        target_idx = nodes[nodes['name'].str.contains(target_name, case=False)].index[0]
        actual_name = nodes.iloc[target_idx]['name']
    except IndexError:
        return f"Ingredient '{target_name}' not found in FlavorGraph."

    # 2. Get the latent vector (embedding) for the target
    target_vec = z[target_idx].unsqueeze(0)  # Shape [1, 64]

    # 3. Calculate Cosine Similarity against ALL other nodes
    similarities = torch.nn.functional.cosine_similarity(target_vec, z)

    # 4. Get the top-K matches
    values, indices = torch.topk(similarities, k=top_k + 1)

    print(f"\n--- Molecular Mirrors for: {actual_name} ---")
    results = []
    for i in range(1, len(indices)):  # Start at 1 to skip the target itself
        node_name = nodes.iloc[indices[i].item()]['name']
        score = values[i].item()
        results.append((node_name, score))
        print(f"{i}. {node_name} (Similarity: {score:.4f})")

    return results