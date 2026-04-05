# molecular-mirror

An AI system that discovers novel food recipes by finding ingredients with similar molecular flavor profiles. It combines graph neural networks trained on the [FlavorGraph](https://github.com/lamypark/FlavorGraph) dataset with a multi-agent LLM to propose creative, molecularly-grounded ingredient substitutions.

## How It Works

1. **Graph Autoencoder** — A GCN-based graph autoencoder (GAE) is trained on FlavorGraph, a knowledge graph of ingredients and their shared flavor compounds. Each ingredient is encoded into a 64-dimensional latent vector that captures its molecular flavor profile.

2. **Molecular Similarity Search** — Given any ingredient, cosine similarity over the learned embeddings surfaces the most molecularly similar ingredients in the graph — its *molecular mirrors*.

3. **Formulator Agent** — An AutoGen-powered LLM agent calls the similarity search iteratively to explore the flavor space and compose novel recipes that mimic the taste of a target dish using unexpected ingredients.

## Project Structure

```
molecular-mirror/
├── data/
│   ├── nodes_191120.csv          # FlavorGraph ingredient nodes
│   └── edges_191120.csv          # Flavor compound edges between ingredients
├── model/
│   ├── graph_autoencoder.py      # GCN encoder definition
│   └── train.py                  # Train GAE and export embeddings
├── agents/
│   ├── inference.py              # get_molecular_mirrors() similarity function
│   └── formulator.py             # Multi-agent recipe formulation orchestration
├── artifacts/
│   ├── molecular_mirror_weights.pth   # Saved model weights
│   └── ingredient_embeddings.pt       # Precomputed ingredient embeddings
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/gabriel-eidelman/molecular-mirror.git
cd molecular-mirror
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your OpenAI API key:

```
OPENAI_API_KEY=your_key_here
```

## Usage

### 1. Train the Model

Run this once to train the graph autoencoder and generate ingredient embeddings:

```bash
cd model
python train.py
```

This saves `molecular_mirror_weights.pth` and `ingredient_embeddings.pt` to `artifacts/`.

### 2. Find Molecular Mirrors

```python
from agents.inference import get_molecular_mirrors

get_molecular_mirrors("vanilla", top_k=5)
# --- Molecular Mirrors for: vanilla ---
# 1. tonka bean (Similarity: 0.9821)
# 2. licorice (Similarity: 0.9714)
# ...
```

### 3. Run the Formulator Agent

Launch the multi-agent system to generate novel recipes:

```bash
python agents/formulator.py
```

The agent calls `get_molecular_mirrors` autonomously across multiple ingredients, exploring molecular flavor space to compose recipes that recreate target flavors with unexpected ingredients.

## Model Details

| Component | Details |
|---|---|
| Dataset | FlavorGraph (Nov 2019 snapshot) |
| Architecture | 2-layer GCN encoder + inner product decoder (GAE) |
| Embedding dim | 64 |
| Node features | Identity matrix (one-hot, no chemical fingerprints) |
| Optimizer | Adam, lr=0.01 |
| Training | 100 epochs, link reconstruction loss |
| Similarity metric | Cosine similarity over latent embeddings |

**Note on node features:** The current implementation initializes node features as an identity matrix. A natural upgrade is to replace this with Morgan fingerprints (molecular circular fingerprints from RDKit), which would encode actual chemical structure into the initial node representations.

## Dependencies

- [PyTorch](https://pytorch.org/) + [PyTorch Geometric](https://pyg.org/) — GNN training
- [AG2 (AutoGen)](https://ag2.ai/) — multi-agent orchestration
- [FlavorGraph data](https://github.com/lamypark/FlavorGraph) — ingredient flavor network
- [RDKit](https://www.rdkit.org/) — molecular fingerprints (for future upgrades)
- pandas, scikit-learn

## License

MIT
