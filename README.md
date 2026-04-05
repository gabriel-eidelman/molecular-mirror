# molecular-mirror

An AI system that discovers novel food recipes by finding ingredients with similar molecular flavor profiles. It combines a GATv2 graph neural network trained on the [FlavorGraph](https://github.com/lamypark/FlavorGraph) dataset with a three-agent LLM pipeline to propose creative, molecularly-grounded ingredient substitutions.

## How It Works

1. **GATv2 Encoder** — A Graph Attention Network v2 (GATv2) encoder is trained on FlavorGraph, a knowledge graph of ingredients connected by shared flavor compounds. Node features are 2048-bit Morgan fingerprints (radius 2) computed via RDKit; ingredients without SMILES data fall back to zero vectors. Each ingredient is encoded into a 128-dimensional latent vector via a multi-head attention stack followed by a DeepChem-inspired projection head.

2. **Combined Training Loss** — Training minimizes a joint objective:
   `L = L_recon + λ * L_triplet`
   where `L_recon` is the standard GAE link-reconstruction loss and `L_triplet` is a triplet margin loss over (anchor, positive, negative) triples sampled from FlavorGraph edges. This pulls flavor-paired ingredients together in embedding space and pushes random negatives apart.

3. **Molecular Similarity Search** — At inference time, cosine similarity over the 128-dim projected embeddings surfaces the most molecularly similar ingredients — the *molecular mirrors* — for any target ingredient.

4. **Reason-then-Substitute Pipeline** — A three-agent AG2 workflow generates novel recipes:
   - **chemical_profiler** — Produces a structured Chemical Profile (dominant flavor compounds, aroma families, key ingredients to mirror) before any tool calls.
   - **mirror_finder** — Calls `get_molecular_mirrors` for each key ingredient and assembles a mirror candidates table annotated with physical states.
   - **substitution_agent** — Selects the best mirrors, applies physical-state quantity conversion rules (e.g. powder → liquid: ×3), and outputs the final recipe with formulator's notes.

## Project Structure

```
molecular-mirror/
├── data/
│   ├── nodes_191120.csv          # FlavorGraph ingredient nodes
│   ├── edges_191120.csv          # Flavor compound edges between ingredients
│   └── load_graph.py             # Graph loader with Morgan fingerprint featurization
├── model/
│   ├── graph_autoencoder.py      # GATv2 encoder + projection head
│   └── train.py                  # Train with recon + triplet loss; export embeddings
├── agents/
│   ├── inference.py              # get_molecular_mirrors() cosine similarity function
│   └── formulator.py             # Three-agent Reason-then-Substitute pipeline
├── artifacts/
│   ├── molecular_mirror_weights.pth   # Saved model weights
│   └── ingredient_embeddings.pt       # Precomputed 128-dim ingredient embeddings
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

### Optional: Morgan Fingerprint Features

To use real chemical structure as node features (instead of zero vectors), provide a SMILES file at `data/ingredient_smiles.csv`. Two schemas are supported:

| Schema | Columns |
|--------|---------|
| Single compound per ingredient | `name, smiles` |
| Multiple compounds per ingredient | `name, smiles_list` (semicolon-separated) |

You can generate this file by querying PubChem or extracting compound–ingredient mappings from the full FlavorGraph dataset. Without it, the model trains on zero-initialized features and still learns useful structure from the graph topology.

## Usage

### 1. Train the Model

Run this once to train the GATv2 encoder and generate ingredient embeddings:

```bash
python model/train.py
```

This saves `molecular_mirror_weights.pth` and `ingredient_embeddings.pt` to `artifacts/`.

### 2. Find Molecular Mirrors

```python
from agents.inference import get_molecular_mirrors

get_molecular_mirrors("vanilla", top_k=5)
# --- Molecular Mirrors for: vanilla (embed_dim=128) ---
#   1. tonka bean  (similarity: 0.9821)
#   2. licorice    (similarity: 0.9714)
#   ...
```

### 3. Run the Formulator Pipeline

Launch the three-agent Reason-then-Substitute workflow:

```bash
python agents/formulator.py
```

Or from Python:

```python
from agents.formulator import formulate_novel_ingredients

formulate_novel_ingredients("classic vanilla crème brûlée", num_recipes=2)
```

The pipeline runs in three sequential stages: the chemical profiler characterizes the dish's flavor chemistry, the mirror finder queries the embedding space for each key ingredient, and the substitution agent outputs a complete recipe with quantity-adjusted substitutions.

## Model Details

| Component | Details |
|---|---|
| Dataset | FlavorGraph (Nov 2019 snapshot) |
| Architecture | GATv2 (4 heads, 64 dim/head → 256) + single-head GATv2 → 128 + projection head |
| Embedding dim | 128 |
| Node features | 2048-bit Morgan fingerprints (radius 2) via RDKit; zero fallback |
| Optimizer | Adam, lr=0.005 |
| Training | 200 epochs, recon loss + triplet margin loss (λ=0.5, margin=0.3) |
| Triplets per batch | 1024 |
| Similarity metric | Cosine similarity over projected embeddings |

## Dependencies

- [PyTorch](https://pytorch.org/) + [PyTorch Geometric](https://pyg.org/) — GNN training
- [AG2](https://ag2.ai/) — multi-agent orchestration
- [RDKit](https://www.rdkit.org/) — Morgan fingerprint featurization
- [FlavorGraph data](https://github.com/lamypark/FlavorGraph) — ingredient flavor network
- pandas, numpy, scikit-learn, python-dotenv

## License

MIT
