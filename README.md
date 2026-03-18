# DISCODE
![DISCODE](/DISCODE.png)
**DISCODE** (**D**eep learning-based **I**terative pipeline to analyze **S**pecificity of **CO**factors and to **D**esign **E**nzyme) is a transformer-based NAD/NADP classification model. The model uses ESM-2 language model for amino acid embedding and represents the probability of NAD and NADP specificity.

## Installation

**Note: Developed and tested on Linux (Ubuntu 18.04+), Python 3.8+.**

### Option 1: pip install (recommended)
```bash
git clone https://github.com/SBML-Kimlab/DISCODE.git
cd DISCODE
pip install .
```

### Option 2: conda environment
```bash
git clone https://github.com/SBML-Kimlab/DISCODE.git
cd DISCODE
conda env create -f discode.yaml
conda activate discode
```

## Usage

### Classification

```python
from discode import models, utils

model = models.load("weights/weights.pt")  # automatically loads on GPU if available
model.eval()

name, sequence = "3M6I", "MASSASKTNIG..."
dataloader = utils.tokenize_and_dataloader(name, sequence)

outlier_idx, probability, predicted_label, _name, attention_weights = utils.model_prediction(dataloader, model)
# outlier_idx: zero-indexed positions of salient residues
# probability: [NAD_prob, NADP_prob]

print(f"NAD: {probability.numpy()[0]:.3f}, NADP: {probability.numpy()[1]:.3f}")
```

### Visualization

```python
utils.make_max_attention_map(attention_weights)
# Plots maximum attention map across all layers/heads [8 x 20]

utils.plot_attention_sum(attention_weights, sequence, threshold="2S")
# Plots per-residue attention sum with outlier threshold line
```

Supported thresholds: `"1S"`, `"2S"`, `"3S"`, `"IQR"`, `"P90"`, `"P95"`, `"P99"`

### Mutation Design

Results are saved at each step as `{pickle_path}/{name}_{mode}_mutation_{step}.pkl`.

```python
utils.scan_switch_mutation(
    model=model,
    name=name,
    sequence=sequence,
    mode="shortest",       # "shortest", "iter_num", or "iter_prob"
    max_num_mutation=3,
    max_num_solution=20,
    prob_thres=0.5,
    threshold="2S",
    batch_size=32,
    pickle_path=".",
)
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `model` | — | Loaded `TransformerClassifier` model |
| `name` | `"unknown"` | Protein accession or identifier |
| `sequence` | — | Wildtype amino acid sequence (single-letter) |
| `mode` | `"iter_num"` | Mutation search mode (see below) |
| `max_num_mutation` | `3` | Maximum number of mutations per design (search depth) |
| `max_num_solution` | `50` | Maximum number of results returned in the final DataFrame |
| `prob_thres` | `0.5` | Probability threshold for cofactor switching (used in `iter_prob` mode) |
| `threshold` | `"2S"` | Outlier detection method for selecting salient residues (`"1S"`, `"2S"`, `"3S"`, `"IQR"`, `"P90"`, `"P95"`, `"P99"`) |
| `batch_size` | `32` | Number of mutation candidates processed per GPU batch |
| `pickle_path` | `"."` | Directory to save intermediate results as `.pkl` files |

**Modes:**
- `shortest` — greedily selects the single highest-probability mutation at each step. Fastest; finds the minimal mutation path.
- `iter_num` — exhaustively scans all candidate combinations and stops as soon as at least one converting mutation is found.
- `iter_prob` — exhaustively scans all combinations across all steps and returns all results sorted by switching probability.

A complete example is provided in [example/example.ipynb](example/example.ipynb).

## Contact
If you have any questions, problems or suggestions, please contact [us](https://sites.google.com/view/systemskimlab/home).

## Citation
Kim J., Woo J., Park J.Y., Kim K.J., Kim D. Deep learning for NAD/NADP cofactor prediction and engineering using transformer attention analysis in enzymes. Metab. Eng. 2025; 87:86-94. https://doi.org/10.1016/j.ymben.2024.11.007.

## Reference
1. A. Vaswani et al., Attention Is All You Need. Adv Neur In 30 (2017).
2. Z. M. Lin et al., Evolutionary-scale prediction of atomic-level protein structure with a language model. Science 379, 1123-1130 (2023).