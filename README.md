# UDA
Code accompanying the paper **"Unsupervised Domain Adaptation for Binary Classification with an Unobservable Source Subpopulation"**
([arXiv:2509.20587](https://arxiv.org/abs/2509.20587)). (accepted by TMLR)

This repository estimates the **target-domain conditional outcome probability**
`η(x) = P(Y = 1 | X = x)` when a binary **spurious / sensitive attribute** `A`
is correlated with the label `Y`, and that correlation **shifts** between a
labeled *source* domain and an *unlabeled* target domain. We never observe `Y`
in the target; instead we estimate the target class proportions and re-weight
source predictions to recover the correct conditional probabilities for the
target, including within each attribute subgroup.

The method is evaluated on three benchmarks: **Waterbirds**, **CelebA**, and
**SOFA** (a tabular ICU dataset).

---

## Problem setup

Each example is a triple `(X, Y, A)`:

| Symbol | Meaning | Waterbirds | CelebA | SOFA |
| --- | --- | --- | --- | --- |
| `X` | features / embedding | image embedding | image embedding | 16 tabular features `X1…X16` |
| `Y` | binary label | bird type (`y`) | `Male` | `y` |
| `A` | binary spurious attribute | background / `place` | `Gray_Hair` | `a` |

There are two domains:

- **Source** (`R = 1`): labels `Y` are observed.
- **Target** (`R = 0`): only `X` (and `A`) are observed.

The joint distribution `P(Y, A)` differs between source and target — i.e. the
strength of the spurious correlation changes — while the class-conditional
feature distribution `P(X | Y, A)` is assumed stable. The goal is to recover the
target subgroup conditionals `η0 = P(Y=1 | X, A=0)`, `η1 = P(Y=1 | X, A=1)`, and
the overall `η`, using source labels and target covariates only.

---

## Method overview

The core estimator lives in each dataset's `ood.py`:

1. **`ood(source, target, method)`** estimates the target class-proportion
   vector `β = (β00, β01, β10, β11)` where `βya = P(Y=y, A=a)` in the target.
   Two estimators are provided:
   - `method='distribution'` — a likelihood-based estimator solved with a 1-D
     bounded optimization (`scipy.optimize.minimize_scalar`).
   - `method='moment1' | 'moment2' | 'moment3'` — method-of-moments estimators
     using feature moments.
2. **`ood_predict(...)` / `calculate(...)`** combine the estimated `β` with the
   source class proportions `α` and logistic models fit on the source to
   produce density-ratio–corrected estimates of `η0`, `η1`, and `η` for the
   target.
3. **Benchmarks**: naive source-only logistic predictions (`eta0_bench`,
   `eta1_bench`, `eta_bench`) and a baseline `gemma` (`get_gemma`) are computed
   for comparison.

---

## Repository structure

```
UDA/
├── ood_waterbird/      # Waterbirds image benchmark
│   ├── model.py            # pretrained backbones (ResNet18/50, VGG11/19, ViT-B/16)
│   ├── embeddings.py       # extract frozen features from Waterbirds images
│   ├── ood.py              # core estimators (distribution / moment)
│   ├── run.py              # sweep over (a, b, c, seed), save results to .pkl
│   ├── evaluate.py         # AUC / F1 / ACC tables from the saved .pkl
│   └── experiment_waterbird.ipynb
├── ood_celeA/          # CelebA image benchmark
│   ├── model.py
│   ├── embedding.py        # extract frozen features from CelebA images
│   ├── switch_y_a.py       # build label/attribute-flipped metadata variants
│   ├── ood.py / run.py / evaluate.py
│   └── experiment_celeA.ipynb
└── ood_sofa/           # SOFA tabular ICU benchmark
    ├── ood.py
    ├── run.py              # reads SOFA features directly (no embedding step)
    ├── switch_y_a.py
    ├── evaluate.py
    └── experiment_SOFA.ipynb
```

Each folder also contains the generated `experiments*.csv` /
`table_*_{acc,f}_varying_{b,c}.csv` result tables and the `eta*_*.pdf` figures.

---

## Installation

```bash
git clone https://github.com/haotianzh/UDA.git
cd UDA
python -m venv .venv && source .venv/bin/activate   # optional

pip install numpy pandas scipy scikit-learn \
            torch torchvision torchsummary \
            pillow tqdm matplotlib jupyter
```

A CUDA-capable GPU is recommended for the embedding-extraction step
(`embeddings.py` / `embedding.py` call `.cuda()`); the estimation and evaluation
steps run on CPU.

---

## Data preparation

**Waterbirds** — download `waterbird_complete95_forest2water2` and its
`metadata.csv` (CUB + Places spurious-correlation benchmark), then point the
paths at the bottom of `ood_waterbird/embeddings.py` to your copy.

**CelebA** — download the aligned CelebA images and attribute annotations, then
set the paths in `ood_celeA/embedding.py`. The label used here is `Male` and the
spurious attribute is `Gray_Hair`.

**SOFA** — already included as tabular CSVs (`SOFA.csv` and the
`SOFA_switch_*.csv` variants), with 16 features `X1…X16` plus `a` and `y`. No
embedding step is needed.

---

## Usage

### 1. Extract embeddings (image datasets only)

```bash
cd ood_waterbird            # or ood_celeA
python embeddings.py        # ood_celeA uses: python embedding.py
```

This writes `embeds/embeds_<backbone>.npy` and `embeds/metadata_<backbone>.csv`.
Choose the backbone (`resnet18`, `resnet50`, `vgg11`, `vgg19`, `vit16`) at the
top of the script. For SOFA, skip this step.

### 2. Run the experiment sweep

```bash
python run.py
```

`run.py` sweeps the source-sampling fractions `a`, `b`, `c` (which control the
induced spurious correlation in the source domain) across several random seeds,
fits the estimators and benchmarks, and pickles everything (e.g.
`experiment_resnet50.pkl`). Edit the `ranges`, seed count, and backbone/metadata
paths in `run.py` to change the grid.

### 3. Evaluate

```bash
python evaluate.py
```

Reads the pickled results and writes a tidy CSV of **AUC / F1 / ACC** for each
estimator (`eta0`, `eta1`, `eta`, their `*_bench` baselines, and `gemma`) across
all `(a, b, c, seed)` settings. The accompanying notebooks
(`experiment_*.ipynb`) reproduce the summary tables and the `eta*_*.pdf`
figures.

> Optional: `switch_y_a.py` regenerates the label/attribute-flipped metadata
> variants (`*_switch_a`, `*_switch_y`, `*_switch_a_y`) used in robustness checks.

---

## Reproducing the figures

The `*.ipynb` notebooks in each folder load the evaluation CSVs and produce the
`eta0_a_0.5_*.pdf`, `eta1_eta_a_0.5_*.pdf`, and `eta1_eta_a_0.7_*.pdf` plots
showing how the estimators and benchmarks behave as the source spurious-
correlation strength varies.

---

## Citation

<!-- TODO: replace with the official BibTeX once confirmed from arXiv. -->

```bibtex
@misc{ying2026unsuperviseddomainadaptationbinary,
      title={Unsupervised Domain Adaptation for Binary Classification with an Unobservable Source Subpopulation}, 
      author={Chao Ying and Jun Jin and Haotian Zhang and Qinglong Tian and Yanyuan Ma and Sharon Li and Jiwei Zhao},
      year={2026},
      eprint={2509.20587},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2509.20587}, 
}
```

---

## License

MIT
