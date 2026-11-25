# When Accuracy Saturates, Reliability Leads

Code, result tables and figures for a staged evaluation of 3D medical image segmentation:

- **Stage 1 (accuracy).** Four architectures that share one 3D residual U-Net backbone — baseline, Fresnel propagation module, spectral edge-attention (SEA) module, and a decoder-dropout variant — trained with five seeds each. Paired Wilcoxon tests with Holm correction, per-seed signs, two one-sided equivalence tests (TOST) with margins fixed before analysis (pancreas 0.03, mass 0.05, hippocampus 0.02 Dice), and the minimum detectable difference at 80% power.
- **Stage 2 (reliability).** On the same held-out test set: predictive entropy, test-time augmentation (6 passes), deep ensemble (5 members), MC Dropout (10 passes), Fresnel feature coherence and Fresnel decision-space disagreement. Each estimator is scored with its own prediction and confidence, in the boundary band, per class and over the whole volume (failure-detection AUROC, ECE, risk–coverage AUC, run time). Patient-level bootstrap 95% confidence intervals and Holm-corrected paired tests against entropy.

Two tasks from the Medical Segmentation Decathlon are used: **Task07 Pancreas** (pancreas and mass, 112³ voxels, 42 test cases) and **Task04 Hippocampus** (anterior and posterior, 48×64×48 voxels).

## Repository layout

| Path | Content |
|---|---|
| `Accuracy_Saturation_Pancreas_Hippocampus.ipynb` | Complete pipeline for Google Colab: data download and caching, Stage-1 training and statistics, Stage-2 estimators and statistics, qualitative panels, hippocampus replication. Resumable; every long step prints progress. |
| `data/pancreas_test_cases.txt` | The fixed held-out pancreas test set (42 case IDs). The notebook checks that its rebuilt split equals this list. |
| `results/Tables/` | Per-case Dice/HD95 for every task × architecture × seed, Stage-1 summaries and tests, Stage-2 per-case metrics, summaries and tests, and the Fresnel-coherence diagnostic. |
| `results/Tables/additional/` | Additional reliability analyses: fitted temperatures, temperature-scaled entropy, decomposition of ensemble and MC-dropout entropy into expected entropy and mutual information, per-case voxel metrics, summaries and tests, and case-level reliability (mean uncertainty per case against case Dice). `results/additional_results.json` bundles them. |
| `results/Figures/` | Figures of the study (framework and estimator overviews, Stage-1 equivalence, Stage-2 estimators, reliability and risk–coverage curves, coherence diagnostic, qualitative panels). |
| `analysis/` | Scripts that turn the tables into statistics, LaTeX tables and figures. |

## Training protocol

Adam (lr 2·10⁻³, weight decay 10⁻⁵), Dice + cross-entropy loss, batch size 1, mixed precision. Training stops when validation Dice has not improved by more than 0.003 for 8 epochs (not before epoch 10; at most 40 epochs for pancreas and 60 for hippocampus). The test split uses seed 12345 and a 15% test fraction; each training seed draws its own 15% validation set from the remaining cases.

## Reproducing

**Full pipeline.** Open the notebook in Google Colab with a GPU runtime (an L4 was used), clone this repository to `/content`, and run all cells. The MSD archives are downloaded automatically. The preprocessed pancreas cache is kept in `MyDrive/Datasets/PancreasCache_112/`; checkpoints, tables, figures and `results.json` go to `MyDrive/Outputs/Accuracy-Saturation/`. Re-running skips finished work.

**From the shipped tables.** Run from the repository root (Python 3 with numpy, pandas, scipy, matplotlib):

```bash
python analysis/stats.py      # Stage-1 summary and tests  -> s1_summary.csv, s1_tests.csv
python analysis/tables.py     # LaTeX tables               -> tab_stage1.tex, tab_stage2.tex, tab_perclass.tex
python analysis/datafig.py    # Stage-1 and Stage-2 figures
python analysis/schem.py      # schematic figures
```

`analysis/heavy.py` and `analysis/post.py` recompute the pooled reliability curves and the coherence diagnostic; they need the trained checkpoints (`RESULTS=<notebook output folder>`) and the pancreas cache (`CACHE=<folder of .npz files>`), and run on CPU.

## Data

Medical Segmentation Decathlon, Tasks 04 and 07 (CC BY-SA 4.0): http://medicaldecathlon.com/

## License

MIT (see `LICENSE`).
