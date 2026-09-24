# XAScribe

XAScribe turns measured X-ray absorption near-edge spectra (XANES) into a quantitative, literature-grounded
analysis paragraph. It keeps two jobs apart:

1. **Quantitative layer.** Neural-network regressors on cumulative-distribution (CDF) features, trained on DFT
   spectra, predict the Ni oxidation state and the mean Ni–O distance. Each prediction comes with an uncertainty
   and with energy-resolved attributions computed through the CDF back to the measured energy axis.
2. **Literature layer.** An LLM agent builds a corpus from open scholarly databases (OpenAlex, Europe PMC, arXiv)
   and legally open full texts. A free open-weight LLM then writes a Results paragraph from the fixed numbers and
   the retrieved passages. Every number must come from layer 1 and every citation from a retrieved passage, and
   an automatic audit checks each claim.

Everything runs on free services: no paid API key is needed.

## Install

```bash
pip install -r requirements.txt        # Python >= 3.10
```

## Free LLM access

The default provider is OpenRouter's free tier: one free key, no credit card. Get a key at
https://openrouter.ai/keys and set it once:

```bash
setx OPENROUTER_API_KEY "sk-or-..."        # Windows (new terminals)
export OPENROUTER_API_KEY="sk-or-..."      # macOS/Linux
```

In the web app you can instead paste your key into the sidebar. It is used only for your session and never
stored. Each user needs their own free key; never commit a key to the repository or put one in a hosted demo.

By default XAScribe tries free open-weight models in order: GLM-5.2 (Z.ai), NVIDIA Nemotron 3 Ultra, Qwen 3.8, Gemma 4,
then OpenRouter's free router. Any OpenAI-compatible provider works:

| `XASCRIBE_LLM_PROVIDER` | key variable | notes |
|---|---|---|
| `openrouter` (default) | `OPENROUTER_API_KEY` | free models, automatic fallback |
| `groq` | `GROQ_API_KEY` | free tier, very fast |
| `cerebras` | `CEREBRAS_API_KEY` | free tier |
| `github` | `GITHUB_TOKEN` | GitHub Models, free with a GitHub account |
| `nvidia` | `NVIDIA_API_KEY` | build.nvidia.com free credits |
| `custom` | `XASCRIBE_LLM_API_KEY` + `XASCRIBE_LLM_BASE_URL` | any OpenAI-compatible endpoint |

To choose specific models, set `XASCRIBE_LLM_MODEL` (comma-separated, tried in order).

## Quick start

```bash
streamlit run app.py                                   # web app; bundled operando NMC622 example
python -m xascribe.corpus.build --element Ni --edge K \
    --material "layered NMC lithium-ion cathode" --target 25 \
    --exclude-doi 10.1021/acs.jpcc.0c08095 --out corpora/ni_k_nmc     # agentic corpus
python paper/reproduce_numbers.py                      # every headline number of the paper
python paper/case_study.py                             # full case study (needs a free LLM key)
```

When you analyse your own data, exclude the paper those spectra come from (`--exclude-doi`). Otherwise the
generator could cite the very study it is interpreting.

## Model card

- **Inputs.** Ni K-edge spectra on a 100-point grid spanning 8338.8–8378.8 eV (experimental energy, Ni-foil
  calibrated; model frame = experiment − 8.8 eV). Missing low-energy points are reconstructed from a conditional
  Gaussian estimated on simulations.
- **Targets.** Average Ni oxidation state; mean first-shell Ni–O distance (Å).
- **Training.**
  - 289 structure-averaged DFT spectra of NMC622/721/811 (FeatureXAS database).
  - 2 eV Gaussian broadening.
  - Ten nuisance variants per spectrum: scale, offset, ±1 eV shift and missing-point gaps.
  - One hidden layer of 32 tanh units per target, fitted with L-BFGS.
- **Accuracy (held-out computed spectra).** RMSE 0.060 for oxidation state and 0.0046 Å for Ni–O
  (R² = 0.981 and 0.986).
- **Operando NMC622 (Tallman et al. 2021).**
  - Oxidation state: RMSE 0.047 after calibration to the pristine electrode (2.67).
  - Ni–O: RMSE 0.013 Å, unadjusted.
  - The released network was chosen by its agreement with this series.
- **Limitations.**
  - Trained on Ni K-edge spectra of layered oxides only.
  - Needs spectra covering the full window.
  - Absolute valences require one reference spectrum of known valence.
  - Oxidation state and Ni–O distance are strongly correlated in the training data, so the two outputs are not
    independent measurements.
  - Transfer to other beamlines and setups must be verified.

## Data and licences

- **Code.** MIT licence (see `LICENSE`).
- **Computed spectra.** Public FeatureXAS dataset: Y. Chen et al., *Chem. Mater.* 2024, **36**, 2304; Zenodo
  record 10476278.
- **Operando spectra and reference curves.** Digitised from the published figures of K. R. Tallman et al.,
  *J. Phys. Chem. C* 2021, **125**, 58–73 (doi:10.1021/acs.jpcc.0c08095). They are redistributed as digitised
  points for reproducibility; raw scans are not included.
- **Valence reference.** Re-digitised from marker centroids; it reproduces the values stated in the source text.

## Citation

If you use XAScribe, please cite the accompanying paper (details to be added on publication).
