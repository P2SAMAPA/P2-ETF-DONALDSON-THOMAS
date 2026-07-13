# Donaldson-Thomas Invariants for ETFs

Counts "instantons" (coherent market structures) in a Calabi-Yau moduli space representation of the market. Donaldson-Thomas invariants are integer topological counts of stable configurations – a completely different way to think about market states. The per‑ETF score is the DT invariant count.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Calabi-Yau moduli space from return statistics and macro factors
- Stable configurations via k-means clustering
- DT invariant = number of stable clusters
- Score = DT invariant (higher = more market structure)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-donaldson-thomas-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High DT invariant → more coherent market structures → potential alpha.
- Low DT invariant → fragmented market structure.

## Requirements

See `requirements.txt`.
