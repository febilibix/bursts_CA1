# Hippocampal LIF Simulations (1D & 2D)

This repository simulates CA1 pyramidal cell dynamics with two-compartment LIF neurons, CA3 inputs, and optional inhibitory plasticity. It includes utilities to generate trajectories, compute activation maps, and analyze correlations across environments (F1, F2, N1, F3, N2).
It contains the simulations and results presented in my thesis

## Features
To recreate figures, all simulation scripts in the simulations folder need to be run. 
They create files that then are plotted in the jupyter notebooks (sorted per figure in the thesis)

---

## Quickstart

### 1) Create and activate an environment
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
