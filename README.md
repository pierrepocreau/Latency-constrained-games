# Quantum Nonlocality under Latency Constraints

Code for the paper [**Quantum Nonlocality under Latency Constraints**](https://arxiv.org/abs/2510.26349) by Dawei Ding, Zhengfeng Ji, Pierre Pocreau, Mingze Xu, and Xinyu Xu.

## Overview

This repository implements optimization tools for analyzing quantum advantages in LC games. The code computes classical and quantum bounds on game values when communication between parties is restricted by physical latency limitations.

**Key features:**
- NPA hierarchy for upper bounds on quantum correlations
- Seesaw algorithm for optimizing quantum strategies on LC games
- Classical deterministic strategy optimization
- Analysis of different latency constraints and layout of parties (no communication, one-round, merged parties)

## Repository Structure

```
.
├── NPA/                         # NPA hierarchy implementation
│   ├── hierarchy.py             # SDP hierarchy for quantum bounds
│   ├── operator.py              # Measurement operator representation
│   └── canonicalOp.py           # Canonical monomial ordering
│
├── LC_seesaw/               # Seesaw optimization for quantum strategies
│   ├── seesaw.py                # Main seesaw algorithm
│   ├── quantumStrategy.py       # Quantum strategy with communication
│   ├── QCFO.py                  # Quantum Circuit with Fixed Order
│   ├── QCFO_utils.py            # Utility functions for QCFOs
│   └── data/                    # Storage for optimized strategies
│
├── latency_simulation.py        # Latency simulation (Tables 4-5)
├── plot.py                      # Latency plot (Fig. 16)
├── game.py                      # Definition of nonlocal games, and computation of classical value
├── random_XOR_game.py           # Random XOR game example (Table 2)
└── extended_XOR_games.py        # Extended XOR games analysis (Appendix C, Table 6)
```

## Installation

### Requirements

```bash
pip install numpy cvxpy networkx matplotlib scipy toqito dill pandas
```

**Required packages:**
- `numpy` - Numerical computing
- `cvxpy` - Convex optimization
- `networkx` - Graph structures for communication networks
- `matplotlib` - Plotting
- `toqito` - Quantum information toolkit
- `dill` - Serialization for saving strategies
- `scipy` - Scientific computing
- `pandas` - Data analysis

**Solvers:**
- MOSEK (required for the see-saw and G-signaling calculations) - Commercial solver with free academic licenses
- SCS - Open-source SDP solver supported by the NPA calculations

Install MOSEK following instructions at [mosek.com](https://www.mosek.com/downloads/)


Legacy strategy files can be loaded with
`from LC_seesaw.storage import load_strategy` and `load_strategy(path)`.
The original extended-XOR filenames use the old ten-bit encoding; for the
3-by-3 games, the effective function ID is the filename ID modulo 512.
The script draws fresh random functions; the legacy archive is not the exact
50-instance sample listed in the revised paper's Table 6.
