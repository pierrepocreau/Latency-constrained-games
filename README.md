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
│   ├── NPAgame.py               # Nonlocal game definition
│   ├── hierarchy.py             # SDP hierarchy for quantum bounds
│   ├── operator.py              # Measurement operator representation
│   └── canonicalOp.py           # Canonical monomial ordering
│
├── NetworkSeesaw/               # Seesaw optimization for quantum strategies
│   ├── seesaw.py                # Main seesaw algorithm
│   ├── quantumStrategy.py       # Quantum strategy with communication
│   ├── QCFO.py                  # Quantum Circuit with Fixed Order
│   ├── QCFO_utils.py            # Utility functions for QCFOs
│   └── data/                    # Storage for optimized strategies
│
├── latency_plot.py              # Simulation and latency plot (Table 4 and Fig. 13)
├── cdnp.py                      # Classical deterministic strategies
├── random_XOR_game.py           # Random XOR game example (Table 2)
└── extended_XOR_games.py        # Extended XOR games analysis (Sec. 4.2.2, Table 6)
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
- MOSEK (recommended) - Commercial solver with free academic licenses
- SCS (fallback) - Open-source SDP solver

Install MOSEK following instructions at [mosek.com](https://www.mosek.com/downloads/)
