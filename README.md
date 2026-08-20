# README: Topological Encoder-Decoder Framework for Temporal Graph Learning

## Overview

Many real-world systems—such as transaction networks, citation graphs, and online communities—evolve dynamically over time. Traditional temporal graph learning methods often struggle when new nodes appear or existing nodes disappear (**node churn**), turning future prediction into a complex graph reconstruction problem.

This project introduces **TOPOGED** (Topological Graph Encoder-Decoder), a fully inductive encoder-decoder framework designed for discrete-time temporal graph forecasting. Rather than predicting edges directly over a fixed node set, TOPOGED frames temporal graph prediction as an **inverse topology problem**.

---

## What We Do & How We Do It

* **Multiscale Topological Encoding**: We summarize each graph snapshot using a degree-based filtration descriptor $\Phi(\mathcal{G}) = (X, Y)$ that records node and induced-edge counts across cumulative thresholds.


* **Budget & Probability Forecasting**: A lightweight temporal predictor forecasts the next snapshot's descriptor along with overall node and edge budgets, alongside the expected fraction of newly appearing nodes.


* **Arrival-Aware Node Memory**: A memory module uses recency, degree, and historical frequency to sample reappearing old nodes, while new nodes are instantiated based on arrival budgets.


* **Multi-Phase Edge Decoding**: Edges are constructed progressively across four distinct inductive categories:
1. **Old-Old Bank** ($\mathcal{E}^{oo-bank}$): Recurring interactions between previously seen old nodes.


2. **Old-Old Nobank** ($\mathcal{E}^{oo-nobank}$): Newly formed edges between old nodes.


3. **Old-New** ($\mathcal{E}^{on}$): Connections bridging existing nodes and newly arrived nodes.


4. **New-New** ($\mathcal{E}^{nn}$): Interactions occurring exclusively among new nodes.





---

## Probability Types & Distributions

To accurately distribute edge budgets across changing network environments, the framework models and projects empirical transition probabilities across the inductive edge categories ($\pi_t^{oo-bank}, \pi_t^{oo-nobank}, \pi_t^{on}, \pi_t^{nn}$). This ensures the model adapts dynamically during regime shifts instead of relying on stationary edge priors.

---

## Benchmarking

We rigorously evaluate performance across **14 temporal interaction datasets** (including College Message, MathOverflow, Reddit-Body, TGBL-Wiki, and 10 ERC20 token Ethereum transaction networks).

TOPOGED is benchmarked against state-of-the-art temporal graph models and dynamic network architectures, including:

* **ROLAND**

* **EvolveGCN**

* **VGRNN**

* **GC-LSTM**

* **HTGN**

* **TGCN**