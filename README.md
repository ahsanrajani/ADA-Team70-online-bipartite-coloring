# Deterministic Online Bipartite Edge Coloring

This repository contains our project for **CS/CE 412/471 – Spring 2025**.

We are implementing and analyzing the algorithm from the paper:

**Title:** Deterministic Online Bipartite Edge Coloring  
**Authors:** Joakim Blikstad, Ola Svensson, Radu Vintan, David Wajc  
**Link:** [http://arxiv.org/abs/2408.03661v2](http://arxiv.org/abs/2408.03661v2)

---

## Project Summary

The goal is to explore a deterministic algorithm that outperforms the greedy method for online edge coloring in bipartite graphs. This algorithm achieves a competitive ratio better than 2 for large maximum degrees.

---

## Repository Contents

### Code
- **`deterministic_coloring.py`**  
    Our CRS-based implementation of the online bipartite edge-coloring algorithm from Blikstad et al.
- **`greedy_coloring.py`**  
    A simple baseline: assigns each new edge the smallest available color.
- **`test_coloring_cases.py`**  
    A suite of six canonical tests (complete bipartite graphs, paths, stars, random bipartite graphs) that runs both algorithms side-by-side, validates each coloring, measures color usage and runtime, and writes everything to `test_output.txt`.


To run the code:

1. Uncomment the desired test cases in `test_coloring_cases.py`.
2. Run the file. The output will be stored in `test_output.txt` in the main repository.

For visualization:
1. Run the test cases to generate the output in `test_output.txt`.
2. Use `animation.py` in the `visualization` folder to visualize the results.

### Documentation
- Research notes and summaries.

### Reports
- Proposal reports, presentations, and other deliverables.

### Visualization
- Animations of the main test cases.
- Use `animation.py` to visualize additional test cases.

---

## Team Members

- **Ali**  
- **Hamza**

---
