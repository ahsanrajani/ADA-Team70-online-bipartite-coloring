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

This section contains the implementation of the algorithms and supporting utilities for the project.

---

## Code Structure Overview

### **1. Class: `BipartiteGraphColoring`**
This is the core class responsible for:
- Representing the bipartite graph.
- Implementing edge coloring algorithms.
- Validating the correctness of the coloring.
- Benchmarking performance across various scenarios.

#### **Key Methods:**
- `__init__(...)`: Initializes graph parameters, including nodes, palettes, and internal data structures.
- `color_graph()`: Executes the hybrid coloring algorithm for efficient edge coloring.
- `process_online_node(online_node)`: Dynamically processes edges as they arrive.
- `validate_coloring()`: Ensures the coloring adheres to edge coloring rules.
- `greedy_coloring_algorithm()`: Implements the standard greedy edge coloring algorithm.

---

### **2. Function: `main()`**
Automates testing of the algorithms on various bipartite graph structures.

#### **Test Cases:**
- Complete Bipartite Graphs (e.g., `K10,10`, `K100,100`).
- Star Topology.
- Sparse Graphs (e.g., Perfect Matching).
- Hybrid and Adversarial Graphs.

For each test:
- Runs both the hybrid and greedy algorithms.
- Logs results, including the number of colors used and execution time.

---

### **3. Output Handling**
- Results are saved to `output.txt` for analysis.
- Includes detailed logs of algorithm performance.

---

## How to Run
1. Ensure Python 3 is installed.
2. Save the script to a `.py` file.
3. Run the script:
    ```bash
    python your_script_name.py
    ```
4. Check `output.txt` for results.

### Documentation
- Algorithms from the research paper

### Reports
-Reports and presentations

### Research Material
- Primary and other relevant research papers, including detailed explanation of Contention Resolution Schemes used

## Team Members

- **Ali Ahsan Rajani**  
- **Hamza Raza**

---
