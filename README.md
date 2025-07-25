# Immune Algorithm for the Weighted Binary String Problem

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Frameworks](https://img.shields.io/badge/Frameworks-NumPy%20%7C%20Optuna-orange.svg)]() 

This project presents a highly customized implementation of the **CLONALG** immune algorithm, designed to efficiently solve the **Weighted Binary String (WBS)** optimization problem. The developed solution integrates several advanced heuristic mechanisms to achieve a robust balance between aggressive exploitation and strategic exploration, resulting in a 100% success rate across all benchmark instances.

## 🔎 Problem Formalization

The **Weighted Binary String (WBS)** problem is a combinatorial optimization benchmark designed to evaluate the performance of metaheuristics. It models scenarios where decision variables are binary but have varying levels of importance, reflected by continuous-valued weights.

Given a binary solution vector **x** of dimension *n*, `x = (x₁, x₂, ..., xₙ)` where `xᵢ ∈ {0, 1}`, and an associated vector of real-valued weights **w** = `(w₁, w₂, ..., wₙ)`, the objective is to find the binary string **x** that maximizes the weighted sum.

The objective function to be maximized is:

`f(x) = ∑ᵢ₌₁ⁿ wᵢxᵢ`

## ✨ Key Features

- **Adaptive Affinity-Proportional Hypermutation**: Utilizes a sophisticated non-linear mutation strategy where the mutation rate is inversely proportional to an antibody's fitness. Governed by an exponential decay function, this method aggressively protects elite solutions while promoting exploration in sub-optimal ones.
- **Elite-Focused Intensification Cloning**: A novel cloning strategy that provides a significant reproductive bonus to the single best antibody in the parent pool. This intensifies the search within the most promising region, accelerating the final convergence to the global optimum.
- **Unified Elitist Survival Strategy**: Employs a `(μ + λ)` survival mechanism where the parent population and all mutated clones compete together for survival. This enforces high selective pressure and guarantees monotonic improvement of the best-found solution.
- **Automated Hyperparameter Tuning**: Integrates **Optuna** for a robust, multi-objective hyperparameter search, enabling the systematic fine-tuning of the algorithm's parameters to maximize both performance and efficiency.
- **Comprehensive Analysis Suite**: Automatically generates a full suite of visualizations, including convergence plots, statistical summaries, and performance comparisons across all instances.

## 📂 Project Structure

The project is organized into a modular and clean structure to ensure clarity and maintainability.

```
Heuristics-MetaHeuristics/
├── src/
│   ├── algorithm/
│   │   ├── clonalg.py          # CLONALG immune algorithm implementation
│   │   └── config.py           # Hyperparameter configuration
│   ├── utils/
│   │   ├── input_reader.py     # WBS instance loading utility
│   │   └── visualization.py    # Complete visualization suite
│   ├── tuning/
│   │   └── tune_hyperparameters.py  # Optuna hyperparameter tuning script
│   └── main.py                 # Main experiment orchestrator
├── data/
│   ├── instance_0.csv         # Benchmark instances (1000-2600+ variables)
│   └── ... (instance_1-8.csv)
├── results/
│   ├── convergence_*.png      # Convergence analysis per instance
│   ├── detailed_runs/         # Detailed convergence of single runs
│   ├── results_summary.png    # Comparative performance analysis
│   └── pareto_front.html      # Multi-objective tuning analysis
├── requirements.txt           # Project dependencies
└── README.md                  # This documentation
```

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Alepassa/Heuristics-MetaHeuristics.git
    cd Heuristics-MetaHeuristics
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🛠️ Configuration

The algorithm's behavior is controlled by the `Config` class within `src/algorithm/config.py`. The key parameters for tuning are:

- **`POPULATION_SIZE`**: The number of antibodies in the population.
- **`SELECTION_RATIO`**: The percentage of the best antibodies to be selected as parents.
- **`CLONE_FACTOR`**: A base multiplier for determining the number of clones.
- **`MUTATION_DECAY_RATE`**: The key parameter (`ρ`) controlling the steepness of the affinity-proportional mutation.
- **`RANDOM_REPLACEMENT_RATIO`**: The percentage of the population replaced by new random individuals each generation, ensuring a baseline level of exploration.

Experiment-wide settings like `NUM_RUNS` and `MAX_EVALUATIONS` are also defined here.

## 🚀 Usage

The project provides two primary execution scripts located in the `src/` directory. **All commands should be executed from within the `src/` directory.**

### 1. Running Final Experiments (`main.py`)

This script executes the CLONALG algorithm with the fixed, optimized set of parameters defined in `config.py` to generate final performance reports.

#### Running on All Instances

To execute the full benchmark (100 runs per instance) on all instances. This is the standard mode for generating the final summary table and plots.

**Command:**
```bash
python3 src/main.py --all
```
*(This is also the default behavior if no arguments are provided.)*

#### Running on a Single Instance

Ideal for re-validating results or generating detailed plots for a specific case.

**Command Example:**
```bash
python3 src/main.py --instance instance_8
```

### 2. Hyperparameter Tuning (`tuning/tune_hyperparameters.py`)

This script uses **Optuna** to perform a robust, multi-objective search for the optimal set of hyperparameters. It is pre-configured to minimize both the **average gap** to the optimum and the **average number of evaluations**.

**To launch the tuning process:**
```bash
python3 src/tuning/tune_hyperparameters.py
```
The results, including an interactive Pareto front plot, will be saved in the `results/` directory.

---

## 📈 Results and Analysis

The final optimized configuration of the algorithm was subjected to a rigorous experimental validation, consisting of 100 independent runs for each of the 9 benchmark instances. The results demonstrate not only the complete effectiveness of the algorithm but also its remarkable consistency and efficiency.

### 1. Overall Performance

The primary goal of achieving a 100% success rate was met across all instances. The algorithm consistently identifies the known global optimum, regardless of the problem's complexity or the stochastic nature of the search.

| Instance | Success Rate | Avg. Fitness | Std. Dev. | Optimal | Avg. Iters | Avg. Evals | Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| instance_0 | 100.0% | 6817.00 | 0.00 | 6817 | 830.9 | 43242 | 4.84 |
| instance_1 | 100.0% | 7957.00 | 0.00 | 7957 | 984.4 | 51223 | 7.36 |
| instance_2 | 100.0% | 9059.00 | 0.00 | 9059 | 1145.8 | 59615 | 11.94 |
| instance_3 | 100.0% | 11108.00 | 0.00 | 11108 | 1277.8 | 66482 | 20.96 |
| instance_4 | 100.0% | 13341.00 | 0.00 | 13341 | 1462.3 | 76074 | 25.02 |
| instance_5 | 100.0% | 15309.00 | 0.00 | 15309 | 1611.9 | 83852 | 33.37 |
| instance_6 | 100.0% | 16961.00 | 0.00 | 16961 | 1756.0 | 91347 | 42.24 |
| instance_7 | 100.0% | 18008.00 | 0.00 | 18008 | 1959.7 | 101941 | 48.66 |
| instance_8 | 100.0% | 19249.00 | 0.00 | 19249 | 2165.3 | 112632 | 53.61 |

### 2. Scalability and Efficiency

Beyond mere correctness, a key quality indicator is how efficiently the algorithm solves problems of increasing size. Two metrics were analyzed: execution time and the number of fitness evaluations required.

#### Execution Time Scalability

The algorithm exhibits excellent scalability with respect to execution time. As shown in the chart below, the time required to complete 100 runs scales in a predictable, near-linear fashion with the problem size (string length). A linear trendline (`y = 2.63e-02x + 24.71`) fits the data with high accuracy, indicating that there is no exponential explosion in computational cost, making the approach viable for even larger-scale problems.

<img width="2394" height="2369" alt="results_summary_1" src="https://github.com/user-attachments/assets/03b1877d-938d-49a5-97aa-75b5698edab7" />

#### Search Efficiency Analysis

A more precise measure of search efficiency is the average number of fitness evaluations required to find the global optimum. The analysis shows that the algorithm consistently finds the solution well within the allocated budget of 150,000 evaluations.

<img width="2394" height="2369" alt="results_summary_2" src="https://github.com/user-attachments/assets/3d33f808-9305-4289-bcce-9558e444443e" />


### 3. Convergence Dynamics

To understand the internal behavior of the algorithm, the convergence trajectories of individual runs were analyzed. The chart below shows the detailed convergence paths for 5 independent runs on `instance_0`.

<img width="3043" height="2100" alt="detailed_convergence_instance_0" src="https://github.com/user-attachments/assets/33561e02-ee39-45b8-bffc-8b4789b5b9bd" />

