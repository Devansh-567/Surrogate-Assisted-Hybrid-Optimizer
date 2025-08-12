# EvoFusion Ultimate

**EvoFusion Ultimate** is a **hybrid, surrogate-assisted optimization framework** that fuses together multiple global optimization paradigms into one unified system. It is designed for **complex, mixed-variable, multi-fidelity, and constrained optimization problems** where evaluations are costly.

---

## 🚀 What Is EvoFusion Ultimate?

EvoFusion Ultimate is an **AI-assisted global optimizer** that:

- Maintains an evolving **population** of candidate solutions.
- Learns a **predictive model** (ensemble of GP, Random Forest, and MLP) of your objective function.
- Uses **CMA-ES-style correlated mutations** for efficient continuous search.
- Applies **Bayesian Optimization** with **batch Expected Improvement (q-EI)** to select the most promising candidates.
- Works with **continuous, integer, and categorical variables** in the same problem.
- Can handle **constraints** and **multi-fidelity** evaluations.

It blends:

- **Evolutionary Algorithms** (diverse exploration, crossover, mutation)
- **CMA-ES** (adaptive step-size, correlated mutations)
- **Bayesian Optimization** (probabilistic improvement targeting)
- **Ensemble Surrogate Modeling** (combining GP, RF, and MLP for robustness)

---

## 📌 Key Points

- **Hybrid Search Engine**: Combines the exploration power of evolutionary algorithms with the exploitation efficiency of Bayesian optimization.
- **Surrogate Ensemble**: Uses GP for smooth landscapes, RF for irregular/noisy landscapes, and MLP for highly nonlinear functions.
- **Mixed Variable Support**: Handles continuous, integer, and categorical inputs natively.
- **Multi-fidelity Optimization**: Supports different evaluation budgets/accuracies.
- **Batch Optimization**: Selects multiple promising candidates at each step using a Kriging Believer heuristic.
- **Constraint Handling**: User-defined feasibility checks or penalty functions.
- **Self-contained**: Implements a lightweight CMA-ES internally, no external CMA library required.
- **Smart Initialization**: Uses Latin Hypercube Sampling (LHS) for space-filling initial designs.

---

## 📂 Applications

EvoFusion Ultimate is ideal for:

- **Engineering design optimization** (e.g., aerodynamics, structural design)
- **Hyperparameter tuning** for machine learning models
- **Scientific experiments** with expensive simulations
- **Mixed-variable industrial process optimization**
- **Constrained optimization problems** with complex feasibility rules
- **Multi-fidelity simulations** where coarse and fine evaluations exist

---

## 🔍 How the Algorithm Works

1. **Initialization**

   - Sample initial solutions using Latin Hypercube Sampling (LHS).
   - Evaluate the objective function (possibly at the lowest fidelity).

2. **Surrogate Model Training**

   - Fit an ensemble of GP, RF, and MLP on all evaluated data.
   - Compute model weights based on training error.

3. **Candidate Generation**

   - Use multiple strategies:
     - LHS exploration
     - CMA-ES continuous search
     - Evolutionary crossover + mutation
     - Local jitter around top performers

4. **Acquisition Scoring**

   - Predict mean and uncertainty for candidates.
   - Use **Expected Improvement (EI)** or **Upper Confidence Bound (UCB)** as the acquisition function.

5. **Batch Selection**

   - Select a batch of candidates using the **Kriging Believer** method.

6. **Evaluation**

   - Evaluate selected candidates on the objective function.
   - Record results in history.

7. **Population Update**

   - Keep elites, high-prediction candidates, and mutated offspring.
   - Update CMA-ES parameters with top solutions.

8. **Repeat** until generation limit or early stopping.

---

## ⚙️ Installation

Clone the repository from GitHub:
git clone https://github.com/Devansh-567/evofusion-ultimate.git
pip install -r requirements.txt
python evo.py

---

## 🛠 Usage Example

```python
Example optimization on the **Ackley function**:

from evofusion import Space, EvoFusionUltimate
import numpy as np

# Define objective (maximize negative Ackley)

def ackley_obj(x, fidelity=0):
x0, x1 = x
val = -20 * np.exp(-0.2*np.sqrt(0.5\*(x0**2 + x1**2))) \

- np.exp(0.5*(np.cos(2*np.pi*x0)+np.cos(2*np.pi\*x1))) \

* np.e + 20
  return -val

# Search space

variables = [
{'name':'x','type':'continuous','bounds':(-5,5)},
{'name':'y','type':'continuous','bounds':(-5,5)}
]
space = Space(variables)

# Create optimizer

ef = EvoFusionUltimate(
space=space,
obj_func=ackley_obj,
pop_size=24,
generations=60,
init_samples=80,
batch_size=6,
seed=42,
verbose=True
)

# Run optimization

best, best_y = ef.run()
print("Best result:", best_y, best['x_decoded'])
```

---

## 📊 Results (Ackley Example)

Example run output:
Gen 001 | Evaluations 280 | GenBest -0.325804 | Best -0.325804
Gen 002 | Evaluations 481 | GenBest -0.022616 | Best -0.022616
...
Converges to near-optimum within a few generations.

---

## 💡 Why This Is an Invention

EvoFusion Ultimate is not just a standard optimizer:

- It **merges three major optimization paradigms** (EA, CMA-ES, BO) in a _single, coordinated_ framework.
- Uses **model stacking** for surrogates — leveraging strengths of GP, RF, and MLP simultaneously.
- Supports **mixed-type** and **multi-fidelity** problems out-of-the-box.
- Implements **lightweight CMA-ES** internally, no dependency on large external packages.
- Designed for **parallel, batch-oriented optimization**.

This unique combination makes it more **flexible, adaptable, and robust** than typical single-method optimizers.

---

## 📥 Cloning From GitHub

To copy this repository to your machine:
git clone https://github.com/Devansh-567/evofusion-ultimate.git
pip install -r requirements.txt
python evo.py

---

## 📜 License

MIT License — feel free to use, modify, and distribute with attribution.

---

## 👤 Author

**Devansh Singh**
GitHub: Devansh-567
Email: devansh.jay.singh@gmail.com
