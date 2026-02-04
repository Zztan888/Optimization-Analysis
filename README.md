# 📉 Mathematical Optimization Analysis

This repository contains the implementation and analysis of optimization algorithms to solve complex mathematical functions. This project was completed for **Assignment 3**.

---

## 📑 Project Components
* **Algorithm:** Optimization (e.g., Gradient Descent / Evolutionary Algorithms)
* **Language:** Python
* **Source Code:** `Optimization.py`
* **Documentation:** `Report_Assignment3.pdf`

---

## 📂 Quick Links
* 📜 **Source Code:** [Optimization.py](./Optimization.py)
* 📄 **Technical Report:** [Report_Assignment3.pdf](./Report_Assignment3-git.pdf)

---

## 🤖 Introduction to Optimization
Optimization is the process of finding the best solution from all feasible solutions. In this project, we focus on minimizing a cost function $f(x)$ to reach the global optimum.



### Key Concepts:
1. **Objective Function:** The function we want to minimize or maximize.
2. **Search Space:** The range of possible values for the variables.
3. **Convergence:** The point where the algorithm settles on a final value.

---

## 💻 Implementation & Logic
The script `Optimization.py` performs the following steps:
1. **Function Definition:** Defining the mathematical landscape (e.g., Sphere, Rosenbrock, or Rastrigin functions).
2. **Initialization:** Starting the search at a random or specific point in the search space.
3. **Iterative Update:** Applying the optimization rule to move toward the optimum.
   - For Gradient Descent, the update rule is: 
     $$x_{new} = x_{old} - \alpha \cdot \nabla f(x_{old})$$
     *(where $\alpha$ is the learning rate)*.
4. **Result Logging:** Tracking the "Fitness" or "Loss" over time.



---

## 📈 Results & Screenshots
The findings are documented in detail within the [Assignment Report](./Report_Assignment3.pdf).
* **Performance:** Analysis of how quickly the algorithm converged.
* **Visualizations:** The report includes convergence curves and final 3D plots of the optimization path.

---

## 🚀 How to Run
1. Ensure you have Python installed.
2. Install necessary libraries:
   ```bash
   pip install numpy matplotlib
