# 🧠 Assignment 3: Core ML Algorithms from Scratch
**Statistical Machine Learning | Due: [DD/MM/YYYY] @ 11:59PM**  
*Implement decision trees, regression analysis, and ensemble methods without sklearn*

---

## 🎯 Objectives
1. **Optimal Predictor**: Derive the true risk-minimizing function 𝑓*(𝑥)
2. **Bias-Variance Tradeoff**: Analyze linear regression models
3. **Decision Tree**: Implement with Gini impurity and pruning
4. **Ensemble Methods**: Bagging with out-of-bag (OOB) error evaluation
5. **Polynomial Regression**: 5-fold cross-validation on synthetic data

---

## 📂 Repository Structure
```bash
A3/
├── src/
│   ├── q1_optimal_predictor.py       # Derives 𝑓*(𝑥) = E[Y|X=x]
│   ├── q2_bias_variance.py           # Computes bias/variance at x=2
│   ├── q3_decision_tree/             # From-scratch implementation
│   │   ├── tree.py                  # Node class + splitting logic
│   │   └── train.py                 # Handles stopping conditions
│   ├── q4_bagging.py                # 10-tree ensemble with OOB
│   └── q5_polynomial_regression/     # Cross-validation pipeline
├── data/
│   ├── synthetic_regression.csv      # Q2 datasets (D1-D3)
│   └── decision_tree.csv            # Q3 training data (Table 1)
└── results/
    ├── decision_boundaries/         # Visualizations for Q1-Q2
    ├── tree_structure.txt           # Q3 printed tree
    └── cv_results.png               # Q5 best polynomial fit
