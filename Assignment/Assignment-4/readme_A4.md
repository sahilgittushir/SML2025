# ⚡ Assignment 4: Advanced ML Algorithms  
**Statistical Machine Learning | Due: [DD/MM/YYYY] @ 11:59PM**  
*Implement AdaBoost, Gradient Boosting, and Neural Networks from scratch*

---

## 🎯 Objectives
1. **AdaBoost**: Binary classification on MNIST using decision stumps  
2. **Gradient Boosting**: Regression with squared/absolute loss  
3. **Neural Network**: Binary classifier with sigmoid activation  

---

## 📂 Repository Structure
```bash
A4/
├── src/
│   ├── adaboost/                # Q1
│   │   ├── stump.py            # Decision stump implementation
│   │   ├── boosting.py         # Weight update logic
│   │   └── mnist_pca.py        # PCA preprocessing (→5D)
│   ├── gradient_boosting/       # Q2
│   │   ├── losses.py           # Squared/Absolute loss
│   │   └── regression.py       # Sequential stump fitting
│   └── neural_net/             # Q3
│       ├── nn.py               # Forward/backward pass
│       └── synthetic_data.py   # Gaussian dataset gen
├── data/
│   ├── mnist_subset/           # Classes 0 and 1 only
│   └── synthetic_regression/   # sin(2πx)+cos(2πx)+noise
└── results/
    ├── Q1_adaboost/            # Accuracy vs rounds plot
    ├── Q2_gb/                  # Train/test loss curves
    └── Q3_nn/                  # Decision boundary plot
