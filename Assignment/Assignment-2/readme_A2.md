# 🔍 Assignment 2: MNIST Classification  
**Statistical Machine Learning (Spring 2025)**   

---

## 🎯 Objective  
Implement and compare four classification methods on MNIST:  
1. **Maximum Likelihood Estimation (MLE)**  
2. **Principal Component Analysis (PCA) + Classifier**  
3. **Fisher’s Discriminant Analysis (FDA)**  
4. **Discriminant Analysis (e.g., LDA/QDA)**  

---

## 📂 File Structure  
```bash
A2/  
├── data/                    # MNIST dataset (preloaded or download script)  
├── src/  
│   ├── mle_classifier.py    # MLE implementation  
│   ├── pca_framework.py     # PCA dimensionality reduction  
│   ├── fda.py               # FDA for classification  
│ 
├── results/  
│   ├── accuracy_report.txt  # Performance comparison  
│   ├── pca_variance.png     # Scree plot  
│   └── confusion_matrices/  # Per-method confusion matrices  
└── report.pdf               # Analysis of results  
