# 📐 Assignment 1: Theoretical Foundations in SML  
**Statistical Machine Learning (Spring 2025)**  

---

## 📝 Problem Summary  
### **Q1: Decision Boundary under Linear Transformation**  
**Task**: Derive the decision boundary `g₁(Y) = g₂(Y)` for multivariate Gaussian classes (`ω₁`, `ω₂`) after linear transformation `Y = AX + b`.  
**Key Concepts**: Bayes classifier, quadratic forms, covariance propagation.  
**Expected Output**: Closed-form equation of the boundary in terms of `A`, `b`, `μᵢ`, `Σᵢ`.  

### **Q2: Poisson Classification**  
**Part (a)-(b)**: Prove `E[x] = λ` and `Var[x] = λ` for Poisson RV.  
**Part (c)**: Derive Bayes decision rule for `λ₁ > λ₂` with equal priors.  
**Part (e)**: Compute Bayes error rate with integer rounding of boundary.  
**Expected Output**:  
- Proofs for mean/variance (LaTeX).  
- Decision rule: `Classify as ω₁ if x ≥ ⌊λ*⌋` where `λ*` is derived.  

### **Q3: Minimum Risk Discriminant**  
**Task**: Show discriminant `g(x) = wᵀx + w₀` minimizes risk given loss constraints `λ₂₁ > λ₁₁`, `λ₁₂ > λ₂₂`.  
**Expected Output**: Full derivation of `w₀` including prior and loss terms.  

### **Q4: Bayes Risk Minimization**  
**Task**: Find optimal decision boundary `x*` minimizing expected risk for arbitrary PDFs under asymmetric loss matrix `Λ`.  
**Expected Output**: Boundary condition involving `p(x|ωᵢ)`, `P(ωᵢ)`, and `λᵢⱼ`.  

---

## 💻 Technical Notes  
- **Format**: Submit PDF or neatly handwritten scans.  
- **Allowed References**: Lecture notes, textbooks. No collaboration.  
- **Critical Steps**:  
  - Q1: Handle singular `Σᵢ` cases.  
  - Q2(e): Round boundary to floor integer.  
  - Q4: Express `x*` in terms of likelihood ratios and loss weights.  

---

