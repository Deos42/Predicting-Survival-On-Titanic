# Predicting Survival on the Titanic using Logistic Regression from Scratch

A complete implementation of **Logistic Regression built entirely from first principles using NumPy**, without using scikit-learn’s LogisticRegression model.

This project demonstrates deep understanding of:

- Binary classification
- Convex optimization
- Gradient descent
- Cross-entropy loss
- Numerical stability techniques

---

## Problem Statement

Predict Titanic passenger survival using:

P(survived = 1 | age, sex, pclass)

Features:

- Age (continuous)
- Sex (encoded: male=0, female=1)
- Passenger Class (1,2,3)

---

## Implementation Details

### Model Formulation

Linear model:
z = wᵀx + b

Sigmoid activation:
σ(z) = 1 / (1 + e⁻ᶻ)

### Loss Function

Binary Cross-Entropy:

J(w,b) = -(1/m) Σ [ y log(ŷ) + (1-y) log(1-ŷ) ]

- Log clipping used for numerical stability
- Fully vectorized gradient computation

### Gradients

∂J/∂w = (1/m) Xᵀ(ŷ - y)  
∂J/∂b = (1/m) Σ (ŷ - y)

### Optimization

Parameters updated via gradient descent until convergence tolerance reached.

---

## Engineering Details

- Feature standardization for stable convergence
- Manual train/test split (80/20)
- Fully vectorized NumPy operations
- Explicit decision threshold at 0.5
- Probability curve visualization across subgroups

---

## Performance

| Metric               | Training   | Testing    |
| -------------------- | ---------- | ---------- |
| Accuracy             | **0.8002** | **0.7381** |
| Binary Cross-Entropy | 0.4513     | 0.5468     |

Model shows strong generalization with moderate overfitting.

---

## Observations

- Females have significantly higher survival probability.
- First-class passengers show higher survival likelihood.
- Survival probability decreases with age.
- Learned parameters reflect known historical patterns.

---

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn (data loading & splitting only)

