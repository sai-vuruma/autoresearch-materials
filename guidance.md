## Task Description

---

You are designing an approach to tackle Out-of-Distribution (OOD) regression. This means the test data will likely contain samples that are considerably different with respect to the training data distribution.

## Key Assumptions

---

- Train and test sets are fixed before any model training begins.
- No oracle is queried during or after training.
- Test labels are never seen during training or calibration.
- The primary objective is **accurate, well-calibrated prediction on OOD test samples**.

## Benchmark Protocol

---

Use the following shared setup to evaluate and compare all approaches head-to-head on a fixed OOD split:

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def make_ood_split(X, y, train_frac=0.8):
    """Sort by y — bottom train_frac for training, top (1-train_frac) for OOD test."""
    idx = np.argsort(y)
    n_train = int(len(y) * train_frac)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

# Standardize X using training stats only. Never refit scaler on test data.
```

**Key diagnostic metric — Extrapolation Slope**: Beyond RMSE and R², compute the slope of `predicted y` vs `true y` on the test set via `np.polyfit(y_test, pred, 1)[0]`. A saturated MLP gives slope ≈ 0 (predicts near `max(y_train)` regardless of true value). A genuinely extrapolating model gives slope ≈ 1. This is the most diagnostic single number for OOD regression.

**Expected performance order on standard tabular benchmarks**:
`GP_linear_RBF > Delta_Ridge_MLP > NAM+PySR > Extrap_Mixup > DKL > Anchor_points > Monotonic_MLP > Gated_ExtrapMLP > NALU > NAM > MLP_log_y > MLP_baseline`

---

## Approaches to Consider

---

## 1. Group DRO — Distributionally Robust Optimization

### Concept
**Group DRO** replaces the standard average-loss training objective with one that minimizes the loss of the **worst-performing subgroup** in the training data. Each subgroup (environment) corresponds to a distinct data partition — e.g., crystal family, synthesis route, source database, or measurement instrument. By never allowing any group's loss to be neglected during training, the model is forced to rely only on features that work universally across all groups — precisely the causally grounded, spurious-correlation-free features that generalize to OOD test sets.

### Implementation Instructions
1. **Data Requirement**: Each training sample must have an environment/group label `g ∈ {0, 1, ..., G-1}`. Partition by any known covariate: source lab, material family, DFT functional used, synthesis temperature bin, etc.
2. **Group DRO Loss**:
   ```python
   def group_dro_loss(predictions, targets, group_ids, n_groups, eta=0.1):
       group_losses = torch.zeros(n_groups, device=predictions.device)
       for g in range(n_groups):
           mask = group_ids == g
           if mask.sum() > 0:
               group_losses[g] = F.mse_loss(predictions[mask], targets[mask])
       # Exponentiated gradient ascent over group weights
       group_weights = F.softmax(group_losses / eta, dim=0).detach()
       return (group_weights * group_losses).sum()
   ```
3. **Training Loop**: Drop-in replace `F.mse_loss(...)` with `group_dro_loss(...)`. Use `Adam` or `SGD`. No other architectural changes needed.
4. **Hyperparameter `eta`**: Controls aggressiveness of worst-group up-weighting. Start with `eta=0.1`. Reduce if training is unstable; increase if worst-group loss is not decreasing.
5. **No Group Labels Available**: Use **JTT** (Just Train Twice) — train a standard ERM model first, identify high-loss training samples, up-weight them in a second training pass as a proxy for hard/OOD groups.
6. **Library Option**: `pip install wilds` — provides a Group DRO trainer and OOD benchmarks with environment labels out of the box.

---

## 2. Invariant Risk Minimization — IRM / REx

### Concept
**IRM** learns a representation Φ(x) such that the optimal predictor on top of that representation is *identical across all training environments*. It penalizes the gradient of the per-environment loss with respect to a fixed scalar w=1.0, forcing the model to find features where no single environment's spurious signal is privileged.

**REx** (Risk Extrapolation) is a simpler, more numerically stable alternative for regression: instead of gradient norm penalties, it minimizes the **variance of per-environment losses** `Var({R_e})`. Both require environment labels but make no assumptions about the causal graph structure.

### Implementation Instructions
1. **Environment Labels**: Partition training data into ≥ 2 environments. Store as integer `env_id` per sample.
2. **IRMv1 Penalty**:
   ```python
   dummy_w = torch.tensor(1.0, requires_grad=True)

   def irm_penalty(loss_per_env):
       penalty = 0
       for loss in loss_per_env:
           grad = torch.autograd.grad(loss * dummy_w, dummy_w, create_graph=True)[0]
           penalty += grad.pow(2).sum()
       return penalty

   total_loss = sum(loss_per_env) + lambda_irm * irm_penalty(loss_per_env)
   ```
3. **REx Loss** (recommended for regression — more stable than IRMv1):
   ```python
   mean_loss = torch.stack(loss_per_env).mean()
   rex_penalty = torch.stack([(l - mean_loss)**2 for l in loss_per_env]).mean()
   total_loss = mean_loss + lambda_rex * rex_penalty
   ```
4. **Training Schedule**: Warm up with `lambda=0` (pure ERM) for ~10 epochs, then linearly anneal lambda to its target value (e.g., `1e3` for IRM, `1.0` for REx) to prevent early training collapse.
5. **Validation**: Use OOD validation performance — not in-distribution validation — for model selection and early stopping. In-distribution validation loss is misleading for IRM/REx-trained models.
6. **Library**: Reference implementation at `github.com/facebookresearch/InvariancePrinciples`. Also available via `wilds`.

---

## 3. Weighted Conformal Prediction — Post-Hoc Coverage Guarantee

### Concept
**Conformal Prediction** is a post-hoc, model-agnostic wrapper that converts any point-prediction regression model into one with **statistically guaranteed prediction intervals**. Under covariate shift (OOD test inputs), standard CP breaks its coverage guarantee. **Weighted Conformal Prediction** restores this by reweighting calibration samples using estimated likelihood ratios `w(x) = p_test(x) / p_train(x)`, giving valid marginal coverage on OOD test samples without retraining the base model.

### Implementation Instructions
1. **Base Model**: Train any regression model on source training data (GNN, MLP, Random Forest, etc.).
2. **Calibration Split**: Hold out 15–20% of training data as a calibration set `{(x_i, y_i)}` — never used for model training.
3. **Nonconformity Scores**: Compute residuals on calibration set: `s_i = |y_i - ŷ_i|`.
4. **Likelihood Ratio Weights** (for covariate shift correction):
   ```python
   from sklearn.ensemble import GradientBoostingClassifier
   # Train domain classifier: calibration (0) vs test (1)
   X_all = np.vstack([X_calibration, X_test])
   y_domain = np.array([0]*len(X_calibration) + [1]*len(X_test))
   clf = GradientBoostingClassifier().fit(X_all, y_domain)
   p_test = clf.predict_proba(X_calibration)[:, 1]
   p_cal  = clf.predict_proba(X_calibration)[:, 0]
   weights = p_test / (p_cal + 1e-8)   # likelihood ratio w(x_i)
   ```
5. **Weighted Quantile**: Compute the weighted `(1-α)`-quantile of nonconformity scores to get threshold `q̂`.
6. **Prediction Interval**: For any test sample: `[ŷ - q̂, ŷ + q̂]`. Guaranteed to have coverage ≥ `1-α` in expectation.
7. **Library**: `pip install mapie` — `MapieRegressor` supports weighted CP directly:
   ```python
   from mapie.regression import MapieRegressor
   mapie = MapieRegressor(estimator=base_model, method='plus', cv='split')
   mapie.fit(X_train, y_train)
   y_pred, y_pi = mapie.predict(X_test, alpha=0.1)
   ```

---

## 4. Spectral-Normalized Neural GP (SNGP) — Distance-Aware Uncertainty

### Concept
Standard neural networks collapse OOD inputs into the same latent region as in-distribution inputs, causing overconfident predictions on novel test samples. **SNGP** enforces a **bi-Lipschitz constraint** on all hidden layers via spectral normalization, ensuring geometrically distant inputs remain distant in latent space. The deterministic regression head is replaced with a **Laplace-approximated Gaussian Process** using random Fourier features, yielding distance-aware uncertainty that inflates naturally as test inputs deviate from the training manifold.

### Implementation Instructions
1. **Spectral Normalization**: Wrap every hidden `nn.Linear` layer (not the final head):
   ```python
   from torch.nn.utils import spectral_norm
   layer = spectral_norm(nn.Linear(in_dim, out_dim))
   ```
2. **Random Feature GP Head**:
   ```python
   class RandomFeatureGPHead(nn.Module):
       def __init__(self, in_dim, num_features=512):
           super().__init__()
           self.W = nn.Parameter(torch.randn(in_dim, num_features), requires_grad=False)
           self.beta = nn.Parameter(torch.rand(num_features) * 2 * torch.pi, requires_grad=False)
           self.output_layer = nn.Linear(num_features, 1)
           self.precision = torch.eye(num_features)   # updated post-training

       def forward(self, x):
           phi = torch.cos(x @ self.W + self.beta) * (2 / self.W.shape[1])**0.5
           return self.output_layer(phi), phi
   ```
3. **Precision Matrix Update** (after training, one pass over train set):
   ```python
   precision = torch.eye(num_features)
   for x_batch, _ in train_loader:
       _, phi = model.gp_head(model.encoder(x_batch))
       precision += phi.T @ phi
   ```
4. **Predictive Variance at Test Time**:
   ```python
   _, phi = model.gp_head(model.encoder(x_test))
   variance = phi @ torch.linalg.inv(precision) @ phi.T
   epistemic_std = variance.diag().sqrt()
   ```
5. **Evaluation**: Report both RMSE and the correlation between `epistemic_std` and absolute prediction error — a well-calibrated SNGP should show strong positive correlation on OOD test samples.

---

## 5. Test-Time Training via Self-Supervised Auxiliary Tasks (TTT)

### Concept
**TTT** adapts the feature extractor at inference time using the *unlabeled test samples themselves*, with no labels required. During source training, the model is jointly optimized on the main regression task and an auxiliary self-supervised task (e.g., masked feature reconstruction). At test time, a brief gradient update on the auxiliary loss on each OOD test sample adjusts the feature extractor to accommodate the novel structure before the frozen regression head makes its prediction.

### Implementation Instructions
1. **Architecture**:
   ```python
   class TTTModel(nn.Module):
       def __init__(self, encoder, reg_head, ssl_head):
           super().__init__()
           self.encoder = encoder      # shared; adapted at test time
           self.reg_head = reg_head    # frozen at test time
           self.ssl_head = ssl_head    # auxiliary task head; updated at test time
   ```
2. **Source Training**: Jointly optimize `L_total = L_regression + α * L_ssl` on training data. Use `α=0.5` as a starting point.
3. **Auxiliary Task Options**:
   - **Tabular/vector data**: Mask 30% of input features randomly; predict masked values (denoising).
   - **Graph data (GNN)**: Mask random atom nodes; predict atom identity (cross-entropy).
   - **General**: Contrastive loss between two augmented views of the same input (SimCLR-style).
4. **Test-Time Adaptation** (run once on all test data before final prediction):
   ```python
   from copy import deepcopy
   encoder_snapshot = deepcopy(model.encoder.state_dict())

   model.encoder.train(); model.ssl_head.train()
   model.reg_head.eval()
   optimizer_ttt = Adam(
       list(model.encoder.parameters()) + list(model.ssl_head.parameters()), lr=1e-4)

   for step in range(50):
       ssl_loss = compute_ssl_loss(model, X_test_all)
       optimizer_ttt.zero_grad(); ssl_loss.backward(); optimizer_ttt.step()

   model.encoder.eval()
   with torch.no_grad():
       y_pred = model.reg_head(model.encoder(X_test_all))

   model.encoder.load_state_dict(encoder_snapshot)  # restore
   ```
5. **Batch vs. Sample-Level TTT**: Adapting on all test samples jointly (batch TTT) is preferred when the test set is large and homogeneous. Adapt per-sample only if test samples are from very different sub-distributions.

---

## 6. Hybrid Latent-TabPFN (In-Context Bayesian Inference)

### Concept
**TabPFN** and **TabICL** perform regression via *in-context learning* (ICL): the training set (support) and test set (query) are passed in a single forward pass with **no gradient updates**. TabPFN was pre-trained on millions of synthetic datasets sampled from diverse Bayesian priors, mathematically approximating **optimal Bayesian posterior predictive inference** over the support set. The hybrid variant prepends a frozen domain-specific encoder to compress high-dimensional inputs into a low-dimensional latent vector.

### Implementation Instructions
1. **Encoder**: Load a pre-trained domain encoder. Freeze all weights: `param.requires_grad = False`. Extract latent vectors of dimension ≤ 100 for all train and test samples using `torch.no_grad()`.
2. **Dimensionality Check**: If latent dim > 100, apply `PCA(n_components=64)` fit **only on training embeddings**, then transform test embeddings with the same fitted PCA. Never refit PCA on test data.
3. **TabPFN Install**: `pip install tabpfn`.
4. **Fit & Predict**:
   ```python
   from tabpfn import TabPFNRegressor
   model = TabPFNRegressor(device='cuda', N_ensemble_configurations=16)
   model.fit(X_train_latent, y_train)
   y_pred, y_std = model.predict(X_test_latent, return_std=True)
   ```
5. **Uncertainty Output**: `y_std` gives predictive standard deviation per test sample. High values indicate OOD test samples.
6. **Sample Limit**: TabPFN degrades above ~1000 training samples. If training set is larger, subsample a representative 512–1000 point support set via k-means centroids on the latent space.

---

## 7. Bayesian Last-Layer with Gaussian Process Head (BLL-GP)

### Concept
Freeze a deep feature extractor pre-trained on source data. Discard the deterministic regression head and replace it with an **exact Gaussian Process** fit on the training set in the encoder's latent space. The GP inherits the source model's structural understanding of the data while providing exact Bayesian uncertainty quantification.

### Implementation Instructions
1. **Feature Extraction**: Load pre-trained source model. Register a forward hook on the penultimate layer to extract embeddings. Run all train and test samples through under `torch.no_grad()`. Store `Z_train`, `Z_test`.
2. **GP Fit on Training Latents**:
   ```python
   from sklearn.gaussian_process import GaussianProcessRegressor
   from sklearn.gaussian_process.kernels import Matern, WhiteKernel
   kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-3)
   gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
   gp.fit(Z_train, y_train)
   y_pred, y_std = gp.predict(Z_test, return_std=True)
   ```
3. **Kernel Selection**: `Matern(nu=2.5)` is the robust default. For normalized embeddings, also try `DotProduct` kernel. Compare marginal log-likelihood via `gp.log_marginal_likelihood_value_`.
4. **Scalability**: Exact GP scales as O(N³). If training set exceeds ~2000 samples, switch to `GPyTorch`'s `ExactGP` with CG solvers, or use SVGP (Approach 10).
5. **OOD Detection**: `y_std` on the fixed test set directly identifies which test predictions are uncertain. Report alongside predictions as a reliability flag.

---

## 8. Model-Agnostic Meta-Learning — MAML / ANIL (Few-Shot Adaptation)

### Concept
**MAML** optimizes model initialization weights θ* such that a single gradient step on a new task's support set yields maximal performance. The training set is structured into synthetic few-shot tasks by environment, and the few labeled OOD test samples (if any) form the adaptation support set. **ANIL** reduces cost by only updating the final regression head during inner-loop adaptation.

### Implementation Instructions
1. **Library**: `pip install learn2learn`.
2. **Task Construction**: Partition source training data by environment label. Each meta-training episode: sample one environment as support (K=5–20 samples), another split as query.
3. **Inner Loop**:
   ```python
   import learn2learn as l2l
   maml = l2l.algorithms.MAML(model, lr=0.01, first_order=True)
   learner = maml.clone()
   support_loss = F.mse_loss(learner(X_support), y_support)
   learner.adapt(support_loss)
   ```
4. **Outer Loop**:
   ```python
   query_loss = F.mse_loss(learner(X_query), y_query)
   meta_optimizer.zero_grad(); query_loss.backward(); meta_optimizer.step()
   ```
5. **Fixed Test Deployment**: If a small labeled support set exists in the OOD test domain, clone the meta-model and run 1–5 inner-loop gradient steps before predicting. If no OOD labels are available, the meta-initialization serves as the zero-shot base model.
6. **ANIL Variant**: Freeze all layers except the final linear head during the inner loop. Reduces meta-training compute by ~60%.

---

## 9. Evidential Deep Learning for Regression (EDL / NIG)

### Concept
**EDL** trains a single deterministic network to output the four parameters of a **Normal-Inverse-Gamma (NIG) distribution**, simultaneously yielding: (1) the predicted mean, (2) aleatoric uncertainty (irreducible data noise), and (3) epistemic uncertainty (model ignorance, which inflates on OOD test samples). No ensembles, no MC Dropout, no sampling needed at inference.

### Implementation Instructions
1. **Output Layer**: Replace final `nn.Linear(..., 1)` with `nn.Linear(..., 4)`:
   ```python
   def evidential_output(raw):
       gamma = raw[:, 0:1]
       nu    = F.softplus(raw[:, 1:2]) + 1e-6
       alpha = F.softplus(raw[:, 2:3]) + 1.0   # must be > 1
       beta  = F.softplus(raw[:, 3:4]) + 1e-6
       return gamma, nu, alpha, beta
   ```
2. **NIG Loss Function**:
   ```python
   def nig_loss(y, gamma, nu, alpha, beta, lambda_reg=0.01):
       omega = 2 * beta * (1 + nu)
       nll = (0.5 * torch.log(torch.pi / nu)
              - alpha * torch.log(omega)
              + (alpha + 0.5) * torch.log(nu * (y - gamma)**2 + omega)
              + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))
       reg = torch.abs(y - gamma) * (2 * nu + alpha)
       return (nll + lambda_reg * reg).mean()
   ```
3. **Uncertainty at Inference**:
   ```python
   epistemic_var = beta / (nu * (alpha - 1))
   aleatoric_var = beta / (alpha - 1)
   ```
4. **Evaluation**: Compute Spearman rank correlation between `epistemic_std` and absolute test errors. A well-calibrated EDL model shows strong positive correlation on OOD test samples.
5. **Hyperparameter**: Tune `lambda_reg ∈ [0.001, 0.1]`. Too low → overconfident OOD. Too high → underfits regression.
6. **Library**: Reference at `github.com/aamini/evidential-deep-learning`.

---

## 10. Sparse Variational GP (SVGP) — Scalable Non-Parametric Regression

### Concept
A plain **Sparse Variational GP** (without online updates) serves as a scalable, uncertainty-aware non-parametric regression model when the training set is too large for exact GPs (> ~2000 samples). SVGP approximates the full GP posterior using M ≪ N learned *inducing points*, yielding O(NM²) complexity.

### Implementation Instructions
1. **Library**: `pip install gpytorch`.
2. **Model Definition**:
   ```python
   import gpytorch, torch

   class SVGPModel(gpytorch.models.ApproximateGP):
       def __init__(self, inducing_points):
           variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
               inducing_points.size(0))
           variational_strategy = gpytorch.variational.VariationalStrategy(
               self, inducing_points, variational_distribution, learn_inducing_locations=True)
           super().__init__(variational_strategy)
           self.mean_module = gpytorch.means.ConstantMean()
           self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

       def forward(self, x):
           return gpytorch.distributions.MultivariateNormal(
               self.mean_module(x), self.covar_module(x))
   ```
3. **Inducing Points Initialization**: Sample M=100–500 points via k-means clustering on input features.
4. **Training**:
   ```python
   likelihood = gpytorch.likelihoods.GaussianLikelihood()
   mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(y_train))
   optimizer = torch.optim.Adam([{'params': model.parameters()},
                                  {'params': likelihood.parameters()}], lr=0.01)
   for epoch in range(200):
       for X_batch, y_batch in train_loader:
           optimizer.zero_grad()
           loss = -mll(model(X_batch), y_batch)
           loss.backward(); optimizer.step()
   ```
5. **Deep Kernel Option**: Replace `MaternKernel` with a learned neural net backbone for higher expressivity on structured inputs.

---

## 11. Target Transformation / Δ-Learning

### Concept
The cheapest, highest-leverage intervention for OOD regression. Two complementary sub-variants: (a) apply a monotone transform to the target `y` so that the transformed target has a more linear relationship with features — compressing the training y-range while expanding the extrapolation region; (b) fit a simple linear baseline first, then train the NN only on residuals so the linear baseline handles all extrapolation and the NN learns a bounded correction.

### Implementation Instructions

**11a. Monotone Target Transform:**
```python
from sklearn.preprocessing import PowerTransformer

# Fit transformer on training targets ONLY
pt = PowerTransformer(method='yeo-johnson')
y_train_t = pt.fit_transform(y_train.reshape(-1, 1)).ravel()

# Train any regression model on y_train_t
# At prediction time, invert the transform:
y_pred_t = model.predict(X_test)
y_pred = pt.inverse_transform(y_pred_t.reshape(-1, 1)).ravel()
```
Other transforms to try: `log(y - y_min + eps)` for positive-skewed targets; rank transform followed by inverse-Gaussian CDF (quantile normalization via `sklearn.preprocessing.QuantileTransformer`).

**11b. Δ-Learning (Residual Learning on Linear Baseline):**
```python
from sklearn.linear_model import Ridge

# Step 1: fit a linear baseline
base = Ridge(alpha=1.0).fit(X_train, y_train)
residuals = y_train - base.predict(X_train)

# Step 2: train MLP to predict residuals
mlp.fit(X_train, residuals)

# Step 3: final prediction = linear baseline + learned residual
y_pred = base.predict(X_test) + mlp.predict(X_test)
```
Try `Ridge`, `Lasso`, and shallow polynomial (`PolynomialFeatures(degree=2) + Ridge`) as baselines. The linear component extrapolates; the NN only corrects bounded residuals. This is the single most reliable improvement over a vanilla MLP baseline.

---

## 12. GP with Linear + RBF Kernel — Structured Extrapolation

### Concept
A carefully chosen GP kernel directly encodes extrapolation behavior. The `DotProduct` (linear) kernel handles long-range extrapolation; the `RBF` kernel captures local nonlinear structure. Far from training data, the RBF contribution decays and predictions revert to the learned linear trend rather than to the prior mean. This single model frequently outperforms deep neural networks on tabular OOD benchmarks.

### Implementation Instructions
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel

kernel = DotProduct() + RBF() + WhiteKernel()
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                               n_restarts_optimizer=10)
gp.fit(X_train, y_train)
y_pred, y_std = gp.predict(X_test, return_std=True)
```
- **Key insight**: Use `DotProduct` not `ConstantMean` — the linear term is what enables principled extrapolation beyond training data.
- **Kernel combinations to try**: `DotProduct + Matern(nu=2.5) + WhiteKernel`; `DotProduct + RBF + ConstantKernel`.
- **Scaling**: StandardScaler both X and y (fit on training only). GP is sensitive to scale.
- **Scalability limit**: Exact GP scales O(N³). For N > 2000, switch to SVGP (Approach 10) with a `LinearKernel + MaternKernel` covariance.

---

## 13. Deep Kernel Learning (DKL) — Neural Features + GP Head

### Concept
A neural network learns task-specific features, and a GP operates on those learned latent representations. This combines the representational power of deep networks with the principled uncertainty quantification and extrapolation behavior of GPs. The mean function must be `LinearMean` (not `ConstantMean`) to enable extrapolation in feature space. Adding a `LinearKernel` to the covariance further reinforces linear extrapolation outside the training hull.

### Implementation Instructions
```python
import gpytorch

class DKLModel(gpytorch.models.ExactGP):
    def __init__(self, X_train, y_train, likelihood, latent_dim=32):
        super().__init__(X_train, y_train, likelihood)
        self.feature_extractor = nn.Sequential(
            nn.Linear(X_train.shape[1], 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        # Use LinearMean, NOT ConstantMean — enables OOD extrapolation
        self.mean_module = gpytorch.means.LinearMean(latent_dim)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel() + gpytorch.kernels.LinearKernel()
        )

    def forward(self, x):
        z = self.feature_extractor(x)
        # Normalize features to unit sphere for numerical stability
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(z), self.covar_module(z))

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = DKLModel(X_train_t, y_train_t, likelihood)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train(); likelihood.train()
    optimizer.zero_grad()
    output = model(X_train_t)
    loss = -mll(output, y_train_t)
    loss.backward(); optimizer.step()
```
- **Critical**: Always use `LinearMean` and add `LinearKernel` to covariance. Without these, DKL reverts to prior mean outside training data, identical to vanilla GP failure mode.
- **Scalability**: Switch to `gpytorch.models.ApproximateGP` with inducing points for N > 2000.

---

## 14. Neural Additive Models + Symbolic Regression Residuals (NAM + PySR)

### Concept
**NAMs** decompose the prediction as a sum of per-feature functions: `f(x) = Σ_j f_j(x_j) + bias`. Each `f_j` is a small independent MLP. The additive structure makes extrapolation interpretable and controllable — you can inspect and enforce linear tails beyond each feature's training support. A **symbolic regression residual** (using PySR) then finds a closed-form correction to the NAM's errors that extrapolates analytically.

### Implementation Instructions
```python
import torch, torch.nn as nn

class NAM(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.feature_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1)
            ) for _ in range(in_dim)
        ])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        contributions = [net(x[:, j:j+1]) for j, net in enumerate(self.feature_nets)]
        return torch.stack(contributions, dim=0).sum(dim=0) + self.bias
```

**Smoothness regularization**: Add a second-derivative penalty per feature network (estimated numerically on a grid of 100 points spanning each feature's range). This prevents oscillations that cause erratic OOD behavior.

**Boundary linearization**: For each `f_j`, compute the learned slope at `max(x_j^train)` via finite difference. For test values beyond the training boundary, apply: `f_j(x_j) = f_j(max) + slope * (x_j - max)`.

**Symbolic residual correction**:
```python
# pip install pysr
from pysr import PySRRegressor

nam_residuals = y_train - nam.predict(X_train)
sr = PySRRegressor(
    niterations=40,
    binary_operators=["+", "*", "-", "/"],
    unary_operators=["exp", "log", "sqrt"]
)
sr.fit(X_train, nam_residuals)

# Final prediction
y_pred = nam.predict(X_test) + sr.predict(X_test)
```
The NAM captures bulk nonlinear structure; PySR finds a physics-plausible closed-form correction that extrapolates correctly.

---

## 15. Monotonic MLP — Constrained Extrapolation

### Concept
For features where the target is known or suspected to be monotonically increasing (or decreasing), enforcing monotonicity during training forces sensible, bounded extrapolation in that direction. Non-negative weights along all paths from monotone input features to output guarantee the monotonicity constraint is never violated, even on OOD inputs far beyond the training support.

### Implementation Instructions
```python
import torch, torch.nn as nn, torch.nn.functional as F

class MonotonicMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, mono_features=None):
        super().__init__()
        # mono_features: list of feature indices that must be monotone-increasing
        self.mono_mask = torch.zeros(in_dim, dtype=torch.bool)
        if mono_features:
            self.mono_mask[mono_features] = True
        self.W1 = nn.Parameter(torch.randn(in_dim, hidden) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.W2 = nn.Parameter(torch.randn(hidden, 1) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        W1_eff = self.W1.clone()
        # Force weights from monotone input features to be non-negative
        W1_eff[self.mono_mask] = F.softplus(self.W1[self.mono_mask])
        # Force last layer weights to be non-negative
        W2_eff = F.softplus(self.W2)
        h = torch.tanh(x @ W1_eff + self.b1)
        return h @ W2_eff + self.b2
```
- **Identify monotone features**: domain knowledge, or empirically via `np.corrcoef` between each feature and y on training data (features with `|r| > 0.5` are candidates).
- **Multi-layer extension**: For deeper networks, apply `softplus` weight projection at every layer on the monotone feature pathway. Use `tanh` activations (monotone) rather than `ReLU` to preserve the global monotonicity guarantee through all layers.
- **Training**: Standard MSE loss. No other changes needed.

---

## 16. Gated Extrapolating MLP — Adaptive Linear/Nonlinear Blending

### Concept
A vanilla MLP saturates at `max(y_train)` on OOD inputs. A linear model extrapolates correctly but underfits in-distribution. This approach blends both: a **gate signal** based on Mahalanobis distance from the training centroid continuously interpolates between the MLP (for in-distribution inputs) and a linear head (for OOD inputs). As test inputs venture further from training data, the model automatically transitions to linear behavior.

### Implementation Instructions
```python
import torch, torch.nn as nn
import numpy as np

class GatedExtrapolatingMLP(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x, gate):
        # gate in [0, 1]: 0 = pure MLP, 1 = pure linear
        return (1 - gate) * self.mlp(x) + gate * self.linear(x)

# Compute gate at inference time using Mahalanobis distance
def compute_gate(X_test, X_train, temperature=1.0):
    mu = X_train.mean(axis=0)
    cov = np.cov(X_train.T) + 1e-6 * np.eye(X_train.shape[1])
    cov_inv = np.linalg.inv(cov)
    diffs = X_test - mu
    mahal = np.sqrt(np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs))
    # Sigmoid gate: 0 near training centroid, 1 far from it
    gate = torch.sigmoid(torch.tensor((mahal - mahal.mean()) / temperature, dtype=torch.float32))
    return gate.unsqueeze(1)

# Training: train without gate (set gate=0 for all training samples)
# Inference: compute gate from Mahalanobis distance, pass to forward()
```
- **Temperature**: Controls the sharpness of the transition. Start with `temperature=1.0`. Higher values create a smoother transition zone.
- **Alternative gate**: Use the GP posterior variance (from a cheap linear+RBF GP) as the gate signal instead of Mahalanobis distance.

---

## 17. Anchor Points with Extrapolation Prior

### Concept
Generate synthetic `(X, y)` training points in the extrapolation region beyond the training distribution using a simple prior (linear fit, power law, or dimensional argument). Add these anchor points to the training set with a down-weighted loss. This "gently tells" the model how to behave in OOD territory without overriding in-distribution fit. The approach is model-agnostic — it works with any base regressor.

### Implementation Instructions
```python
from sklearn.linear_model import Ridge
import numpy as np

# Step 1: fit linear prior on training data
base = Ridge(alpha=1.0).fit(X_train, y_train)

# Step 2: generate anchor X beyond training support
# Strategy A: perturb top-decile training samples in the direction of high y
top_idx = np.argsort(y_train)[-len(y_train)//10:]
X_anchor = X_train[top_idx] + 0.5 * X_train[top_idx].std(axis=0) * np.random.randn(
    len(top_idx), X_train.shape[1])
# Strategy B: sample uniformly in 2x the feature bounding box
X_anchor_bbox = np.random.uniform(
    X_train.min(axis=0), 2 * X_train.max(axis=0) - X_train.min(axis=0),
    size=(200, X_train.shape[1]))

# Step 3: set anchor labels from linear prior
y_anchor = base.predict(X_anchor)

# Step 4: combine with down-weighted sample weights
X_aug = np.vstack([X_train, X_anchor])
y_aug = np.concatenate([y_train, y_anchor])
weights = np.concatenate([np.ones(len(y_train)), 0.1 * np.ones(len(y_anchor))])

# Step 5: train any model with sample_weight argument
# For PyTorch: multiply anchor losses by 0.1 before .backward()
```
- **Anchor weight**: `0.1` is a good starting point. Too high → anchor prior dominates; too low → no effect. Tune via OOD validation set.
- **Prior choice**: Linear is safest. If domain suggests power-law behavior, use `y_anchor = a * X_anchor[:, j]**b` for the dominant feature.

---

## 18. Asymmetric / Tilted Loss (Quantile-Shifted Training)

### Concept
Standard MSE penalizes over- and under-prediction symmetrically. In OOD regression where test targets are systematically higher than training targets, asymmetrically penalizing **under-prediction** more than over-prediction shifts the model's learned function upward, compensating for the OOD gap. The tilted/pinball loss provides a continuous knob controlling this asymmetry.

### Implementation Instructions
```python
def tilted_loss(pred, target, tau=0.7):
    """
    tau=0.5 is equivalent to MAE (symmetric).
    tau > 0.5 penalizes under-prediction more (pushes predictions higher).
    tau < 0.5 penalizes over-prediction more.
    """
    err = target - pred
    return torch.mean(torch.maximum(tau * err, (tau - 1) * err))
```
- **How to choose `tau`**: If OOD test y is systematically larger than training y, use `tau ∈ [0.6, 0.85]`. Tune on an OOD validation set. If direction of shift is unknown, use `tau=0.5` (default MAE).
- **Combination**: Stack with Δ-learning (Approach 11) — use tilted loss on the residuals in the Δ-learning framework for double benefit.
- **Multiple quantiles**: Train separate models for `tau=0.1`, `0.5`, `0.9` and use the ensemble's spread as an uncertainty signal (quantile regression ensemble).

---

## 19. Extrapolation Mixup — Linear Behavior Beyond Training Pairs

### Concept
Standard Mixup interpolates between training pairs with `λ ∈ [0, 1]`. **Extrapolation Mixup** extends this to `λ > 1` or `λ < 0`, generating synthetic training points *beyond* the convex hull of the training data. Training on extrapolated pairs forces the model to behave approximately linearly between and beyond training examples — directly targeting the extrapolation failure mode of vanilla MLPs.

### Implementation Instructions
```python
import numpy as np
import torch

def extrap_mixup_batch(X_batch, y_batch, extrap_prob=0.3, extrap_range=0.3):
    """
    With probability extrap_prob, use lambda outside [0,1] for extrapolation.
    Otherwise, use standard interpolation lambda in [0,1].
    """
    batch_size = X_batch.shape[0]
    idx = torch.randperm(batch_size)
    X_j, y_j = X_batch[idx], y_batch[idx]

    # Sample lambda: mix of interpolation and extrapolation
    lam = np.where(
        np.random.rand(batch_size) < extrap_prob,
        np.random.uniform(-extrap_range, 1 + extrap_range, batch_size),  # extrapolation
        np.random.uniform(0, 1, batch_size)                               # interpolation
    )
    lam = torch.tensor(lam, dtype=torch.float32).unsqueeze(1)

    X_mix = lam * X_batch + (1 - lam) * X_j
    y_mix = lam.squeeze() * y_batch + (1 - lam.squeeze()) * y_j
    return X_mix, y_mix

# In training loop:
for X_batch, y_batch in train_loader:
    X_mix, y_mix = extrap_mixup_batch(X_batch, y_batch)
    loss = F.mse_loss(model(X_mix), y_mix)
    loss.backward(); optimizer.step()
```
- **`extrap_range`**: Controls how far beyond the training pairs the model is trained to extrapolate linearly. Start with `0.3` (i.e., `λ ∈ [-0.3, 1.3]`). Increase carefully — too large causes training instability.
- **`extrap_prob`**: Fraction of batches using extrapolation mixup. `0.3` is a good default.

---

## 20. NALU — Neural Arithmetic Logic Units

### Concept
**NALU** units learn to perform gated addition/subtraction and multiplication/division operations instead of standard linear transformations. Because they encode arithmetic operations rather than arbitrary mappings, NALU networks extrapolate arithmetic relationships (e.g., F = Gm₁m₂/r², energy = mass × velocity²) far outside the training distribution. Most effective when the true underlying relationship is multiplicative or involves ratios.

### Implementation Instructions
```python
import torch, torch.nn as nn

class NALU(nn.Module):
    """Learns gated add-vs-multiply operations for arithmetic extrapolation."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_hat = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        self.M_hat = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        self.G     = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)

    def forward(self, x):
        # Additive path: W = tanh(W_hat) * sigmoid(M_hat)
        W = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
        a = x @ W.T                                                        # linear combination
        # Multiplicative path: log-space linear combination → exponentiate
        m = torch.exp(torch.log(torch.abs(x) + 1e-7) @ W.T)
        g = torch.sigmoid(x @ self.G.T)                                    # gate
        return g * a + (1 - g) * m

# Usage: stack 2-3 NALU layers
model = nn.Sequential(
    NALU(in_dim, 32),
    NALU(32, 16),
    nn.Linear(16, 1)
)
```
- **When to use**: When domain knowledge suggests the relationship involves products, ratios, or power laws (e.g., material property ∝ density × elastic_modulus).
- **When to skip**: On messy real-world tabular data without clear arithmetic structure, NALU often underperforms simpler baselines. Use as a specialized tool, not a default.
- **Numerical stability**: The log in the multiplicative path requires all inputs to be positive. Apply `F.softplus(x)` before NALU if inputs can be negative.

---

## 21. GP + Neural Network Weighted Hybrid

### Concept
A GP and a neural network ensemble are trained independently. At inference, their predictions are blended using a **uncertainty-weighted average**: the GP receives higher weight where its uncertainty is low (near training data) and lower weight far from training data. The NN ensemble picks up the slack in regions where the GP's expressivity is limited. This hybrid exploits the GP's principled extrapolation behavior while retaining the NN's representational power in-distribution.

### Implementation Instructions
```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, WhiteKernel

# Train GP
kernel = DotProduct() + RBF() + WhiteKernel()
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
gp.fit(X_train, y_train)
y_gp, sigma_gp = gp.predict(X_test, return_std=True)

# Train NN ensemble (5 MLPs with different random seeds)
nn_preds = []
for seed in range(5):
    torch.manual_seed(seed)
    mlp = train_mlp(X_train, y_train)   # your training function
    nn_preds.append(mlp.predict(X_test))
y_nn = np.mean(nn_preds, axis=0)
sigma_nn = np.std(nn_preds, axis=0)

# Uncertainty-weighted blend
# GP weight is high when GP is confident (low sigma_gp)
w_gp = sigma_nn / (sigma_gp + sigma_nn + 1e-8)
w_nn = 1 - w_gp
y_hybrid = w_gp * y_gp + w_nn * y_nn
```
- **Interpretation**: In-distribution → NN tends to dominate (lower sigma_nn). OOD → GP tends to dominate (GP uncertainty is more calibrated far from data than NN ensemble uncertainty).
- **Extension**: Replace the 5-MLP ensemble uncertainty with EDL epistemic variance (Approach 9) for a single-model NN uncertainty estimate.

---

## 22. Lipschitz-Constrained Neural Networks

### Concept
Vanilla MLP gradients can be arbitrarily large outside the training distribution, causing erratic extrapolation. A **Lipschitz-constrained network** bounds the gradient norm globally, ensuring that extrapolation changes at most `L` per unit distance in input space. This gives predictable, bounded extrapolation rather than wild saturation or divergence. Spectral normalization of every layer is the standard practical implementation.

### Implementation Instructions
```python
import torch.nn as nn
from torch.nn.utils import spectral_norm

class LipschitzMLP(nn.Module):
    def __init__(self, in_dim, hidden=64, n_layers=3):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden] * n_layers + [1]
        for i in range(len(dims) - 1):
            layers.append(spectral_norm(nn.Linear(dims[i], dims[i+1])))
            if i < len(dims) - 2:
                layers.append(nn.LeakyReLU(0.1))  # LeakyReLU preserves gradient flow
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```
- **Global Lipschitz constant** = product of per-layer spectral norms. For `n_layers=3` all SN-normalized, global L ≤ 1³ = 1. Multiply by a learned scalar output weight if you need predictions to exceed the [−1, +1] range.
- **Training**: Standard MSE loss. The spectral norm constraint is enforced implicitly via the power iteration in `spectral_norm`.
- **Combination**: Use Lipschitz MLP as the NN component in the GP+NN hybrid (Approach 21) for better-behaved uncertainty-weighted blending.

---

## 23. Stochastic Weight Averaging Gaussian (SWAG) — Cheap Posterior Approximation

### Concept
**SWAG** approximates the Bayesian posterior over model weights by tracking the **running mean and low-rank covariance of weight checkpoints** during the final phase of SGD training. At inference, weights are sampled from this Gaussian approximation, producing ensemble-style uncertainty without training multiple independent models. As a bonus, **SWA** (the mean alone, without sampling) tends to converge to flatter loss minima than standard SGD, which improves OOD generalization even as a point estimate.

### Implementation Instructions
```python
# pip install torch (SWAG is built into torch.optim.swa_utils)
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

# Phase 1: standard training for N epochs
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_standard(model, optimizer, train_loader, epochs=80)

# Phase 2: SWA — cyclical LR with weight averaging
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=5e-4)

for epoch in range(20):   # additional 20 epochs of SWA
    train_one_epoch(model, optimizer, train_loader)
    swa_model.update_parameters(model)
    swa_scheduler.step()

# Update BatchNorm statistics
update_bn(train_loader, swa_model)

# SWA prediction (point estimate, more robust than final SGD checkpoint)
y_pred_swa = swa_model(X_test)
```

**For uncertainty (full SWAG)**: Manually collect weight checkpoints every K steps during SWA phase. Compute mean and diagonal + low-rank covariance. At inference, sample 30–50 weight vectors, run forward passes, take mean and std.

```python
# Lightweight diagonal SWAG uncertainty
weight_snapshots = []   # collect model.state_dict() every 5 steps in SWA phase
# At inference:
preds = []
for snapshot in weight_snapshots:
    model.load_state_dict(snapshot)
    preds.append(model(X_test).detach())
y_swag = torch.stack(preds).mean(0)
sigma_swag = torch.stack(preds).std(0)
```

---

## 24. Neural Processes (ANP) — Task-Level Latent Uncertainty

### Concept
**Attentive Neural Processes (ANP)** explicitly model a global latent variable `z` capturing uncertainty about the entire function, not just pointwise predictions. The training set acts as a "context" set and the test set as the "target" set — mirroring few-shot learning. Unlike TabPFN (Approach 6), ANPs are **architecture-agnostic**: the context encoder can be a GNN, CNN, or MLP, making them directly compatible with structured inputs like graphs or sequences. They natively output predictive distributions, providing uncertainty estimates per test point.

### Implementation Instructions
```python
import torch, torch.nn as nn

class ANP(nn.Module):
    def __init__(self, x_dim, y_dim=1, hidden=64, latent_dim=32):
        super().__init__()
        # Deterministic path: cross-attention between context and target
        self.det_encoder = nn.Sequential(nn.Linear(x_dim + y_dim, hidden), nn.ReLU(),
                                          nn.Linear(hidden, hidden))
        self.cross_attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)

        # Latent path: encodes global uncertainty
        self.lat_encoder = nn.Sequential(nn.Linear(x_dim + y_dim, hidden), nn.ReLU(),
                                          nn.Linear(hidden, latent_dim * 2))  # mean + log_var

        # Decoder
        self.decoder = nn.Sequential(nn.Linear(hidden + latent_dim + x_dim, hidden), nn.ReLU(),
                                      nn.Linear(hidden, y_dim * 2))  # mean + log_var of output

    def encode_latent(self, context_x, context_y):
        ctx = torch.cat([context_x, context_y], dim=-1)
        stats = self.lat_encoder(ctx).mean(dim=1)   # aggregate over context
        mu, log_var = stats.chunk(2, dim=-1)
        return mu, log_var

    def forward(self, context_x, context_y, target_x, target_y=None):
        # Latent sample
        mu_z, lv_z = self.encode_latent(context_x, context_y)
        z = mu_z + torch.randn_like(mu_z) * (0.5 * lv_z).exp()
        z_expanded = z.unsqueeze(1).expand(-1, target_x.shape[1], -1)

        # Deterministic path with cross-attention
        ctx_repr = self.det_encoder(torch.cat([context_x, context_y], dim=-1))
        tgt_query = self.det_encoder(torch.cat([target_x,
                                                 torch.zeros(*target_x.shape[:-1], 1)], dim=-1))
        r, _ = self.cross_attn(tgt_query, ctx_repr, ctx_repr)

        # Decode
        dec_input = torch.cat([r, z_expanded, target_x], dim=-1)
        out = self.decoder(dec_input)
        y_mu, y_lv = out.chunk(2, dim=-1)
        return y_mu, y_lv, mu_z, lv_z

# Training: maximize ELBO = log p(y_target | context, z) - KL(q(z|all) || q(z|context))
```
- **Context = training set**, **Target = test set** at inference time. No gradient update needed at test time.
- **Scalability**: For large training sets, subsample a random context of 200–500 points per forward pass during both training and inference.

---

## 25. Anchor Regression — Confounder-Robust Extrapolation

### Concept
**Anchor Regression** handles the case where a known *anchor variable* `A` (e.g., DFT functional, measurement instrument, lab source, synthesis temperature) confounds the relationship between features and target. Standard regression learns coefficients corrupted by this confounding. Anchor Regression regularizes predictions to be robust against interventions on `A`, by penalizing the correlation between residuals and the anchor: `min_β ||Y - Xβ||² + γ||A^T(Y - Xβ)||²`. The parameter `γ` interpolates between OLS (γ=0) and an IV-like causal estimator (γ→∞).

### Implementation Instructions
```python
import numpy as np
from sklearn.linear_model import LinearRegression

def anchor_regression(X, y, A, gamma=1.0):
    """
    X: feature matrix (N x p)
    y: target (N,)
    A: anchor variable(s) (N x q) — known confounders
    gamma: regularization strength. 0 = OLS, large = maximally robust to A-interventions.
    """
    n = X.shape[0]
    # Project X and y onto space orthogonal to A (residualize)
    # H_A = A(A^T A)^{-1} A^T  (projection onto A)
    H_A = A @ np.linalg.pinv(A.T @ A) @ A.T

    # Anchor-augmented gram matrix
    # min ||y - Xβ||² + γ||A^T(y - Xβ)||² = min ||(I + γ H_A)(y - Xβ)||²
    W = np.eye(n) + gamma * H_A
    X_aug = W @ X
    y_aug = W @ y

    # Solve weighted OLS
    beta = np.linalg.lstsq(X_aug.T @ X_aug, X_aug.T @ y_aug, rcond=None)[0]
    return beta

# Usage
beta = anchor_regression(X_train, y_train, A_train, gamma=10.0)
y_pred = X_test @ beta
```

**For neural networks**: Add an anchor regularization term to any loss:
```python
def anchor_loss(pred, target, A, gamma=1.0):
    residuals = target - pred.squeeze()
    # Penalize correlation between residuals and anchor variables
    anchor_penalty = (A.T @ residuals).pow(2).mean()
    return F.mse_loss(pred.squeeze(), target) + gamma * anchor_penalty
```
- **`gamma` selection**: Use cross-validation on a held-out set from a different anchor condition (e.g., different DFT functional). `gamma ∈ [1, 100]` is a typical search range.
- **Anchor variable examples**: DFT functional used (PBE vs. HSE), source database (Materials Project vs. AFLOW), synthesis method, characterization instrument.

---

## Summary Comparison Table

| # | Method | Needs Env/Group Labels | Needs Labeled OOD | Uncertainty Output | Best For |
|---|--------|----------------------|-------------------|--------------------|----------|
| 1 | **Group DRO** | ✅ Yes | ❌ No | ❌ None | Worst-case robust training when groups are known |
| 2 | **IRM / REx** | ✅ Yes | ❌ No | ❌ None | Spurious correlation removal across environments |
| 3 | **Weighted Conformal Prediction** | ❌ No | ❌ No | ✅ Guaranteed intervals | Post-hoc coverage on any base model |
| 4 | **SNGP** | ❌ No | ❌ No | ✅ Distance-aware std | Overconfidence prevention, geometry-grounded UQ |
| 5 | **TTT** | ❌ No | ❌ No (unlabeled ok) | ❌ None | Adapts encoder to unlabeled test distribution |
| 6 | **Hybrid Latent-TabPFN** | ❌ No | ✅ As support set | ✅ Posterior std | Small-data Bayesian inference, high-dim inputs |
| 7 | **BLL-GP** | ❌ No | ❌ No | ✅ Exact GP variance | Lightest few-shot adaptation, exact UQ |
| 8 | **MAML / ANIL** | ✅ As tasks | ✅ Optional | ❌ None | Few-shot OOD with labeled support |
| 9 | **EDL** | ❌ No | ❌ No | ✅ Epistemic + aleatoric | Single-model UQ, no ensemble needed |
| 10 | **SVGP** | ❌ No | ❌ No | ✅ GP variance | Scalable GP for N > 2000 |
| 11 | **Δ-Learning / Target Transform** | ❌ No | ❌ No | ❌ None | Cheapest, highest-leverage OOD improvement |
| 12 | **GP Linear + RBF Kernel** | ❌ No | ❌ No | ✅ GP variance | Strongest simple baseline for tabular OOD |
| 13 | **Deep Kernel Learning (DKL)** | ❌ No | ❌ No | ✅ GP variance | High-dim data where plain GP struggles |
| 14 | **NAM + Symbolic Residuals (PySR)** | ❌ No | ❌ No | ❌ None | Interpretable model + physics-plausible extrapolation |
| 15 | **Monotonic MLP** | ❌ No | ❌ No | ❌ None | Known monotone feature-target relationships |
| 16 | **Gated Extrapolating MLP** | ❌ No | ❌ No | Partial (via gate) | Smooth MLP→linear transition based on OOD distance |
| 17 | **Anchor Points with Prior** | ❌ No | ❌ No | ❌ None | Any model; enforce linear behavior beyond training hull |
| 18 | **Asymmetric / Tilted Loss** | ❌ No | ❌ No | ❌ None | Systematic directional OOD shift in y |
| 19 | **Extrapolation Mixup** | ❌ No | ❌ No | ❌ None | Enforcing linear behavior beyond training pairs |
| 20 | **NALU** | ❌ No | ❌ No | ❌ None | Multiplicative / ratio-based physical relationships |
| 21 | **GP + NN Weighted Hybrid** | ❌ No | ❌ No | ✅ Blended UQ | Best of GP extrapolation + NN expressivity |
| 22 | **Lipschitz-Constrained MLP** | ❌ No | ❌ No | ❌ None | Bounded, predictable extrapolation slope |
| 23 | **SWAG** | ❌ No | ❌ No | ✅ Approx posterior std | Cheap ensemble-quality UQ from single training run |
| 24 | **Attentive Neural Processes (ANP)** | ❌ No | ✅ As context | ✅ Predictive dist | GNN-compatible, task-level Bayesian UQ |
| 25 | **Anchor Regression** | ❌ No | ❌ No | ❌ None | Known confounders (instrument, lab, functional) |

---

## Recommended Stacks by Scenario

**Scenario A — Environment labels available, medium dataset:**
> **Group DRO** or **IRM/REx** as training objective → **SNGP** or **DKL** as model → **Weighted CP** as post-hoc interval wrapper.

**Scenario B — No environment labels, small dataset (< 1000 samples):**
> **Δ-Learning** (Ridge + MLP residuals) → **GP Linear+RBF** as primary model → **BLL-GP** or **Hybrid Latent-TabPFN** for Bayesian head → **Weighted CP** for coverage guarantee.

**Scenario C — No environment labels, larger dataset (> 2000 samples):**
> **Extrapolation Mixup** + **Asymmetric Loss** during training → **SNGP** or **DKL** as model → **SVGP** as non-parametric baseline → **Weighted CP** for intervals.

**Scenario D — Known physical structure (monotonicity, multiplicative laws):**
> **Monotonic MLP** (if monotone features exist) or **NALU** (if multiplicative structure) → **NAM + PySR** for interpretable residual → **Anchor Regression** if known confounders exist → **Anchor Points** to regularize OOD behavior.

**Scenario E — Small labeled OOD support set available at test time:**
> **MAML** initialization → fast-adapt with few OOD samples → **ANP** for full predictive distribution → **Weighted CP** for guaranteed intervals.

**Scenario F — Minimal compute budget:**
> **Δ-Learning** (Approach 11, near-zero overhead) → **GP Linear+RBF** (Approach 12, no deep learning needed) → **SWAG** (Approach 23, single training run) → **Tilted Loss** (Approach 18, one-line change).