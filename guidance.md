## Task Description

---

You are designing an approach to tackle Out-of-Distribution (OOD) regression. This means the test data will likely contain samples that are considerably different with respect to the training data distribution.

## Key Assumptions

---

- Train and test sets are fixed before any model training begins.
- No oracle is queried during or after training.
- Test labels are never seen during training or calibration.
- The primary objective is **accurate, well-calibrated prediction on OOD test samples**.
 
## Approaches to consider

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
   # Save a snapshot of encoder weights before adaptation
   encoder_snapshot = deepcopy(model.encoder.state_dict())
 
   model.encoder.train(); model.ssl_head.train()
   model.reg_head.eval()
   optimizer_ttt = Adam(
       list(model.encoder.parameters()) + list(model.ssl_head.parameters()), lr=1e-4)
 
   for step in range(50):   # adapt on full test batch
       ssl_loss = compute_ssl_loss(model, X_test_all)
       optimizer_ttt.zero_grad(); ssl_loss.backward(); optimizer_ttt.step()
 
   # Predict with adapted encoder + frozen regression head
   model.encoder.eval()
   with torch.no_grad():
       y_pred = model.reg_head(model.encoder(X_test_all))
 
   # Restore encoder weights (good practice for reproducibility)
   model.encoder.load_state_dict(encoder_snapshot)
   ```
5. **Batch vs. Sample-Level TTT**: Adapting on all test samples jointly (batch TTT) is preferred when the test set is large and homogeneous. Adapt per-sample only if test samples are from very different sub-distributions.
 
---
 
## 6. Hybrid Latent-TabPFN (In-Context Bayesian Inference)
 
### Concept
**TabPFN** and **TabICL** perform regression via *in-context learning* (ICL): the training set (support) and test set (query) are passed in a single forward pass with **no gradient updates**. TabPFN was pre-trained on millions of synthetic datasets sampled from diverse Bayesian priors, mathematically approximating **optimal Bayesian posterior predictive inference** over the support set. The hybrid variant prepends a frozen domain-specific encoder (e.g., a GNN or VAE encoder) to compress high-dimensional inputs into a low-dimensional latent vector.
 
### Implementation Instructions
1. **Encoder**: Load a pre-trained domain encoder (e.g., `DimeNet++` or `CGCNN` from `torch_geometric`). Freeze all weights: `param.requires_grad = False`. Extract latent vectors of dimension ≤ 100 for all train and test samples using `torch.no_grad()`.
2. **Dimensionality Check**: If latent dim > 100, apply `PCA(n_components=64)` fit **only on training embeddings**, then transform test embeddings with the same fitted PCA. Never refit PCA on test data.
3. **TabPFN Install**: `pip install tabpfn`.
4. **Fit & Predict**:
   ```python
   from tabpfn import TabPFNRegressor
   model = TabPFNRegressor(device='cuda', N_ensemble_configurations=16)
   model.fit(X_train_latent, y_train)
   y_pred, y_std = model.predict(X_test_latent, return_std=True)
   ```
5. **Uncertainty Output**: `y_std` gives predictive standard deviation per test sample. High values indicate OOD test samples — useful for flagging low-confidence predictions in the fixed test set.
6. **Sample Limit**: TabPFN degrades above ~1000 training samples. If training set is larger, subsample a representative 512–1000 point support set (e.g., via k-means centroids on the latent space).
 
---
 
## 7. Bayesian Last-Layer with Gaussian Process Head (BLL-GP)
 
### Concept
Freeze a deep feature extractor pre-trained on source data. Discard the deterministic regression head and replace it with an **exact Gaussian Process** fit on the training set in the encoder's latent space. The GP inherits the source model's structural understanding of the data while providing exact Bayesian uncertainty quantification. Because the GP is fit on the fixed training set and evaluated on the fixed test set, no streaming or oracle interaction is needed.
 
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
3. **Kernel Selection**: `Matern(nu=2.5)` is the robust default. For normalized graph embeddings, also try `DotProduct` kernel (equivalent to cosine similarity covariance). Compare marginal log-likelihood via `gp.log_marginal_likelihood_value_`.
4. **Scalability**: Exact GP scales as O(N³). If training set exceeds ~2000 samples, switch to `GPyTorch`'s `ExactGP` with CG solvers, or use SVGP (see below).
5. **OOD Detection**: `y_std` on the fixed test set directly identifies which test predictions are uncertain/OOD. Report alongside predictions as a reliability flag.
 
---
 
## 8. Model-Agnostic Meta-Learning — MAML / ANIL (Few-Shot Adaptation)
 
### Concept
**MAML** optimizes model initialization weights θ* such that a single gradient step on a new task's support set yields maximal performance. In the fixed train-test setting, the training set is used as the meta-training source (structured into synthetic few-shot tasks by environment), and the few labeled OOD test samples (if any) form the adaptation support set. **ANIL** (Almost No Inner Loop) reduces cost by only updating the final regression head during inner-loop adaptation.
 
### Implementation Instructions
1. **Library**: `pip install learn2learn`.
2. **Task Construction**: Partition source training data by environment label. Each meta-training episode: sample one environment as support (K=5–20 samples), another split as query.
3. **Inner Loop** (task-specific fast adaptation):
   ```python
   import learn2learn as l2l
   maml = l2l.algorithms.MAML(model, lr=0.01, first_order=True)  # first_order=True for FOMAML (cheaper)
   learner = maml.clone()
   support_loss = F.mse_loss(learner(X_support), y_support)
   learner.adapt(support_loss)
   ```
4. **Outer Loop** (meta-update across tasks):
   ```python
   query_loss = F.mse_loss(learner(X_query), y_query)
   meta_optimizer.zero_grad(); query_loss.backward(); meta_optimizer.step()
   ```
5. **Fixed Test Deployment**: If a small labeled support set exists in the OOD test domain, clone the meta-model and run 1–5 inner-loop gradient steps on it before predicting on the unlabeled test samples. If no OOD labels are available, the meta-initialization itself serves as the base model — evaluate zero-shot.
6. **ANIL Variant**: Freeze all layers except the final linear head during the inner loop. Reduces meta-training compute by ~60% with marginal performance loss.
 
---
 
## 9. Evidential Deep Learning for Regression (EDL / NIG)
 
### Concept
**EDL** trains a single deterministic network to output the four parameters of a **Normal-Inverse-Gamma (NIG) distribution** — a higher-order prior over Gaussian likelihood parameters. From these four outputs (γ, ν, α, β), the model simultaneously yields: (1) the predicted mean, (2) aleatoric uncertainty (irreducible data noise), and (3) epistemic uncertainty (model ignorance, which inflates on OOD test samples). No ensembles, no MC Dropout, no sampling needed at inference.
 
### Implementation Instructions
1. **Output Layer**: Replace final `nn.Linear(..., 1)` with `nn.Linear(..., 4)` outputting raw logits for (γ, ν, α, β):
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
   epistemic_var = beta / (nu * (alpha - 1))   # inflates on OOD test samples
   aleatoric_var = beta / (alpha - 1)           # irreducible data noise
   ```
4. **Evaluation on Fixed Test Set**: Report RMSE alongside epistemic uncertainty calibration — compute the Spearman rank correlation between `epistemic_std` and absolute test errors. A well-calibrated EDL model should show strong positive correlation.
5. **Hyperparameter**: Tune `lambda_reg ∈ [0.001, 0.1]`. Too low → overconfident OOD predictions. Too high → regression target underfits.
6. **Library**: Reference implementation at `github.com/aamini/evidential-deep-learning`.
 
---
 
## 10. Sparse Variational GP (SVGP) — Scalable Non-Parametric Regression
 
### Concept
In the fixed train-test setting, a plain **Sparse Variational GP** (without online updates) serves as a scalable, uncertainty-aware non-parametric regression model when the training set is too large for exact GPs (> ~2000 samples). SVGP approximates the full GP posterior using M ≪ N learned *inducing points* {Z_m}, yielding O(NM²) complexity. The variational posterior q(f) is optimized via ELBO maximization over the fixed training set.
 
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
3. **Inducing Points Initialization**: Sample M=100–500 points from the training set via k-means clustering on the input features. More inducing points → better approximation, higher cost.
4. **Training**:
   ```python
   likelihood = gpytorch.likelihoods.GaussianLikelihood()
   mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(y_train))
   optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': likelihood.parameters()}], lr=0.01)
   for epoch in range(200):
       for X_batch, y_batch in train_loader:
           optimizer.zero_grad()
           output = model(X_batch)
           loss = -mll(output, y_batch)
           loss.backward(); optimizer.step()
   ```
5. **Prediction on Fixed Test Set**:
   ```python
   model.eval(); likelihood.eval()
   with torch.no_grad(), gpytorch.settings.fast_pred_var():
       pred = likelihood(model(X_test))
       y_pred = pred.mean
       y_std  = pred.stddev
   ```
6. **Deep Kernel Option**: Replace `MaternKernel` with a learned neural net backbone (`gpytorch.kernels.GridInterpolationKernel` over a 2-layer MLP) for higher expressivity on structured or high-dimensional inputs.
 
---
 
## Summary Comparison Table
 
| Rank | Method | Requires Group/Env Labels | Requires Labeled OOD Samples | Uncertainty Output | Primary Benefit in Fixed Split Setting |
|------|--------|--------------------------|------------------------------|--------------------|----------------------------------------|
| 1 | **Group DRO** | ✅ Yes | ❌ No | ❌ None | Worst-case robust training objective |
| 2 | **IRM / REx** | ✅ Yes | ❌ No | ❌ None | Removes spurious correlations across envs |
| 3 | **Weighted Conformal Prediction** | ❌ No | ❌ No (calibration set only) | ✅ Guaranteed intervals | Post-hoc coverage guarantee on any model |
| 4 | **SNGP** | ❌ No | ❌ No | ✅ Distance-aware std | Overconfidence prevention + robust features |
| 5 | **TTT** | ❌ No | ❌ No (unlabeled test ok) | ❌ None | Adapts encoder to unlabeled test distribution |
| 6 | **Hybrid Latent-TabPFN** | ❌ No | ✅ Used as support set | ✅ Calibrated posterior std | Strong small-data Bayesian regressor |
| 7 | **BLL-GP** | ❌ No | ❌ No | ✅ Exact GP variance | Lightest-weight uncertainty over frozen features |
| 8 | **MAML / ANIL** | ✅ (as tasks) | ✅ Optional few-shot | ❌ None | Good initialization for few-shot OOD |
| 9 | **EDL** | ❌ No | ❌ No | ✅ Epistemic + aleatoric | Single-model uncertainty, no ensemble needed |
| 10 | **SVGP** | ❌ No | ❌ No | ✅ GP variance | Scalable GP when training set > 2000 samples |
 
---