# Scanova — Explainable Breast Cancer Intelligence Platform

Scanova is a breast-imaging **decision-support** system. Four EfficientNet-B0 encoders (CBIS-DDSM mammography, RSNA mammography, BUSI ultrasound, Duke breast MRI) are trained separately, then frozen. A fusion MLP reads their 1280-dimensional embeddings and outputs one malignancy probability.

**Fusion V3** (all four encoders) is the model the website runs. **Fusion V2** (no MRI) is the fallback if `fusion_v3_model.pth` is missing.

This repository is the **Flask site** that serves the trained weights. Training scripts and metric JSON files live separately (historically under `G:\dev_part1`). Website copies of preprocessing, fusion, Grad-CAM++, and Monte Carlo dropout are in `backend/services/image_service.py` and `backend/inference/engine.py`.

> **Disclaimer.** Scanova is a research prototype. It does **not** replace a radiologist’s reading or a pathology result.

---

## Table of contents

1. [Quick start](#quick-start)
2. [Algorithms used](#algorithms-used)
3. [Project flow](#project-flow)
4. [Recorded test results](#recorded-test-results)
5. [What each model is](#what-each-model-is)
6. [Datasets](#datasets)
7. [How training was done](#how-training-was-done)
8. [What the website does](#what-the-website-does)
9. [Roles](#roles)
10. [Inference steps](#inference-steps)
11. [Folder layout](#folder-layout)
12. [Environment](#environment)
13. [Security and storage](#security-and-storage)
14. [API](#api)
15. [Pages](#pages)
16. [Troubleshooting](#troubleshooting)
17. [References](#references)

---

## Quick start

```bash
git clone https://github.com/gandhidev2005/Scanova-Explainable-Breast-Cancer-Intelligence-Platform.git
cd Scanova-Explainable-Breast-Cancer-Intelligence-Platform

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env          # Windows; use cp on Unix
# Edit .env → set SCANOVA_SECRET, ADMIN_PASSWORD, optional GROQ_*

python app.py
```

Open [http://localhost:5000](http://localhost:5000).

| Check | URL |
| --- | --- |
| Health | `http://localhost:5000/api/health` |
| Model catalog | `http://localhost:5000/api/models/info` |

Optional NVIDIA wheel (after `requirements.txt`):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## Algorithms used

These are the methods in the training scripts and in the website (`backend/inference/engine.py`, `backend/services/image_service.py`). Formulas match that code.

A comment at the top of `fusion_v3_train.py` once said the MRI encoder AUC is `0.6461`. The saved test file `mri_metrics.json` says **`0.5679`**, and that is the number used everywhere else below. The MRI training script itself is not shipped with the site. There is no 4-class cross-entropy stage in the saved pipeline: every present training script uses `BCEWithLogitsLoss`. Runtime storage is **JSON**, not SQLite.

### 1. EfficientNet-B0

Backbone for every encoder. This project uses the torchvision B0 network with ImageNet-1K V1 weights. It does not re-search the compound coefficients. Those coefficients are the published rule that produced B0:

$$
d = \alpha^{\phi},\quad
w = \beta^{\phi},\quad
r = \gamma^{\phi}
\quad\text{with}\quad
\alpha\cdot\beta^{2}\cdot\gamma^{2}\approx 2,\quad \alpha,\beta,\gamma \ge 1
$$

The paper’s baseline (B0) is $\phi = 0$, so $d = w = r = 1$, with the searched constants $\alpha = 1.2$, $\beta = 1.1$, $\gamma = 1.15$ used only when $\phi > 0$ (B1 and up).

Each MBConv block is an inverted residual:

1. expand by factor $t$ (6 in most blocks)
2. depthwise $3\times 3$ convolution
3. squeeze-excitation
4. project back to $C_{\mathrm{out}}$
5. add the input when the shape matches

Depthwise-separable cost versus a full convolution of kernel $K$:

$$
\begin{aligned}
\text{full} &= H\cdot W\cdot C_{\mathrm{in}}\cdot C_{\mathrm{out}}\cdot K^{2} \\
\text{separable} &= H\cdot W\cdot C_{\mathrm{in}}\cdot K^{2} + H\cdot W\cdot C_{\mathrm{in}}\cdot C_{\mathrm{out}}
\end{aligned}
$$

The ratio of separable cost to full cost is about $1/C_{\mathrm{out}} + 1/K^{2}$. For $K = 3$ that second term is $1/9$. It is not a flat $9\times$ reduction, because the pointwise $1\times 1$ convolution is still there.

Squeeze-excitation inside the block, reduction $r = 4$:

$$
\text{GAP} \rightarrow
\text{Linear } C\!\to\! C/r \rightarrow \mathrm{ReLU} \rightarrow
\text{Linear } C/r\!\to\! C \rightarrow \sigma
\rightarrow \text{channel multiply}
$$

### 2. Transfer learning

Encoder training (`step3_train.py`, `busi_train.py`, `rsna_train.py`):

- start from ImageNet-1K V1 weights
- replace the 1000-way classifier with `Dropout(0.3)` + `Linear(1280 → 1)`
- train every parameter with AdamW

Nothing is frozen during encoder training. The low layers still begin at filters that already detect edges and texture, and training moves them toward mammography or ultrasound.

When a finished encoder is loaded for fusion (and on the website):

- every weight is frozen (`requires_grad = False`)
- the last feature block, `features[-1]`, is marked `requires_grad = True` so Grad-CAM can read its gradients
- the optimizer still updates **only** the fusion MLP

Embeddings are taken under `torch.no_grad`, so that last block is not actually updated during fusion training.

### 3. Binary classification

Each encoder maps a $224\times 224\times 3$ image to one probability:

$$
f(x) = \sigma\!\bigl(w^{\top}\phi(x) + b\bigr),\qquad
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

$\phi(x)$ is the 1280-d embedding. The decision is

$$
\hat{y} =
\begin{cases}
1 & \text{if } f(x) \ge \theta \\
0 & \text{otherwise}
\end{cases}
$$

- $\theta = 0.5$ for CBIS, BUSI, all three fusion heads, and every prediction the website returns
- RSNA’s reported test point uses $\theta = 0.70$
- the saved MRI test point uses $\theta = 0.55$

Labels: malignant $= 1$ and benign $= 0$, except Duke MRI, where triple-negative $= 1$ and other subtypes $= 0$.

### 4. BCEWithLogitsLoss

Loss on the raw logit $\hat{z}$, label $y \in \{0,1\}$, positive weight $w$:

$$
\mathcal{L} = -\Bigl[
  w\cdot y\cdot\log\sigma(\hat{z})
  + (1-y)\cdot\log\bigl(1-\sigma(\hat{z})\bigr)
\Bigr]
$$

PyTorch computes the log-sigmoid in a stable way:

$$
\log\sigma(\hat{z}) = -\log\bigl(1 + e^{-\hat{z}}\bigr)
$$

With $w = 1$ the gradient is $\sigma(\hat{z}) - y$.

$w$ is not a single stored constant, except on RSNA:

| Model | Positive weight $w$ |
| --- | --- |
| CBIS, BUSI, every fusion head | $(\text{benign count}) / (\text{malignant count})$ on that training split (printed, not saved) |
| RSNA encoder | $w = 2.0$ exactly (sampler already balances batches) |

### 5. AdamW

Scripts call `torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)` and leave the other arguments at PyTorch defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^{2} \\
\hat{m}_t &= m_t / (1 - \beta_1^{t}) \\
\hat{v}_t &= v_t / (1 - \beta_2^{t}) \\
\theta &\leftarrow \theta - \alpha\cdot \hat{m}_t / \bigl(\sqrt{\hat{v}_t} + \varepsilon\bigr) - \alpha\cdot\lambda\cdot\theta
\end{aligned}
$$

$\lambda = 10^{-4}$ is applied to the weights themselves, not added into $g_t$. That is the AdamW difference from Adam.

| Model | Learning rate $\alpha$ |
| --- | ---: |
| CBIS, BUSI | $10^{-4}$ |
| RSNA | $3\times 10^{-5}$ |
| Fusion V1, V2, V3 | $5\times 10^{-4}$ |

### 6. CosineAnnealingLR

$$
\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\bigl(1 + \cos(\pi t / T_{\max})\bigr)
$$

The scripts set $T_{\max}$ to the planned epoch count and leave $\eta_{\min}$ at $0$. $\eta_{\max}$ is the learning rate above. The decay is smooth across the run rather than a step drop.

### 7. Data augmentation and normalization

**Training** transforms:

1. resize to $224\times 224$
2. horizontal flip, $p = 0.5$
3. vertical flip, $p = 0.5$
4. rotation uniform in $\pm 15^{\circ}$
5. brightness and contrast jitter $0.2$
   - brightness: $x' = x + \delta$, $\delta \in [-0.2, 0.2]$
   - contrast: $x' = \alpha(x - \mu) + \mu$, $\alpha \in [0.8, 1.2]$

Then ImageNet normalization:

$$
x'_c = \frac{x_c - \mu_c}{\sigma_c},\quad
\mu = [0.485, 0.456, 0.406],\quad
\sigma = [0.229, 0.224, 0.225]
$$

Validation and test use resize and this normalization only.

### 8. Early stopping

1. best validation ROC-AUC starts at $0$
2. if this epoch’s validation AUC is **strictly greater**: save the checkpoint and reset the stall counter
3. else: increment the stall counter; stop when the counter reaches patience, and reload the best file

| Model | Patience |
| --- | ---: |
| CBIS | 10 |
| BUSI | 10 |
| RSNA | 12 |
| Fusion V1 | 8 |
| Fusion V2 | 8 |
| Fusion V3 | 10 |

The MRI script is not in the folder. Its saved history is 14 epochs and the best validation AUC is at the 2nd logged epoch.

### 9. Splits

**Patient-level, CBIS encoder** (`step3_train.py`):

- official test set stays held out (645 images)
- validation is 15% of official-train patients
- patient id is the `P_#####` token in the path
- every image of one patient stays in one split

**Patient-level, RSNA encoder:**

- each patient is labeled by the max cancer value across that patient’s images
- 70% of patients train; the other 30% is split evenly into val and test
- seed 42

**Stratified image split** (BUSI, and the RSNA / BUSI / MRI pools inside fusion training):

```text
train_test_split(..., stratify=labels, random_state=42)
70% / 15% / 15%
```

Fusion’s own CBIS cut is different from the encoder cut: the official test is still held out, but the 15% validation slice is a row shuffle, not the patient split from step3.

### 10. WeightedRandomSampler

Used only by `rsna_train.py`.

$$
w_c = \frac{1}{n_c},\qquad
P(\text{draw sample } i) = \frac{w_{y_i}}{\sum_j w_{y_j}}
$$

`num_samples` = length of the training set, `replacement = True`.

Each batch is about half cancer and half normal (inverse-frequency sampling). The loss weight on top is the fixed $2.0$ from section 4, not a second inverse-frequency weight.

### 11. CLAHE and DICOM loading

In `step2_preprocess.py` and in `backend/services/image_service.py`:

1. read the DICOM pixel array
2. if `PhotometricInterpretation` is `MONOCHROME1`: $\mathrm{arr} \leftarrow \max(\mathrm{arr}) - \mathrm{arr}$
3. min–max scale to $0$–$255$
4. CLAHE with `clipLimit = 2.0` and tile grid $8\times 8$
5. convert to RGB

CLAHE equalizes the histogram inside each tile, after clipping bin counts so a nearly empty background does not get stretched into noise. JPG and PNG uploads skip CLAHE and are only converted to RGB.

### 12. Embedding extraction

$$
\phi(x) = \mathrm{flatten}\bigl(\mathrm{avgpool}(\mathrm{encoder.features}(x))\bigr) \in \mathbb{R}^{1280}
$$

`features()` returns a $7\times 7\times 1280$ map. Global average pooling is

$$
\mathrm{GAP}(A_k) = \frac{1}{H\cdot W}\sum_{i,j} A_k(i,j)
$$

The website and the fusion trainers both use this function. The classifier head is **not** part of the embedding.

### 13. Late fusion by concatenation

$$
\begin{aligned}
z_{\mathrm{V3}} &= [z_{\mathrm{CBIS}};\, z_{\mathrm{RSNA}};\, z_{\mathrm{BUSI}};\, z_{\mathrm{MRI}}] \in \mathbb{R}^{5120} \\
z_{\mathrm{V2}} &= [z_{\mathrm{CBIS}};\, z_{\mathrm{RSNA}};\, z_{\mathrm{BUSI}}] \in \mathbb{R}^{3840} \\
z_{\mathrm{V1}} &= [z_{\mathrm{CBIS}};\, z_{\mathrm{BUSI}}] \in \mathbb{R}^{2560}
\end{aligned}
$$

Each $z_*$ is 1280-d. Concatenation keeps every coordinate. The MLP then learns the mix. **No** cross-attention layer is in the code.

### 14. Fusion MLP

ReLU between linear layers. Dropout after each ReLU except the last linear, which is a single logit (sigmoid is applied in the loss or at inference, not inside the module).

| Version | Widths | Dropout |
| --- | --- | --- |
| V1 | $2560 \to 512 \to 128 \to 1$ | $0.4,\; 0.3$ |
| V2 | $3840 \to 1024 \to 256 \to 64 \to 1$ | $0.4,\; 0.3,\; 0.2$ |
| V3 | $5120 \to 1024 \to 256 \to 64 \to 1$ | $0.4,\; 0.3,\; 0.2$ |

V3, written out:

$$
\begin{aligned}
h_1 &= \mathrm{ReLU}(W_1 z + b_1), & W_1 &\in \mathbb{R}^{1024\times 5120} \\
h_2 &= \mathrm{ReLU}(W_2 h_1 + b_2), & W_2 &\in \mathbb{R}^{256\times 1024} \\
h_3 &= \mathrm{ReLU}(W_3 h_2 + b_3), & W_3 &\in \mathbb{R}^{64\times 256} \\
\hat{y} &= \sigma(W_4 h_3 + b_4), & W_4 &\in \mathbb{R}^{1\times 64}
\end{aligned}
$$

The website’s `FusionHead` class is this same V2/V3 MLP. It does not load the V1 file.

### 15. Modality dropout

Training only. `DROPOUT_PROB = 0.5` in all three fusion scripts.

With probability $0.5$, draw $r$ uniform on $[0,1)$ and zero **exactly one** embedding. Otherwise leave all of them intact.

**Fusion V3:**

| Condition on $r$ | Zeroed encoder |
| --- | --- |
| $r < 0.25$ | CBIS |
| $0.25 \le r < 0.50$ | RSNA |
| $0.50 \le r < 0.75$ | BUSI |
| else | MRI |

Fusion V2 uses the same $0.5$ gate and then thirds ($0.33 / 0.66$) over CBIS, RSNA, and BUSI. Fusion V1 flips a coin between mammography and ultrasound.

At inference the website does **not** apply this dropout. A missing file is a zero embedding for that slot. VinDr zeros BUSI (and MRI, for V3) for the same reason.

### 16. Unpaired label-matched sampling

`FusionDatasetV3.__getitem__` (and the V1/V2 datasets):

1. take the next CBIS row; its label is the label of the whole group
2. draw a random RSNA row with that label
3. draw a random BUSI row with that label
4. draw a random MRI row with that label (V3 only)
5. if a same-label pool is empty, draw from the other pool

The four images are **not** the same patient. The shared label is what keeps a malignant mammogram from being paired with a benign ultrasound.

### 17. Grad-CAM++

- Library: `pytorch_grad_cam.GradCAMPlusPlus`
- Target layer: `encoder.features[-1]` (last MBConv block)
- The website tries encoders in order CBIS → MRI → RSNA → BUSI and uses the first one that loaded

For feature map $A^k$ and class score $S^c$, Grad-CAM++ builds a pixel-wise weight from the second derivative of $S^c$ with respect to $A^k$, then

$$
L^{c} = \mathrm{ReLU}\!\left(\sum_k \alpha_k^{c}\, A^{k}\right)
$$

The single-logit head is indexed as `ClassifierOutputTarget(0)`. That index is “the only logit”, not “class benign”.

The map is min–max scaled, colored with the jet colormap, and returned as a base64 PNG.

### 18. Monte Carlo dropout

Website only, after the fusion probability has already been computed in eval mode. That eval-mode sigmoid is the **stored** probability. It is **not** the average of the dropout samples.

Then, for $T = 15$:

1. turn every Dropout module back to train mode
2. $p_t = \sigma(\mathrm{fusion}(z_{\mathrm{CBIS}}, z_{\mathrm{RSNA}}, z_{\mathrm{BUSI}}, z_{\mathrm{MRI}}))$
3. $\mathrm{uncertainty} = \mathrm{std}\{p_1,\ldots,p_{15}\}$

The UI shows that standard deviation times $100$. If no fusion head loaded, the placeholder standard deviation is $0.03$.

This is the Gal and Ghahramani (2016) use of dropout as a stand-in for a posterior over the fusion-head weights. It is an **epistemic** spread for the head, not a calibrated confidence interval, and it does not measure noise in the image itself.

### 19. L2 norm share

$$
\|z\|_2 = \sqrt{\sum_i z_i^{2}},\qquad
C_k = \frac{\|z_k\|_2}{\|z_{\mathrm{CBIS}}\|_2 + \|z_{\mathrm{RSNA}}\|_2 + \|z_{\mathrm{BUSI}}\|_2 + \|z_{\mathrm{MRI}}\|_2}
$$

The four percentages are what the contribution bars show. JSON fields are named `shap_cbis`, `shap_rsna`, `shap_us`, `shap_mri`. They are these ratios, **not** SHAP values. A zeroed embedding has norm $0$, so a missing modality shows as $0\%$. Magnitude is not the same thing as causal effect on the logit.

### 20. Youden’s $J$

$$
J(\theta) = \mathrm{sensitivity}(\theta) + \mathrm{specificity}(\theta) - 1
= \mathrm{TPR}(\theta) - \mathrm{FPR}(\theta),\qquad
\theta^{*} = \arg\max_{\theta} J(\theta)
$$

`rsna_train.py` computes it via `roc_curve` then $j = \mathrm{tpr} - \mathrm{fpr}$.

| Operating point | Threshold |
| --- | ---: |
| RSNA test | $0.70$ |
| Duke MRI test | $0.55$ |
| VinDr, Fusion V2 | $0.200$ (fit on those 398 images) |
| VinDr, Fusion V3 | $0.498$ (fit on those 398 images) |

The website does **not** use these. It thresholds fusion output at $0.5$.

### 21. Test metrics

| Metric | Definition |
| --- | --- |
| ROC-AUC | ranking quality; does not depend on $\theta$ |
| Sensitivity | $\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$ |
| Specificity | $\mathrm{TN}/(\mathrm{TN}+\mathrm{FP})$ |
| Accuracy | $(\mathrm{TP}+\mathrm{TN})/N$ |

Fusion V2’s confusion counts were not recovered, so its accuracy is the $98.3\%$ stored in `fusion_v2_metrics.json` rather than a recomputed $(\mathrm{TP}+\mathrm{TN})/N$.

### 22. Account and session methods

These are part of the website, not the imaging model:

| Mechanism | Detail |
| --- | --- |
| bcrypt | password hash (`passlib`, bcrypt pinned to `4.0.1`) |
| JWT HS256 | login token, lifetime `JWT_TOKEN_HOURS` (default 24) |
| TOTP | MFA via `pyotp`, verify window of one step |
| Fixed window | login limited to 10 attempts per 15 minutes per IP |

---

## Project flow

```text
CBIS-DDSM DICOM ── step1 CSV ── step2 CLAHE PNG ── step3_train.py
                                                      │
                                                      ▼
                                                best_model.pth   (test AUC 0.7434)

BUSI PNG ── busi_train.py ── busi_model.pth     (test AUC 0.9647)
RSNA PNG ── rsna_train.py ── rsna_model.pth     (test AUC 0.6817)
Duke MRI ── (script not in folder)
            mri_model.pth                       (test AUC 0.5679)

Those four checkpoints are frozen → each image → 1280-d embedding (section 12)

Fusion V1   CBIS+BUSI            2560-d   test AUC 0.9903   acc 97.8%
Fusion V2   CBIS+RSNA+BUSI       3840-d   test AUC 0.9913   acc 98.3%
Fusion V3   CBIS+RSNA+BUSI+MRI   5120-d   test AUC 0.9897   acc 98.76%
            specificity 100% (FP = 0), sensitivity 96.97%, n = 645

External check (no retraining):
  VinDr-Mammo, 398 mammograms, missing embeddings = 0
  Fusion V2  AUC 0.6319
  Fusion V3  AUC 0.6549

Website (python app.py):
  upload JPG / PNG / DICOM
  → same CLAHE path for DICOM
  → Fusion V3 if present, else Fusion V2
  → probability at threshold 0.5
  → Grad-CAM++, 15-pass dropout uncertainty, four norm shares
  → row in data/scans.json
  → optional chat (Groq, then Ollama) that only sees those numbers
```

Training of each fusion head uses label-matched unpaired images and modality dropout (sections 15–16).

---

## Recorded test results

“AUC” in this file and on the website is the **test** AUC. “Best val AUC” is the best validation AUC stored during training. They are different numbers. Sensitivity and specificity use the threshold in the Threshold column.

| Model | Test AUC | Sens. | Spec. | Accuracy | Best val AUC | Threshold | N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CBIS encoder | 0.7434 | 71.97% | 63.25% | 66.8% | 0.7763 | 0.50 | 645 |
| RSNA encoder | 0.6817 | 61.49% | 67.77% | 66.03% | 0.6471 | 0.70 | 627 |
| BUSI encoder | 0.9647 | 83.87% | 96.51% | 93.2% | 0.9813 | 0.50 | 117 |
| Duke MRI | 0.5679 | 88.89% | 22.22% | 35.56% | 0.7873 | 0.55 | 45 |
| Fusion V1 | 0.9903 | 95.83% | 99.21% | 97.8% | 1.0 | 0.50 | 645 |
| Fusion V2 | 0.9913 | 96.59% | 99.48% | 98.3% | 0.99997 | 0.50 | not stored |
| **Fusion V3** | **0.9897** | **96.97%** | **100.00%** | **98.76%** | **1.0** | **0.50** | **645** |

### Confusion matrices (TP / FP / FN / TN)

| Model | TP | FP | FN | TN | Source file |
| --- | ---: | ---: | ---: | ---: | --- |
| CBIS | 190 | 140 | 74 | 241 | `training_metrics.json` |
| RSNA | 107 | 146 | 67 | 307 | `rsna_metrics.json` |
| BUSI | 26 | 3 | 5 | 83 | `busi_metrics.json` |
| Duke MRI | 8 | 28 | 1 | 8 | `mri_metrics.json` |
| Fusion V1 | 253 | 3 | 11 | 378 | `fusion_metrics.json` |
| Fusion V2 | — | — | — | — | matrix not recoverable |
| Fusion V3 | 256 | 0 | 8 | 381 | `fusion_v3_metrics.json` |

Accuracy is $(\mathrm{TP}+\mathrm{TN})/N$, except Fusion V2, where $98.3\%$ is the value stored in the JSON.

### External test — VinDr-Mammo (n = 398, no retraining)

Ultrasound and MRI embeddings are set to zero. The threshold is Youden’s $J$ on this same set.

| Metric | Fusion V2 | Fusion V3 |
| --- | ---: | ---: |
| AUC | 0.6319 | 0.6549 |
| Sensitivity | 35.86% | 56.57% |
| Specificity | 79.50% | 69.50% |
| Accuracy | 57.79% | 63.07% |
| Threshold | 0.200 | 0.498 |
| TP / FP / FN / TN | 71 / 41 / 127 / 159 | 112 / 61 / 86 / 139 |

The internal fusion test pairs each official CBIS test mammogram with a random image of the **same label** from the other datasets. Those images are not from the same patient. That is why internal fusion AUC is about $0.99$ and the VinDr mammography-only result is about $0.65$.

---

## What each model is

### Positive class

| Dataset | Label $1$ | Label $0$ |
| --- | --- | --- |
| CBIS, RSNA, BUSI | malignant | benign (BUSI folder `normal` is also $0$) |
| Duke MRI | triple-negative | other subtypes (**not** malignant-vs-benign; fusion still uses it as the positive proxy) |

### Checkpoints shipped in `models/`

| File | Role | Test AUC |
| --- | --- | ---: |
| `best_model.pth` | CBIS-DDSM encoder | 0.7434 |
| `rsna_model.pth` | RSNA encoder | 0.6817 |
| `busi_model.pth` | BUSI ultrasound encoder | 0.9647 |
| `mri_model.pth` | Duke MRI encoder | 0.5679 |
| `fusion_v3_model.pth` | Fusion V3 head (**preferred**) | 0.9897 |
| `fusion_v2_model.pth` | Fusion V2 head (fallback) | 0.9913 |

`fusion_model.pth` (Fusion V1) is not loaded by the website.

Checkpoint files also store a **validation** AUC. That value is **not** the number shown as AUC on the site:

| File | Checkpoint `val_auc` | Epoch field |
| --- | ---: | ---: |
| `best_model.pth` | 0.7763 | 12 |
| `rsna_model.pth` | 0.64713 | 13 |
| `busi_model.pth` | 0.98125 | 28 |
| `mri_model.pth` | 0.7873 | 1 |
| `fusion_v2_model.pth` | 0.99997 | 2 |
| `fusion_v3_model.pth` | 1.0 | 10 |

MRI checkpoint task string: **“Triple Negative vs Other”**.

---

## Datasets

### CBIS-DDSM

- Full mammograms only. ROI masks and crops are skipped for encoder training.
- `BENIGN` and `BENIGN_WITHOUT_CALLBACK` are both label $0$.
- $3{,}103$ images in `dataset.csv`:

| Split | Images | Benign | Malignant | Mass | Calcification |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official train | 2,458 | 1,353 | 1,105 | 1,231 | 1,227 |
| Official test | 645 | 381 | 264 | 361 | 284 |
| All | 3,103 | 1,734 | 1,369 | 1,592 | 1,511 |

Encoder validation is 15% of official-train patients. The official 645-image test set is held out. DICOM loading inverts `MONOCHROME1`, scales to $0$–$255$, then CLAHE (`clipLimit 2.0`, tile $8\times 8$) and saves $224\times 224$ RGB PNG.

### BUSI ultrasound

- Folders: `benign` and `normal` → $0$, `malignant` → $1$. Mask files are skipped.
- Stratified image split $70\% / 15\% / 15\%$, seed $42$.
- Test matrix has $117$ images.

### RSNA screening mammography

- Filename `{patient_id}_{image_id}.png`. Label is the `cancer` column.
- Full CSV ~$54{,}706$ images with ~$1{,}158$ cancers.
- Encoder trained on a subsample: up to $1{,}158$ cancers + up to $3{,}000$ normals, seed $42$.
- Reported test set = held-out slice of that subsample ($627$ images), not the full RSNA set.
- Patient-level split; weighted sampler; a separate $3{,}000$-image RSNA cap is used only while training the fusion head.

### Duke Breast Cancer MRI

- Listed in `mri_dataset_expanded.csv`.
- Training script is not shipped with this repo. Metrics: `mri_metrics.json`. Test set size $45$.

### VinDr-Mammo

- $398$ mammograms. Used **only** for the external test (no retraining).

---

## How training was done

Shared settings unless a model below says otherwise:

| Setting | Value |
| --- | --- |
| Backbone | EfficientNet-B0, ImageNet-1K weights |
| Encoder head | `Dropout(0.3)` then `Linear(1280 → 1)` |
| Loss | `BCEWithLogitsLoss` |
| Class weight | $\mathrm{benign}/\mathrm{malignant}$ on the training split |
| Optimizer | AdamW, weight decay $10^{-4}$ |
| Schedule | cosine annealing over the planned epoch count |
| Checkpoint | weights with the best validation AUC |
| Input | $224\times 224$ RGB, ImageNet mean/std |
| Augmentation | H/V flip, rotation $\pm 15^{\circ}$, brightness/contrast jitter $0.2$ |
| Seed | $42$ |

| Model | Script | Batch | Max epochs | LR | Patience | Epochs run |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CBIS | `step3_train.py` | 16 | 40 | $10^{-4}$ | 10 | 22 |
| BUSI | `busi_train.py` | 16 | 40 | $10^{-4}$ | 10 | 38 |
| RSNA | `rsna_train.py` | 16 | 40 | $3\times 10^{-5}$ | 12 | 25 |
| Fusion V1 | `fusion_train.py` | — | 30 | $5\times 10^{-4}$ | 8 | 10 |
| Fusion V2 | `fusion_v2_train.py` | — | 30 | $5\times 10^{-4}$ | 8 | history lost |
| Fusion V3 | `fusion_v3_train.py` | — | 35 | $5\times 10^{-4}$ | 10 | 21 |

The fusion head does **not** update the encoders. Embeddings come from `features` + `avgpool` (1280 values). The optimizer trains only the MLP.

### Fusion V3 training curve (21 epochs logged)

| Epoch | Train loss | Val loss | Val AUC |
| ---: | ---: | ---: | ---: |
| 1 | 0.1454 | 0.0196 | 0.9998 |
| 2 | 0.0965 | 0.0407 | 0.9997 |
| 3 | 0.0984 | 0.0385 | 0.9991 |
| 4 | 0.0578 | 0.0286 | 0.9998 |
| 5 | 0.0906 | 0.0184 | 0.9999 |
| 6 | 0.0760 | 0.0271 | 0.9996 |
| 7 | 0.0598 | 0.0450 | 0.9998 |
| 8 | 0.0658 | 0.0206 | 0.9998 |
| 9 | 0.0455 | 0.0317 | 0.9994 |
| 10 | 0.0827 | 0.0164 | 0.9999 |
| 11 | 0.0734 | 0.0076 | **1.0000** |
| 12 | 0.0686 | 0.0431 | 0.9990 |
| 13 | 0.0695 | 0.0456 | 0.9991 |
| 14 | 0.0579 | 0.0127 | 0.9999 |
| 15 | 0.0661 | 0.0387 | 0.9995 |
| 16 | 0.0633 | 0.0508 | 0.9995 |
| 17 | 0.0616 | 0.0550 | 0.9993 |
| 18 | 0.0503 | 0.0508 | 0.9996 |
| 19 | 0.0408 | 0.0476 | 0.9992 |
| 20 | 0.0411 | 0.0772 | 0.9990 |
| 21 | 0.0334 | 0.0182 | **1.0000** |

Best validation AUC is $1.0$ at epochs 11 and 21. The **test** AUC of this same checkpoint is **$0.9897$**.

The fusion trainers delete the previous checkpoint and metrics file when they start. Fusion V2’s original metrics file was deleted and later restored from recorded scalars; its confusion matrix and epoch history are gone.

---

## What the website does

One Flask process serves `frontend/index.html` and the `/api` routes. Login uses a JWT kept in the browser.

Upload JPG, PNG, or DICOM. The server runs the active fusion model and stores prediction, probability, confidence, uncertainty, risk band, a Grad-CAM++ image, and four embedding-norm shares (CBIS, RSNA, BUSI, MRI). A chat box can explain those numbers. The language model does **not** see the pixels and does **not** change the classifier output.

### Which checkpoint runs

1. **Fusion V3** when `fusion_v3_model.pth` is present (missing encoders → zero embeddings)
2. **Fusion V2** when V3 is absent and `fusion_v2_model.pth` is present
3. The **CBIS encoder alone** when no fusion head is present
4. Fixed demo outputs when nothing loads (benign, probability $28.4$, label “Demo mode”)

- Runtime decision threshold: $0.5$
- Risk bands: $<40\%$ low, $40$–$70\%$ medium, $>70\%$ high (product settings — not the Youden thresholds)

### Explainers attached to each scan

| Explainer | Detail |
| --- | --- |
| Grad-CAM++ | last conv block of the first available encoder (order CBIS, MRI, RSNA, BUSI) |
| MC dropout | 15 stochastic passes of the fusion MLP; uncertainty = std of those probabilities |
| Norm shares | L2 norm of each 1280-d embedding / sum (fields named `shap_*`, not SHAP) |

Chat uses Groq when `GROQ_API_KEY`, `GROQ_API_URL`, and `GROQ_MODEL` are set. If Groq is missing or the call fails, and Ollama is enabled, the same prompt goes to the local model. Generation settings: `max_tokens = 1024`, `temperature = 0.6`.

---

## Roles

| Role | Capabilities |
| --- | --- |
| Patient | upload, own history, plain-language chat, general health chat, request a role upgrade |
| Radiologist | own scans, appointments, queue and 7-day throughput, clinical wording in chat |
| Clinic | patient registry, analytics, triage wording, schedule |
| Admin | users, role changes, role-request approval, audit log, system dashboard, choice of chat style |

The only built-in account is created from `.env` on first run (`ADMIN_EMAIL`, `ADMIN_PASSWORD`). Everyone else registers as a patient.

---

## Inference steps

1. `POST /api/scan/analyze` with the file and the JWT  
2. DICOM: pydicom, `MONOCHROME1` inversion, CLAHE · JPG/PNG: Pillow RGB · AVIF rejected  
3. Models load on the first request (CUDA if available, else CPU)  
4. Resize to $224\times 224$ and apply ImageNet normalization  
5. Up to four 1280-d embeddings  
6. Fusion V3 concatenates them to $5120$ and applies the MLP, then sigmoid · Fusion V2 uses three embeddings ($3840$) and skips MRI  
7. 15 dropout passes for the uncertainty figure  
8. Norm shares and Grad-CAM++  
9. Save the row in `data/scans.json`, including `shap_mri` and `model_used`

---

## Folder layout

```text
Scanova/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
├── data/                          # JSON tables used at runtime
├── database/mysql_schema.sql      # unused schema (includes shap_mri)
├── models/                        # the six .pth files
├── frontend/index.html
└── backend/
    ├── config.py
    ├── inference/engine.py        # model catalog and run_inference()
    ├── services/llm_service.py    # Groq, then Ollama
    ├── services/prompt_service.py
    ├── services/image_service.py
    └── routes/                    # auth, scans, chat, clinic, admin, health
```

Training code (not imported by Flask) historically lived in a separate `dev_part1` tree: `step*_*.py`, `busi_train.py`, `rsna_train.py`, `fusion_*_train.py`, `vindr_validate*.py`, and `*_metrics.json`.

### Setup checklist

1. Place the six `.pth` files in `models/`
2. `copy .env.example .env` — set `SCANOVA_SECRET`, `ADMIN_PASSWORD`, and either Groq or Ollama
3. Create venv → `pip install -r requirements.txt`
4. `python app.py`
5. Open `http://localhost:5000`

The startup banner prints OK / MISSING for CBIS, RSNA, BUSI, MRI, Fusion V3, and Fusion V2, then Groq and Ollama status.

> Do **not** rerun a fusion trainer if you still need the current checkpoint: the script deletes the previous `.pth` and metrics file at start.

---

## Environment

Secrets and the LLM model name are read from `.env` only.

| Variable | Purpose |
| --- | --- |
| `FLASK_HOST`, `FLASK_PORT`, `FLASK_DEBUG` | server bind |
| `ALLOW_INSECURE_DEFAULTS` | local-dev only |
| `SCANOVA_SECRET` | JWT key (≥ 32 random characters) |
| `JWT_ALGORITHM`, `JWT_TOKEN_HOURS` | token settings |
| `ADMIN_EMAIL`, `ADMIN_PASSWORD`, `ADMIN_NAME` | seed admin |
| `GROQ_API_KEY`, `GROQ_API_URL`, `GROQ_MODEL` | cloud chat |
| `OLLAMA_ENABLED`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | local chat fallback |
| `CORS_ORIGINS` | allowed origins |
| `MAX_UPLOAD_MB` | default 16 |
| `MIN_PASSWORD_LENGTH` | default 8 |
| `RATE_LIMIT_LOGIN` | default 10 / 15 min / IP |
| `STORAGE_BACKEND` | `json` |
| `DATA_DIR`, `MODELS_DIR`, `FRONTEND_DIR` | paths |

Generate a secret:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

---

## Security and storage

- bcrypt password hashes (`bcrypt==4.0.1` for passlib)
- JWT on protected routes
- four roles, optional TOTP
- login rate limit
- CORS from the configured list
- upload size and extension limits (`jpg`, `png`, `dcm`)
- basenames only on stored filenames
- startup warning list for weak secrets

Runtime data is JSON under `data/`:

`users`, `scans`, `patients`, `appointments`, `role_requests`, `admin_logs`, `scan_chats`, `direct_chat_sessions`, `direct_chat_messages`

MySQL is described in `database/mysql_schema.sql` and is not switched on.

---

## API

### Auth / MFA

| Method | Path |
| --- | --- |
| `POST` | `/api/auth/login` |
| `POST` | `/api/auth/register` |
| `GET` | `/api/auth/me` |
| `PUT` | `/api/auth/profile` |
| `PUT` | `/api/auth/password` |
| `GET` | `/api/auth/mfa/status` |
| `POST` | `/api/auth/mfa/start` |
| `POST` | `/api/auth/mfa/verify` |
| `POST` | `/api/auth/mfa/disable` |

### Scans / chat

| Method | Path |
| --- | --- |
| `POST` | `/api/scan/analyze` |
| `POST` | `/api/scan/chat` |
| `POST` | `/api/chat/general` |
| `GET` | `/api/scans` |
| `GET` | `/api/scans/<id>` |
| `PATCH` | `/api/scans/<id>/label` |
| `DELETE` | `/api/scans/<id>` |
| `GET` | `/api/scans/<id>/chats` |
| `DELETE` | `/api/scans/<id>/chats` |

### Direct chat / clinic / admin

| Method | Path |
| --- | --- |
| `GET`/`POST` | `/api/direct-chats` |
| `GET`/`POST`/`PATCH`/`DELETE` | `/api/direct-chats/<sid>` … |
| `GET` | `/api/dashboard/stats` |
| `GET`/`POST`/`DELETE` | `/api/patients` … |
| `GET`/`POST`/`PUT`/`DELETE` | `/api/schedule` … |
| `GET`/`DELETE`/`PUT` | `/api/users` … |
| `GET`/`POST`/`PUT` | `/api/role-requests` … |
| `GET` | `/api/admin-logs` |
| `GET` | `/api/health` |
| `GET` | `/api/models/info` |

`/api/models/info` returns `test_auc`, sensitivity, specificity, accuracy, plus checkpoint `val_auc` and which model will run.

---

## Pages

| Page | Content |
| --- | --- |
| Landing | Fusion V3 test AUC $0.9897$, accuracy $98.76\%$, specificity $100\%$; comparison for all six models |
| Diagnostic lens | multi-file upload, risk badge, four contribution bars, heatmap |
| Scan detail | metrics, heatmap, per-scan chat, rename, delete |
| Direct chat | saved sessions, role-conditioned answers |
| Scheduling | mammography, ultrasound, and breast MRI appointments |
| Patients | clinic and admin registry |
| Users | admin list, role change, detail export |
| Analytics | the same recorded test table |
| Settings | name, email, password, MFA |
| Admin home | counts, which checkpoints exist, audit log |

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| AVIF rejected | Convert to JPG or PNG |
| Page looks stale | Restart `python app.py` and hard-refresh the browser |
| Chat unavailable | Set the three `GROQ_*` variables, or run Ollama with the tag in `OLLAMA_MODEL` |
| Result label says Fusion V2 | `fusion_v3_model.pth` is not in `models/` — trust `GET /api/models/info` field `active` |
| A model line prints MISSING | That file is not in `models/`; Fusion V3 can still run with zero embeddings for the missing slot |
| CUDA OOM while training | Lower batch size from 16 to 8 in that training script |
| passlib / bcrypt warning | `requirements.txt` pins `bcrypt==4.0.1` on purpose |

---

## References

| Topic | Citation |
| --- | --- |
| EfficientNet | Tan and Le, ICML 2019 |
| Grad-CAM++ | Chattopadhay, Sarkar, Howlader, and Balasubramanian, IEEE WACV 2018 |
| MC dropout | Gal and Ghahramani, ICML 2016 |
| Datasets | CBIS-DDSM; RSNA Screening Mammography (Kaggle); BUSI ultrasound; Duke Breast Cancer MRI; VinDr-Mammo |

---

**Scanova is a research prototype. It does not replace a radiologist’s reading or a pathology result.**
