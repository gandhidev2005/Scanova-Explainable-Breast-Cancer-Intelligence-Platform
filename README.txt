Scanova — Breast Cancer AI
==========================

Scanova is a breast-imaging decision-support system. Four EfficientNet-B0
encoders (CBIS-DDSM mammography, RSNA mammography, BUSI ultrasound, Duke
breast MRI) are trained separately, then frozen. A fusion MLP reads their
1280-dimensional embeddings and outputs one malignancy probability.

Fusion V3 (all four encoders) is the model the website runs. Fusion V2
(no MRI) is the fallback if fusion_v3_model.pth is missing.

Training scripts and the metric JSON files live in G:\dev_part1.
This repository is the Flask site that serves the trained weights.

Every AUC, sensitivity, specificity, and accuracy figure below is taken
from those JSON files. The website shows the same test-set numbers.

The full list of algorithms is in the next section. Training code is in
G:\dev_part1. The website copies of preprocessing, fusion, Grad-CAM++,
and Monte Carlo dropout are in backend/services/image_service.py and
backend/inference/engine.py.


═══════════════════════════════════════════════════
ALGORITHMS USED
═══════════════════════════════════════════════════

These are the methods in the training scripts (G:\dev_part1) and in the
website (backend/inference/engine.py, backend/services/image_service.py).
Formulas below match that code. A comment at the top of fusion_v3_train.py
says the MRI encoder AUC is 0.6461. The saved test file mri_metrics.json
says 0.5679, and that is the number used everywhere else in this README.
The MRI training script itself is not in the folder. There is no
4-class cross-entropy stage in the saved pipeline: every training script
that is present uses BCEWithLogitsLoss. Storage in the running site is
JSON, not SQLite.


1. EfficientNet-B0
──────────────────
Backbone for every encoder. This project uses the torchvision B0 network
with ImageNet-1K V1 weights. It does not re-search the compound
coefficients. Those coefficients are the published rule that produced B0:

  depth      d = α^φ
  width      w = β^φ
  resolution r = γ^φ
  with α · β² · γ² ≈ 2, and α, β, γ ≥ 1

The paper's baseline (B0) is φ = 0, so d = w = r = 1, with the searched
constants α = 1.2, β = 1.1, γ = 1.15 used only when φ > 0 (B1 and up).

Each MBConv block is an inverted residual:

  expand by factor t (6 in most blocks)
  → depthwise 3×3 convolution
  → squeeze-excitation
  → project back to C_out
  → add the input when the shape matches

Depthwise-separable cost versus a full convolution of kernel K:

  full:        H · W · C_in · C_out · K²
  separable:   H · W · C_in · K²  +  H · W · C_in · C_out

The ratio of separable cost to full cost is about 1/C_out + 1/K².
For K = 3 that second term is 1/9. It is not a flat 9× reduction,
because the pointwise 1×1 convolution is still there.

Squeeze-excitation inside the block, reduction r = 4:

  global average pool
  → linear C → C/r → ReLU
  → linear C/r → C → sigmoid
  → multiply those gates onto the channels


2. Transfer learning
────────────────────
Encoder training (step3_train.py, busi_train.py, rsna_train.py):

  start from ImageNet-1K V1 weights
  replace the 1000-way classifier with Dropout(0.3) + Linear(1280 → 1)
  train every parameter with AdamW

Nothing is frozen during encoder training. The low layers still begin
at filters that already detect edges and texture, and training moves
them toward mammography or ultrasound.

When a finished encoder is loaded for fusion (fusion_v3_train.py and
the website):

  every weight is frozen (requires_grad = False)
  the last feature block, features[-1], is marked requires_grad = True
  so Grad-CAM can read its gradients
  the optimizer still updates only the fusion MLP

Embeddings are taken under torch.no_grad, so that last block is not
actually updated during fusion training.


3. Binary classification
────────────────────────
Each encoder maps a 224×224×3 image to one probability:

  f(x) = σ(wᵀ φ(x) + b)
  σ(z) = 1 / (1 + e^(−z))

φ(x) is the 1280-d embedding. The decision is

  ŷ = 1 if f(x) ≥ θ, else 0

θ = 0.5 for CBIS, BUSI, all three fusion heads, and every prediction
the website returns. RSNA's reported test point uses θ = 0.70.
The saved MRI test point uses θ = 0.55.
Labels: malignant = 1 and benign = 0, except Duke MRI, where
triple-negative = 1 and other subtypes = 0.


4. BCEWithLogitsLoss
────────────────────
Loss on the raw logit ẑ, label y ∈ {0, 1}, positive weight w:

  L = −[ w · y · log σ(ẑ) + (1 − y) · log(1 − σ(ẑ)) ]

PyTorch computes the log-sigmoid in a stable way:

  log σ(ẑ) = −log(1 + e^(−ẑ))

With w = 1 the gradient is σ(ẑ) − y.

w is not a single stored constant, except on RSNA:

  CBIS, BUSI, and every fusion head:
      w = (benign count on that training split) / (malignant count)
      computed when the script runs; the value is printed, not saved
  RSNA encoder:
      w = 2.0 exactly
      (rsna_train.py: the sampler already balances the batches)


5. AdamW
────────
Scripts call torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
and leave the other arguments at the PyTorch defaults
β1 = 0.9, β2 = 0.999, ε = 1e-8.

  m_t  = β1 m_(t−1) + (1−β1) g_t
  v_t  = β2 v_(t−1) + (1−β2) g_t²
  m̂_t = m_t / (1 − β1^t)
  v̂_t = v_t / (1 − β2^t)
  θ   ← θ − α · m̂_t / (√v̂_t + ε) − α · λ · θ

λ = 1e-4 is applied to the weights themselves, not added into g_t.
That is the AdamW difference from Adam.

Learning rates in the scripts:

  CBIS, BUSI          α = 1e-4
  RSNA                α = 3e-5
  Fusion V1, V2, V3   α = 5e-4


6. CosineAnnealingLR
────────────────────
  η_t = η_min + ½ (η_max − η_min) (1 + cos(π t / T_max))

The scripts set T_max to the planned epoch count and leave η_min at 0.
η_max is the learning rate above. The decay is smooth across the run
rather than a step drop.


7. Data augmentation and normalization
──────────────────────────────────────
Training transforms in the scripts:

  resize to 224×224
  horizontal flip, p = 0.5
  vertical flip, p = 0.5
  rotation uniform in ±15°
  brightness and contrast jitter 0.2
      brightness: x' = x + δ, δ in [−0.2, 0.2]
      contrast:   x' = α (x − μ) + μ, α in [0.8, 1.2]

Then, in the tensor transform:

  x'_c = (x_c − μ_c) / σ_c
  μ = [0.485, 0.456, 0.406]
  σ = [0.229, 0.224, 0.225]

Validation and test use resize and this normalization only.


8. Early stopping
─────────────────
  best validation ROC-AUC starts at 0
  if this epoch's validation AUC is strictly greater:
      save the checkpoint and reset the stall counter
  else:
      increment the stall counter
      stop when the counter reaches patience, and reload the best file

Patience: CBIS 10, BUSI 10, RSNA 12, Fusion V1 8, Fusion V2 8,
Fusion V3 10. The MRI script is not in the folder. Its saved history
is 14 epochs and the best validation AUC is at the 2nd logged epoch.


9. Splits
─────────
Patient-level, CBIS encoder (step3_train.py):
  official test set stays held out (645 images)
  validation is 15% of official-train patients
  patient id is the P_##### token in the path
  every image of one patient stays in one split

Patient-level, RSNA encoder:
  each patient is labeled by the max cancer value across that patient's images
  70% of patients train; the other 30% is split evenly into val and test
  seed 42

Stratified image split, BUSI, and the RSNA / BUSI / MRI pools inside
fusion training:
  train_test_split(..., stratify=labels, random_state=42)
  70% / 15% / 15%

Fusion's own CBIS cut is different from the encoder cut: the official
test is still held out, but the 15% validation slice is a row shuffle,
not the patient split from step3.


10. WeightedRandomSampler
─────────────────────────
Used only by rsna_train.py.

  w_c = 1 / n_c
  P(draw sample i) = w_(y_i) / Σ_j w_(y_j)
  num_samples = length of the training set
  replacement = True

The training comment says each batch is about half cancer and half
normal. This is the same algorithm as inverse-frequency sampling.
It is not a second, different method. The loss weight on top of it
is the fixed 2.0 from section 4, not a second inverse-frequency weight.


11. CLAHE and DICOM loading
───────────────────────────
In step2_preprocess.py and in backend/services/image_service.py:

  read the DICOM pixel array
  if PhotometricInterpretation is MONOCHROME1: arr = max(arr) − arr
  min-max scale to 0–255
  CLAHE with clipLimit = 2.0 and tile grid 8×8
  convert to RGB

CLAHE equalizes the histogram inside each tile, after clipping bin
counts so a nearly empty background does not get stretched into noise.
JPG and PNG uploads skip CLAHE and are only converted to RGB.


12. Embedding extraction
────────────────────────
  φ(x) = flatten(avgpool(encoder.features(x))) ∈ R^1280

features() returns a 7×7×1280 map. Global average pooling is

  GAP(A_k) = (1 / (H · W)) Σ_(i,j) A_k(i, j)

The website and the fusion trainers both use this function. The
classifier head is not part of the embedding.


13. Late fusion by concatenation
────────────────────────────────
  z = [z_CBIS ; z_RSNA ; z_BUSI ; z_MRI] ∈ R^5120     (V3)
  z = [z_CBIS ; z_RSNA ; z_BUSI]          ∈ R^3840     (V2)
  z = [z_CBIS ; z_BUSI]                   ∈ R^2560     (V1)

Each z_* is 1280-d. Concatenation keeps every coordinate. The MLP
then learns the mix. No cross-attention layer is in the code.


14. Fusion MLP
──────────────
ReLU between linear layers. Dropout after each ReLU except the last
linear, which is a single logit (sigmoid is applied in the loss or
at inference, not inside the module).

  V1    2560 → 512 → 128 → 1       dropout 0.4, 0.3
  V2    3840 → 1024 → 256 → 64 → 1 dropout 0.4, 0.3, 0.2
  V3    5120 → 1024 → 256 → 64 → 1 dropout 0.4, 0.3, 0.2

V3, written out:

  h1 = ReLU(W1 z + b1),   W1 is 1024 × 5120
  h2 = ReLU(W2 h1 + b2),  W2 is 256 × 1024
  h3 = ReLU(W3 h2 + b3),  W3 is 64 × 256
  ŷ  = σ(W4 h3 + b4),     W4 is 1 × 64

The website's FusionHead class is this same V2/V3 MLP. It does not
load the V1 file.


15. Modality dropout
────────────────────
Training only. DROPOUT_PROB = 0.5 in all three fusion scripts.

With probability 0.5, draw r uniform on [0, 1) and zero exactly one
embedding. Otherwise leave all of them intact.

Fusion V3 (fusion_v3_train.py):

  if r < 0.25:  zero CBIS
  elif r < 0.50: zero RSNA
  elif r < 0.75: zero BUSI
  else:          zero MRI

Fusion V2 uses the same 0.5 gate and then thirds (0.33 / 0.66) over
CBIS, RSNA, and BUSI. Fusion V1 flips a coin between mammography and
ultrasound.

At inference the website does not apply this dropout. A missing file
is a zero embedding for that slot, which is the case the training
dropout was meant to survive. VinDr zeros BUSI (and MRI, for V3)
for the same reason.


16. Unpaired label-matched sampling
───────────────────────────────────
FusionDatasetV3.__getitem__ (and the V1/V2 datasets):

  take the next CBIS row; its label is the label of the whole group
  draw a random RSNA row with that label
  draw a random BUSI row with that label
  draw a random MRI row with that label (V3 only)
  if a same-label pool is empty, draw from the other pool

The four images are not the same patient. The shared label is what
keeps a malignant mammogram from being paired with a benign ultrasound.


17. Grad-CAM++
──────────────
Library: pytorch_grad_cam.GradCAMPlusPlus.
Target layer: encoder.features[-1], the last MBConv block.
The website tries encoders in order CBIS, MRI, RSNA, BUSI and uses
the first one that loaded.

For feature map A^k and class score S^c, Grad-CAM++ builds a
pixel-wise weight from the second derivative of S^c with respect to
A^k, then

  L^c = ReLU( Σ_k α_k^c · A^k )

The single-logit head is indexed as ClassifierOutputTarget(0). That
index is "the only logit", not "class benign".

The map is min-max scaled, colored with the jet colormap, and returned
as a base64 PNG.


18. Monte Carlo dropout
───────────────────────
Website only, after the fusion probability has already been computed
in eval mode. That eval-mode sigmoid is the stored probability.
It is not the average of the dropout samples.

Then, for T = 15:

  turn every Dropout module back to train mode
  p_t = σ(fusion(z_CBIS, z_RSNA, z_BUSI, z_MRI))
  uncertainty = standard deviation of {p_1 … p_15}

The UI shows that standard deviation times 100. If no fusion head
loaded, the placeholder standard deviation is 0.03.

This is the Gal and Ghahramani (2016) use of dropout as a stand-in
for a posterior over the fusion-head weights. It is an epistemic
spread for the head, not a calibrated confidence interval, and it
does not measure noise in the image itself.


19. L2 norm share
─────────────────
  ||z||_2 = sqrt(Σ_i z_i²)
  C_k = ||z_k||_2 / (||z_CBIS||_2 + ||z_RSNA||_2 + ||z_BUSI||_2 + ||z_MRI||_2)

The four percentages are what the contribution bars show. The JSON
fields are named shap_cbis, shap_rsna, shap_us, shap_mri. They are
these ratios, not SHAP values. A zeroed embedding has norm 0, so a
missing modality shows as 0%. A large activation vector shows as a
larger share. Magnitude is not the same thing as causal effect on
the logit.


20. Youden's J
──────────────
  J(θ) = sensitivity(θ) + specificity(θ) − 1
       = TPR(θ) − FPR(θ)

  θ* = argmax_θ J(θ)

rsna_train.py computes it as:

  fpr, tpr, thresholds = roc_curve(y_true, y_prob)
  j_scores = tpr − fpr
  θ* = thresholds[argmax(j_scores)]

Recorded operating points from the JSON files:

  RSNA test            0.70
  Duke MRI test        0.55
  VinDr, Fusion V2     0.200     (fit on those 398 images)
  VinDr, Fusion V3     0.498     (fit on those 398 images)

The website does not use these. It thresholds fusion output at 0.5.


21. Test metrics
────────────────
  ROC-AUC          ranking quality; does not depend on θ
  sensitivity      TP / (TP + FN)
  specificity      TN / (TN + FP)
  accuracy         (TP + TN) / N

The numbers for each model are in the next section and in the JSON
files named there. Fusion V2's confusion counts were not recovered,
so its accuracy is the 98.3 stored in fusion_v2_metrics.json rather
than a recomputed (TP+TN)/N.


22. Account and session methods
───────────────────────────────
These are part of the website, not the imaging model.

  bcrypt          password hash (passlib, bcrypt pinned to 4.0.1)
  JWT HS256       login token, lifetime JWT_TOKEN_HOURS (default 24)
  TOTP            MFA, pyotp, verify window of one step
  fixed window    login limited to 10 attempts per 15 minutes per IP
                  (RATE_LIMIT_LOGIN, RATE_LIMIT_WINDOW_SEC)


═══════════════════════════════════════════════════
PROJECT FLOW
═══════════════════════════════════════════════════

  CBIS-DDSM DICOM ── step1 CSV ── step2 CLAHE PNG ── step3_train.py
                                                         │
                                                         ▼
                                                   best_model.pth
                                                   test AUC 0.7434

  BUSI PNG ── busi_train.py ── busi_model.pth     test AUC 0.9647
  RSNA PNG ── rsna_train.py ── rsna_model.pth     test AUC 0.6817
  Duke MRI ── (script not in the folder)
              mri_model.pth                       test AUC 0.5679

  Those four checkpoints are frozen.
  Each image becomes a 1280-d embedding (section 12).

  Fusion V1   CBIS+BUSI            2560-d   test AUC 0.9903   acc 97.8%
  Fusion V2   CBIS+RSNA+BUSI       3840-d   test AUC 0.9913   acc 98.3%
  Fusion V3   CBIS+RSNA+BUSI+MRI   5120-d   test AUC 0.9897   acc 98.76%
              specificity 100% (FP = 0), sensitivity 96.97%, n = 645

  Training of each fusion head uses label-matched unpaired images
  and modality dropout (sections 15 and 16).

  External check, no retraining:
    VinDr-Mammo, 398 mammograms, missing embeddings set to 0
    Fusion V2  AUC 0.6319
    Fusion V3  AUC 0.6549

  Website, python app.py:
    upload JPG / PNG / DICOM
    → same CLAHE path for DICOM
    → Fusion V3 if fusion_v3_model.pth is present, else Fusion V2
    → probability at threshold 0.5
    → Grad-CAM++, 15-pass dropout uncertainty, four norm shares
    → row in data/scans.json
    → optional chat (Groq, then Ollama) that only sees those numbers

Gradio demos in G:\dev_part1, separate from Flask:
  step4_app.py, rsna_demo.py, fusion_demo.py, fusion_v2_demo.py.
  step4_app.py listens on port 7860.


═══════════════════════════════════════════════════
RECORDED TEST RESULTS
═══════════════════════════════════════════════════

"AUC" in this file and on the website is the test AUC. "Best val AUC"
is the best validation AUC stored during training. They are different
numbers. Sensitivity and specificity use the threshold in the Threshold
column.

  Model         Test AUC   Sens.    Spec.    Accuracy   Best val AUC   Threshold   N
  ────────────  ────────   ──────   ──────   ────────   ────────────   ─────────   ───
  CBIS encoder   0.7434    71.97%   63.25%    66.8%      0.7763         0.50       645
  RSNA encoder   0.6817    61.49%   67.77%    66.03%     0.6471         0.70       627
  BUSI encoder   0.9647    83.87%   96.51%    93.2%      0.9813         0.50       117
  Duke MRI       0.5679    88.89%   22.22%    35.56%     0.7873         0.55        45
  Fusion V1      0.9903    95.83%   99.21%    97.8%      1.0            0.50       645
  Fusion V2      0.9913    96.59%   99.48%    98.3%      0.99997        0.50       not stored
  Fusion V3      0.9897    96.97%  100.00%    98.76%     1.0            0.50       645

Confusion matrices (TP / FP / FN / TN):

  Model         TP    FP    FN    TN     File
  ────────────  ────  ────  ────  ────   ─────────────────────────
  CBIS          190   140    74   241    training_metrics.json
  RSNA          107   146    67   307    rsna_metrics.json
  BUSI           26     3     5    83    busi_metrics.json
  Duke MRI        8    28     1     8    mri_metrics.json
  Fusion V1     253     3    11   378    fusion_metrics.json
  Fusion V2     —     —     —     —      fusion_v2_metrics.json
                                         (matrix was not recoverable)
  Fusion V3     256     0     8   381    fusion_v3_metrics.json

Accuracy is (TP + TN) / N, except Fusion V2, where 98.3% is the value
stored in the JSON.

External test, VinDr-Mammo, 398 mammograms, no retraining.
Ultrasound and MRI embeddings are set to zero because VinDr has no
ultrasound or MRI for these cases. The threshold is Youden's J on
this same set.

                        Fusion V2          Fusion V3
  ────────────────────  ─────────────────  ─────────────────
  AUC                   0.6319             0.6549
  Sensitivity           35.86%             56.57%
  Specificity           79.50%             69.50%
  Accuracy              57.79%             63.07%
  Threshold             0.200              0.498
  TP / FP / FN / TN     71 / 41 / 127 / 159   112 / 61 / 86 / 139
  Files                 vindr_validation_results.json
                        vindr_v3_validation_results.json
                        plus the matching CSV score files

The internal fusion test pairs each official CBIS test mammogram with
a random image of the same label from the other datasets. Those images
are not from the same patient. That is why internal fusion AUC is about
0.99 and the VinDr mammography-only result is about 0.65.


═══════════════════════════════════════════════════
WHAT EACH MODEL IS
═══════════════════════════════════════════════════

Positive class:

  CBIS, RSNA, BUSI     malignant = 1, benign = 0
                       BUSI folder "normal" is also 0
  Duke MRI             triple-negative = 1, other subtypes = 0
                       This is not a malignant-versus-benign label.
                       Fusion still uses it as the positive proxy.

Checkpoints shipped in models/:

  File                    Role                         Test AUC
  ──────────────────────  ───────────────────────────  ────────
  best_model.pth          CBIS-DDSM encoder            0.7434
  rsna_model.pth          RSNA encoder                 0.6817
  busi_model.pth          BUSI ultrasound encoder      0.9647
  mri_model.pth           Duke MRI encoder             0.5679
  fusion_v3_model.pth     Fusion V3 head (preferred)   0.9897
  fusion_v2_model.pth     Fusion V2 head (fallback)    0.9913

fusion_model.pth (Fusion V1) stays in G:\dev_part1. The website does
not load it.

Checkpoint files also store a validation AUC. That value is not the
number shown as AUC on the site:

  File                    Checkpoint val_auc    Epoch field in the file
  ──────────────────────  ──────────────────    ───────────────────────
  best_model.pth          0.7763                12
  rsna_model.pth          0.64713               13
  busi_model.pth          0.98125               28
  mri_model.pth           0.7873                1
  fusion_v2_model.pth     0.99997               2
  fusion_v3_model.pth     1.0                   10

MRI checkpoint task string: "Triple Negative vs Other".


═══════════════════════════════════════════════════
DATASETS
═══════════════════════════════════════════════════

CBIS-DDSM
  Path used by training:
    G:\DATASET_PART1\manifest-ZkhPvrLo5216730872708713142\CBIS-DDSM
  Full mammograms only. ROI masks and crops are skipped.
  BENIGN and BENIGN_WITHOUT_CALLBACK are both label 0.
  3,103 images in dataset.csv:

    Split             Images   Benign   Malignant   Mass    Calcification
    Official train     2,458    1,353       1,105   1,231           1,227
    Official test        645      381         264     361             284
    All                3,103    1,734       1,369   1,592           1,511

  Encoder validation is 15% of official-train patients (id P_##### in
  the path). The official 645-image test set is held out.
  DICOM loading inverts MONOCHROME1, scales to 0–255, then CLAHE
  (clipLimit 2.0, tile 8×8) and saves 224×224 RGB PNG.

BUSI ultrasound
  Path: G:\archive (1)\Dataset_BUSI_with_GT
  benign and normal → 0, malignant → 1. Mask files are skipped.
  Split is stratified by image, 70% / 15% / 15%, seed 42.
  Test matrix has 117 images.

RSNA screening mammography
  Path: G:\rsna-breast-cancer-512-pngs with train.csv.
  Filename {patient_id}_{image_id}.png. Label is the cancer column.
  The full CSV is about 54,706 images with about 1,158 cancers.
  The encoder was trained on a subsample: up to 1,158 cancers plus
  up to 3,000 normals, seed 42. The reported test set is the held-out
  slice of that subsample (627 images), not the full RSNA set.
  Split is patient-level. Training batches are rebalanced with a
  weighted sampler. A separate 3,000-image RSNA cap is used only
  while training the fusion head.

Duke Breast Cancer MRI
  Images listed in D:\Duke-MRI\mri_dataset_expanded.csv.
  The training script is not in G:\dev_part1. Metrics are in
  mri_metrics.json. Test set size is 45 images.

VinDr-Mammo
  D:\VinDr-Mammo\vindr_val_subset.csv
  Images under D:\VinDr-Mammo\images\{study_id}\{image_id}.dicom
  398 mammograms. Used only for the external test.


═══════════════════════════════════════════════════
HOW TRAINING WAS DONE
═══════════════════════════════════════════════════

Shared settings, unless a model below says otherwise:

  Backbone          EfficientNet-B0, ImageNet-1K weights
  Encoder head      Dropout(0.3) then Linear(1280 → 1)
  Loss              BCEWithLogitsLoss
  Class weight      benign_count / malignant_count on the training split
  Optimizer         AdamW, weight decay 1e-4
  Schedule          cosine annealing over the planned epoch count
  Checkpoint        weights with the best validation AUC
  Input             224×224 RGB, ImageNet mean and std
  Augmentation      horizontal flip, vertical flip, rotation ±15°,
                    brightness and contrast jitter 0.2
  Seed              42

  Model      Script                 Batch   Max epochs   Learning rate   Patience   Epochs run
  ─────────  ─────────────────────  ─────   ──────────   ─────────────   ────────   ──────────
  CBIS       step3_train.py            16           40   1e-4                  10           22
  BUSI       busi_train.py             16           40   1e-4                  10           38
  RSNA       rsna_train.py             16           40   3e-5                  12           25
  Fusion V1  fusion_train.py           —            30   5e-4                   8           10
  Fusion V2  fusion_v2_train.py        —            30   5e-4                   8   history lost
  Fusion V3  fusion_v3_train.py        —            35   5e-4                  10           21

RSNA uses a fixed positive weight of 2.0 because the sampler already
balances batches. RSNA and MRI test thresholds are Youden's J (0.70
and 0.55). Everything else in the internal table uses 0.5.

The fusion head does not update the encoders. Each encoder is loaded
from its checkpoint and the embedding is taken from features + avgpool
(1280 values). The optimizer trains only the MLP. In V2 and V3 the last
convolution block is left trainable so Grad-CAM can run; its weights
are still not in the fusion optimizer.

Each fusion sample starts from one CBIS mammogram. A same-label image
is drawn at random from each other modality. The label of that group
is the CBIS label.

Modality dropout (probability 0.5) replaces one embedding with zeros.
That is what lets a later mammography-only test zero the missing
ultrasound and MRI channels.

Fusion MLP, ReLU between layers:

  V1    2560 → 512 → 128 → 1          dropout 0.4, 0.3
  V2    3840 → 1024 → 256 → 64 → 1    dropout 0.4, 0.3, 0.2
  V3    5120 → 1024 → 256 → 64 → 1    dropout 0.4, 0.3, 0.2

Fusion V3 training curve (21 epochs actually logged):

  Epoch   Train loss   Val loss   Val AUC
      1       0.1454     0.0196    0.9998
      2       0.0965     0.0407    0.9997
      3       0.0984     0.0385    0.9991
      4       0.0578     0.0286    0.9998
      5       0.0906     0.0184    0.9999
      6       0.0760     0.0271    0.9996
      7       0.0598     0.0450    0.9998
      8       0.0658     0.0206    0.9998
      9       0.0455     0.0317    0.9994
     10       0.0827     0.0164    0.9999
     11       0.0734     0.0076    1.0000
     12       0.0686     0.0431    0.9990
     13       0.0695     0.0456    0.9991
     14       0.0579     0.0127    0.9999
     15       0.0661     0.0387    0.9995
     16       0.0633     0.0508    0.9995
     17       0.0616     0.0550    0.9993
     18       0.0503     0.0508    0.9996
     19       0.0408     0.0476    0.9992
     20       0.0411     0.0772    0.9990
     21       0.0334     0.0182    1.0000

Best validation AUC is 1.0 at epochs 11 and 21. The test AUC of this
same checkpoint is 0.9897.

The fusion trainers delete the previous checkpoint and metrics file
when they start. Fusion V2's original metrics file was deleted and
later restored from the recorded scalars. Its confusion matrix and
epoch history are gone. fusion_v2_splits.json still has the image
lists from a later V2 run (seed 42).


═══════════════════════════════════════════════════
WHAT THE WEBSITE DOES
═══════════════════════════════════════════════════

One Flask process serves frontend/index.html and the /api routes.
Login uses a JWT kept in the browser.

Upload JPG, PNG, or DICOM. The server runs the active fusion model
and stores prediction, probability, confidence, uncertainty, risk
band, a Grad-CAM++ image, and four embedding-norm shares (CBIS, RSNA,
BUSI, MRI). A chat box can explain those numbers. The language model
does not see the pixels and does not change the classifier output.

Which checkpoint runs:

  1. Fusion V3 when fusion_v3_model.pth is present
     A missing encoder is replaced with a zero embedding.
  2. Fusion V2 when V3 is absent and fusion_v2_model.pth is present
  3. The CBIS encoder alone when no fusion head is present
  4. Fixed demo outputs when nothing loads
     (benign, probability 28.4, label "Demo mode")

Runtime decision threshold is 0.5.
Risk bands: below 40% low, 40–70% medium, above 70% high.
These bands are product settings. They are not the Youden thresholds
used for the RSNA, MRI, and VinDr rows in the results table.

Explainers attached to each scan:

  Grad-CAM++     last conv block of the first available encoder,
                 in order CBIS, MRI, RSNA, BUSI
  MC dropout     15 stochastic passes of the fusion MLP;
                 uncertainty is the standard deviation of those
                 probabilities
  Norm shares    L2 norm of each 1280-d embedding, divided by the sum
                 The stored field names start with shap_. They are
                 norm shares, not SHAP values.

Chat uses Groq when GROQ_API_KEY, GROQ_API_URL, and GROQ_MODEL are
set. If Groq is missing or the call fails, and Ollama is enabled, the
same prompt goes to the local model. Generation settings are
max_tokens 1024 and temperature 0.6 on both.


═══════════════════════════════════════════════════
ROLES
═══════════════════════════════════════════════════

  Patient        upload, own history, plain-language chat,
                 general health chat, request a role upgrade
  Radiologist    own scans, appointments, queue and 7-day
                 throughput, clinical wording in chat
  Clinic         patient registry, analytics, triage wording, schedule
  Admin          users, role changes, role-request approval,
                 audit log, system dashboard, choice of chat style

The only built-in account is created from .env on first run
(ADMIN_EMAIL, ADMIN_PASSWORD). Everyone else registers as a patient.


═══════════════════════════════════════════════════
INFERENCE STEPS
═══════════════════════════════════════════════════

  1. POST /api/scan/analyze with the file and the JWT
  2. DICOM: pydicom, MONOCHROME1 inversion, CLAHE
     JPG/PNG: Pillow RGB
     AVIF is rejected
  3. Models load on the first request (CUDA if available, else CPU)
  4. Resize to 224×224 and apply ImageNet normalization
  5. Up to four 1280-d embeddings
  6. Fusion V3 concatenates them to 5120 and applies the MLP, then sigmoid
     Fusion V2 uses three embeddings (3840) and skips MRI
  7. 15 dropout passes for the uncertainty figure
  8. Norm shares and Grad-CAM++
  9. Save the row in data/scans.json, including shap_mri and model_used


═══════════════════════════════════════════════════
FOLDER LAYOUT
═══════════════════════════════════════════════════

  This repository
  ├── app.py
  ├── requirements.txt
  ├── .env.example
  ├── README.txt
  ├── data/                          JSON tables used at runtime
  ├── database/mysql_schema.sql      unused schema; includes shap_mri
  ├── models/                        the six .pth files
  ├── frontend/index.html
  └── backend/
      ├── config.py
      ├── inference/engine.py        model catalog and run_inference()
      ├── services/llm_service.py    Groq, then Ollama
      ├── services/prompt_service.py
      ├── services/image_service.py
      └── routes/                    auth, scans, chat, clinic, admin, health

  G:\dev_part1  (training only; Flask does not import these)
  ├── step1_explore_and_build_csv.py
  ├── step2_preprocess.py
  ├── step3_train.py
  ├── step4_app.py                   Gradio demo for the CBIS model
  ├── busi_train.py
  ├── rsna_train.py
  ├── fusion_train.py
  ├── fusion_v2_train.py
  ├── fusion_v3_train.py
  ├── vindr_validate.py
  ├── vindr_validate_v3.py
  ├── *_metrics.json
  └── vindr_*_results.csv / .json


═══════════════════════════════════════════════════
SETUP
═══════════════════════════════════════════════════

  1. Copy the six .pth files into models/.
  2. copy .env.example .env
     Set SCANOVA_SECRET, ADMIN_PASSWORD, and either Groq
     (GROQ_API_KEY, GROQ_API_URL, GROQ_MODEL) or a local Ollama model.
  3. python -m venv .venv
     .venv\Scripts\activate
     pip install -r requirements.txt
     Optional NVIDIA wheel, after that install:
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
     Optional local chat:
       ollama pull llama3.1:8b
  4. python app.py
  5. Open http://localhost:5000
     Health:        http://localhost:5000/api/health
     Model catalog: http://localhost:5000/api/models/info

The startup banner prints OK or MISSING for CBIS, RSNA, BUSI, MRI,
Fusion V3, and Fusion V2, then Groq and Ollama status.

To retrain from G:\dev_part1, with the datasets at the paths above:

  pip install -r requirements.txt
  python step1_explore_and_build_csv.py
  python step2_preprocess.py
  python step3_train.py
  python busi_train.py
  python rsna_train.py
  python fusion_train.py
  python fusion_v2_train.py
  python fusion_v3_train.py
  python vindr_validate.py
  python vindr_validate_v3.py

The MRI encoder is assumed to already exist as mri_model.pth.
Do not rerun a fusion trainer if you still need the current checkpoint:
the script deletes the previous .pth and metrics file at start.
CBIS training on the machine used for step3 took about 1–2 hours
on a Dell G15 GPU (CUDA cu118 wheel).


═══════════════════════════════════════════════════
ENVIRONMENT
═══════════════════════════════════════════════════

Secrets and the LLM model name are read from .env only.

  FLASK_HOST, FLASK_PORT, FLASK_DEBUG
  ALLOW_INSECURE_DEFAULTS
  SCANOVA_SECRET              JWT key, at least 32 random characters
  JWT_ALGORITHM, JWT_TOKEN_HOURS
  ADMIN_EMAIL, ADMIN_PASSWORD, ADMIN_NAME
  GROQ_API_KEY, GROQ_API_URL, GROQ_MODEL
  OLLAMA_ENABLED, OLLAMA_BASE_URL, OLLAMA_MODEL
  CORS_ORIGINS
  MAX_UPLOAD_MB               default 16
  MIN_PASSWORD_LENGTH         default 8
  RATE_LIMIT_LOGIN            default 10 per 15 minutes per IP
  STORAGE_BACKEND             json
  DATA_DIR, MODELS_DIR, FRONTEND_DIR

  python -c "import secrets; print(secrets.token_urlsafe(48))"


═══════════════════════════════════════════════════
SECURITY AND STORAGE
═══════════════════════════════════════════════════

bcrypt password hashes (bcrypt pinned to 4.0.1 for passlib), JWT on
protected routes, four roles, optional TOTP, login rate limit, CORS
from the configured list, upload size and extension limits
(jpg, png, dcm), basenames only on stored filenames, and a startup
warning list for weak secrets.

Runtime data is JSON under data/:

  users, scans, patients, appointments, role_requests,
  admin_logs, scan_chats, direct_chat_sessions, direct_chat_messages

MySQL is described in database/mysql_schema.sql and is not switched on.


═══════════════════════════════════════════════════
API
═══════════════════════════════════════════════════

  POST   /api/auth/login
  POST   /api/auth/register
  GET    /api/auth/me
  PUT    /api/auth/profile
  PUT    /api/auth/password
  GET    /api/auth/mfa/status
  POST   /api/auth/mfa/start
  POST   /api/auth/mfa/verify
  POST   /api/auth/mfa/disable

  POST   /api/scan/analyze
  POST   /api/scan/chat
  POST   /api/chat/general
  GET    /api/scans
  GET    /api/scans/<id>
  PATCH  /api/scans/<id>/label
  DELETE /api/scans/<id>
  GET    /api/scans/<id>/chats
  DELETE /api/scans/<id>/chats

  GET    /api/direct-chats
  POST   /api/direct-chats
  GET    /api/direct-chats/<sid>
  POST   /api/direct-chats/<sid>/message
  PATCH  /api/direct-chats/<sid>/title
  DELETE /api/direct-chats/<sid>

  GET    /api/dashboard/stats
  GET    /api/patients
  POST   /api/patients
  DELETE /api/patients/<id>
  GET    /api/schedule
  POST   /api/schedule
  PUT    /api/schedule/<id>
  DELETE /api/schedule/<id>

  GET    /api/users
  DELETE /api/users/<id>
  PUT    /api/users/<id>/role
  GET    /api/users/<id>/detail
  GET    /api/admin-logs
  POST   /api/role-requests
  GET    /api/role-requests/my
  GET    /api/role-requests
  PUT    /api/role-requests/<id>
  GET    /api/role-requests/pending-count

  GET    /api/health
  GET    /api/models/info
         test_auc, sensitivity, specificity, accuracy,
         plus checkpoint val_auc and which model will run


═══════════════════════════════════════════════════
PAGES
═══════════════════════════════════════════════════

  Landing         Fusion V3 test AUC 0.9897, accuracy 98.76%,
                  specificity 100%; comparison chart for all six models
  Diagnostic lens multi-file upload, risk badge, four contribution bars,
                  heatmap
  Scan detail     metrics, heatmap, per-scan chat, rename, delete
  Direct chat     saved sessions, role-conditioned answers
  Scheduling      mammography, ultrasound, and breast MRI appointments
  Patients        clinic and admin registry
  Users           admin list, role change, detail export
  Analytics       the same recorded test table
  Settings        name, email, password, MFA
  Admin home      counts, which checkpoints exist, audit log


═══════════════════════════════════════════════════
TROUBLESHOOTING
═══════════════════════════════════════════════════

  AVIF rejected
    Convert to JPG or PNG.

  Page looks stale
    Restart python app.py and hard-refresh the browser.

  Chat unavailable
    Set the three GROQ_* variables, or run Ollama with the tag in
    OLLAMA_MODEL. The startup banner shows which path is configured.

  Result label says Fusion V2
    fusion_v3_model.pth is not in models/.
    GET /api/models/info field "active" is the one to trust.

  A model line prints MISSING
    That file is not in models/. Fusion V3 can still run; the missing
    encoder contributes zeros.

  CUDA out of memory while training
    Lower batch size from 16 to 8 in that training script.

  CBIS step 1 finds no images
    Set BASE_DIR in step1_explore_and_build_csv.py to the manifest path
    in the Datasets section.

  passlib / bcrypt warning
    requirements.txt pins bcrypt==4.0.1 on purpose.


═══════════════════════════════════════════════════
REFERENCES FOR THE METHODS
═══════════════════════════════════════════════════

  EfficientNet     Tan and Le, ICML 2019
  Grad-CAM++       Chattopadhay, Sarkar, Howlader, and Balasubramanian,
                   IEEE WACV 2018
  MC dropout       Gal and Ghahramani, ICML 2016
  Datasets         CBIS-DDSM, RSNA Screening Mammography (Kaggle),
                   BUSI ultrasound, Duke Breast Cancer MRI,
                   VinDr-Mammo

Scanova is a research prototype. It does not replace a radiologist's
reading or a pathology result.
