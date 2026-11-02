**I would test smooth, monotone reconstruction of the final patient predictions first, then a supervised patient-level measurement model that jointly handles medication response and informative missingness.** These address two different limitations: incoherent predictions along a patient's trajectory, and uncertainty about the underlying severity that produced their observed scores.

This is a research report, not an implementation or a new benchmark. I read `README.md`, the six notebooks directly under `exploration/`, and only `exploration/stacking/opt.ipynb` inside the stacking directory. I examined the original training/test tables and two saved prediction CSVs. I did not inspect the figures, fitted-model directory, other stacking notebooks, or `transformer details.md`. No project code was changed and no training jobs or hyperparameter searches were launched. The numerical work below consists of descriptive calculations, small least-squares fits to characterize target shapes, and in-memory inspection of saved predictions.

**What the existing project actually establishes.**

The current reference is the complete optimized Encoder–HistGB pipeline, not the original HistGB benchmark or the older feature-engineering results:

| Model | Recorded mean validation MSE | Recorded SD | RMSE derived from mean MSE |
|---|---:|---:|---:|
| HistGB used by the stacking notebook | 11.9234 | 0.3527 | 3.4530 |
| Optimized standalone Transformer | 8.4295 | 0.7942 | 2.9034 |
| Optimized Encoder–HistGB | **7.4771** | 0.3367 | **2.7344** |

These are saved outputs in `opt.ipynb`, cells 16–17, not scores reproduced during this investigation. The encoder results summarize five patient splits and three seeds. They are averages of individual model scores, not scores from averaging those models' predictions. The challenge specifies RMSE, while the repository optimizes MSE. They rank predictions identically on a fixed evaluation set; final reporting should retain MSE for repository comparisons and calculate RMSE from pooled squared errors. Averaging fold RMSEs is a different aggregation. [Official challenge description](https://challengedata.ens.fr/challenges/159)

The implemented preprocessing is more specific than “imputation plus histories.” `features_selection.ipynb`, cells 3–8, supplies the definitions executed by the stacking notebook:

1. Create patient histories for the imputation regressors. Score/dose medians and slopes exclude the current visit.
2. One-hot encode cohort and gene, and remove `Index`.
3. Predict missing diagnosis age, ON, OFF and LEDD with four small HistGB regressors. Preserve original missingness flags; median-fill remaining numeric gaps. Each imputer predicts from the original input table, so this is not chained imputation.
4. Drop the imputation helper histories. Construct final slopes, trends, neighboring values, dispersions and counts from the **imputed** measurements.
5. Impute remaining engineered-feature gaps and fit the final HistGB without feature selection.

The encoder receives the imputed original features, original missingness indicators and elapsed time, rather than the final engineered features. Patient IDs group sequences but are not model inputs. Attention is bidirectional, with padding excluded from attention keys and the loss. Input and target scaling are fitted inside training patients. An internal 20% patient holdout selects the epoch count, followed by a fresh encoder fit on all outer-training patients. HistGB then receives the frozen contextual embeddings alongside its engineered features.

The winning saved configuration is width 32, eight heads, three layers, feed-forward multiplier 3, dropout approximately 0.135, and a two-layer regression head with 16 units. The search uses 40 Optuna trials on the first three splits with seed 42. The actual code confirms **two** configurations on five splits and seeds 42–44, although some markdown says three. This distinction does not invalidate the results, but the implementation should determine any replication budget.

The implementation and README already cover substantial ground. History-assisted imputation reduced the recorded HistGB MSE from 54.234 to 45.286. Final history features and encoder embeddings produced much larger subsequent gains. Feature selection produced only small exploratory gains: mutual-information selection reached 11.8646 against 11.9234 for the HistGB reference. The outlier notebook found no convincing benefit from removing difficult cases; its 14.7793 baseline predates the current pipeline. The README also records unsuccessful CLS tokens, sinusoidal time encoding, engineered encoder inputs, an encoder-prediction variant using a reduced training set, and triangular medication-effect features. I have not independently verified the excluded stacking variants.

**The dataset findings that determine my recommendations.**

| Property | Direct observation |
|---|---|
| Training set | 55,603 visits; 6,971 patients |
| Test set | 23,672 visits; 2,988 patients |
| Patient overlap | None between train and test |
| Sequence length | 4–12 visits per patient; mean 7.98 in training |
| Follow-up span | Mean 7.29 years in training |
| Missing ON / OFF / LEDD | 29.96% / 42.10% / 37.04% |
| Missing ON / OFF intake delay | 46.65% / 78.82% |
| Patients without any observed ON / OFF | 765 / 84 |
| Within-patient target ordering | **All 6,971 target trajectories are nondecreasing with age** |

There are no duplicate patient–age pairs, and every visit has at least one observed ON or OFF score. Sex, cohort, gene and observed diagnosis age do not vary within a patient. The test-only `time_since_diagnosis` column equals `age - age_at_diagnosis` to floating-point precision; the notebook drops it by selecting training columns. Test diagnosis age has no missing values, compared with 5.09% missing in training. That availability difference merits a diagnostic subgroup, not an assumption that the extra test column supplies a new training feature.

The target is strikingly smooth. Fitting polynomials to **all known labels of each patient** gives these reconstruction errors:

| Polynomial degree | All training visits: reconstruction MSE | Only 12-visit patients: reconstruction MSE |
|---|---:|---:|
| Linear | 2.5789 | 2.8638 |
| Quadratic | 0.1116 | 0.1373 |
| Cubic | 0.0243 | 0.0348 |

These are **oracle shape diagnostics**, not out-of-sample prediction results, attainable model scores, or estimates of irreducible error. In particular, a cubic exactly interpolates four points; the 12-visit result is the more persuasive evidence of smoothness. Curvature varies in sign, so a universal convex or concave constraint would be unjustified.

Missingness is also strongly associated with severity. ON is missing in 98.9% of visits with target at most 20, versus 6.6% with target between 40 and 60. OFF is missing in approximately 0.6% and 56.1% of those respective groups. Even after subtracting each patient's mean target, visits with missing OFF have a mean centered target of +2.19, compared with −1.60 when OFF is observed. These are descriptive associations; they do not establish a causal missing-data mechanism.

There is a concrete limitation in the final history representation: because `build_visit_features` runs after ON/OFF/LEDD imputation, all three `*_n_observed` features equal the patient's visit count. Neighboring-score features also refer to adjacent imputed visits rather than necessarily adjacent genuine measurements. Original row-level flags survive, so the information is not completely lost, but these aggregates do not summarize genuine observation density or its temporal gaps.

**How much confidence to place in the current validation.**

The outer patient separation and independently fitted preprocessing are appropriate for predicting complete histories of new patients. Using later predictor visits is consistent with the README's retrospective task; no later target should be available at prediction time.

There are three qualifications when evaluating improvements:

- Search and confirmation reuse validation patients. The five `GroupShuffleSplit` splits cover 4,667 distinct validation patients, and some appear repeatedly. Fifteen fold–seed scores are not fifteen independent experiments; their SD is not a confidence interval for a performance gain.
- The encoder's supervised training embeddings are in-sample for the downstream HistGB training rows. Because the entire encoder is fitted inside each outer-training split, this is **not demonstrated outer-validation target leakage**. It can nevertheless create a representation/generalization mismatch. Independently trained encoders also have unaligned latent coordinates, so simply concatenating embeddings generated by different cross-fitting models is not a sound automatic remedy.
- HistGB retains default `early_stopping='auto'`. I checked the installed scikit-learn 1.9.1 implementation: at these sample sizes it uses an internal row-level validation split. This does not contaminate the outer patient validation, but its stopping criterion differs from the new-patient objective. Keep this behavior identical between paired models; if changing it, use a patient holdout within training and establish a new reference for both arms. [HistGB documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)

The following are exactly the two methods I recommend.

**1. Reconstruct a smooth, nondecreasing trajectory from the final Encoder–HistGB predictions.**

For a patient, sort visits by age and denote the existing final predictions by vector `p`. Replace independent visit predictions with a jointly reconstructed vector `z`:

\[
\hat z_i=\arg\min_{z_1\le\cdots\le z_{n_i}}
\left\{\|z-p_i\|_2^2+\lambda\|\widetilde D_i z\|_2^2\right\}.
\]

Here `D` measures changes between successive slopes using the actual age gaps. It penalizes roughness in time, not merely differences by visit number. Normalize elapsed ages to [0,1] within a patient and normalize the operator, for example by its Frobenius norm, so one global smoothing strength is usable across sequence lengths. A lower bound of zero can also be imposed.

With `lambda=0`, this is ordinary isotonic projection of predictions. With positive `lambda`, it also suppresses noisy changes in progression rate. The relevant statistical idea is to restrict predictions to a plausible low-complexity output space after a flexible model has extracted the input signal. Projection-based correction of initially nonmonotone estimators has a substantial literature. [Westling, van der Laan and Carone](https://arxiv.org/abs/1810.09022)

There is a particularly useful finite-dimensional guarantee for the **unweighted, unpenalized** case. If the unknown true vector `y` belongs to the nondecreasing cone `C`, Euclidean projection satisfies

\[
\|P_C(p)-y\|_2^2\le\|p-y\|_2^2-\|P_C(p)-p\|_2^2.
\]

Thus isotonic projection cannot increase total squared error on a genuinely nondecreasing patient trajectory. This follows directly from convex projection geometry. It does **not** extend unchanged to positive smoothing penalties, arbitrary reliability weights, or a test population whose targets can decrease.

**Why this fits this project.** The target trajectories satisfy the ordering constraint throughout the available training set, and their very low quadratic/cubic reconstruction error supports additional smoothness. Neither the per-visit encoder regression head nor HistGB enforces coherent output trajectories. Adding more input history features does not impose this output structure.

The saved file `submission_6__11_01_2026_encoder_histgb_optimized.csv` has 584 downward steps among 20,684 adjacent visit pairs, affecting 479 of 2,988 test patients. In-memory isotonic projection changes those predictions by a mean squared amount of only 0.00438. This is a prediction-coherence diagnostic, **not a measured test gain**, and I did not verify that this renamed CSV came from the exact configuration underlying the saved notebook scores. It suggests that isotonic correction alone may be modest. The larger hypothesis is that smooth reconstruction improves trajectories even when their predictions are already ordered.

**Implementation.** Add a prediction-only function beside `evaluate_config` in `opt.ipynb`. Apply it immediately after the final HistGB prediction in cell 12, before calculating validation MSE, and in the optional final prediction path in cell 19. Group by patient, use their ages, and restore the original row order. The imputation, encoder and HistGB definitions can remain the same for the first experiment. With at most 12 predictions per patient, each penalized problem is a tiny convex quadratic program; ordinary isotonic projection is cheaper still.

Start with equal visit weights to match the evaluation metric. Compare the untouched predictions, isotonic projection, and smooth monotone reconstruction. For subsequent tuning, use a small predeclared grid such as `lambda in {0, 0.1, 1, 10}` after defining the normalization above. Avoid selecting a separate smoothing strength from each validation patient's labels. Do not impose a common curvature sign or force every trajectory to be linear.

**Controlled validation.** The smallest meaningful experiment is one existing patient split, the fixed winning encoder configuration, and one fixed seed. Fit the existing pipeline once and retain its validation predictions. Compare unchanged predictions with isotonic projection and one prespecified moderate smoothing strength, such as 1 with the stated normalization. Treat that split as an exploratory screen, not a tuning set plus independent confirmation. If predictions were already cached, no retraining would be required; the notebook's saved aggregate scores alone are insufficient to perform this test now.

If promising, select smoothing strength using patient-held-out predictions inside each outer-training set, then score the frozen choice on outer-validation patients. Evaluate the same predictions before and after smoothing so the comparison is exactly paired. Report results by sequence length, follow-up span and genuine observation density, as well as overall MSE/RMSE.

**Critical assessment.** This has the best cost-to-evidence ratio: implementation is small, it applies directly to the strongest model, and it adds negligible inference cost. It can reduce visit-level variance while preserving the information already learned. Its main limitation is that smoothing generally cannot correct a patient-wide level error. Positive smoothing can also erase genuine local structure or distort endpoints, particularly for short histories. Monotonicity is established for this synthetic training target, not for arbitrary clinical motor measurements. Expected potential is modest for isotonic projection alone and plausibly more meaningful for smoothing; the actual gain is unknown.

**2. Learn a supervised, partially pooled measurement model and use its patient-level reconstructions as new features.**

Model the observed ON/OFF values as noisy measurements of an underlying unbiased trajectory, with a stable patient medication-response component and a learned observation process. Instead of asking a tree to discover all these relationships from separately imputed columns, explicitly infer which latent severity path could have generated the patient's measurements and missingness pattern.

A useful starting formulation is

\[
M_{ijq}=S_i(t_{ij})-A_i h_\theta(d_{ijq})+b_{c_i,q}+\epsilon_{ijq},
\qquad q\in\{ON,OFF\}.
\]

`S` is the unbiased trajectory, `A` a positive patient response amplitude, `h` a flexible shared intake-delay response, and `b` a strongly regularized cohort/measurement offset if needed. Put population priors on the trajectory coefficients and log amplitude, optionally conditioned on the available static covariates. Patients with sparse evidence then shrink toward an appropriate population distribution rather than receiving unconstrained individualized fits.

During training, the known target supervises `S` and anchors the observed residuals `target - ON` and `target - OFF`. At inference, **all parameters specific to a new patient must be inferred from that patient's X values only**, using population parameters learned from training patients. A patient ID indexes an inference problem; it must not retrieve a supervised embedding or fitted target curve.

This formulation matches the challenge's stated synthetic construction: patient-specific response curves are assumed stable over time, and the target represents an estimated unbiased OFF score. This is a modeling hypothesis about the challenge, not a claim that a particular physiological equation generated these files. [Official challenge description](https://challengedata.ens.fr/challenges/159)

Population pharmacokinetic/pharmacodynamic modeling and Bayesian inference of individual responses provide a relevant methodological precedent, including work examining sparse observations per patient. That evidence supports the partial-pooling approach, not an expected numerical improvement here. [Triggs et al., 1996](https://pubmed.ncbi.nlm.nih.gov/8880053/)

**Why this differs from the failed medication experiment.** The README's triangular response fit uses OFF–ON differences only when both scores and both times are present: 11,188 visits in these files. It estimates a shared curve and patient amplitudes, then supplies effect features. The proposed model instead uses the unbiased training target to learn measurement correction, can train from individual ON or OFF measurements, and uses population priors for unseen patients. There are 29,663 timed ON observations and 11,775 timed OFF observations: **41,438 measurement terms**, not 41,438 distinct visits or independent observations. Their dependencies must remain patient-grouped.

It also explicitly uses information that imputation blurs. Model the observed score-availability state `R` as one of ON only, OFF only, or both, with a regularized probability `p(R | S, cohort, time)`. This respects the fact that no visit has both scores missing. Estimate this relationship rather than hardcoding a target threshold of 20. Research on informative missingness supports modeling masks and measurement timing; it does not establish that replacing this project's Transformer with GRU-D would help. [Che et al., 2018](https://clinicalml.org/publication/che-et-al-nature-sr-18/)

**A feasible first implementation.** Keep the initial model deliberately small:

- Use a low-dimensional monotone spline for `S`, approximately three or four basis functions with partial pooling. Begin with one random log amplitude per patient and a shared response shape; defer patient-specific peak and decay parameters until evidence supports their identifiability.
- Represent `h` with a small smooth absorption/decay curve or a shared unimodal spline with approximately four to six knots. It should be more flexible than a triangle without allowing an arbitrary curve per sparse patient. Fix its scale, for example peak height 1, to separate amplitude from response shape.
- Fit measurement noise separately for ON and OFF. Do not impose `target >= observed OFF` on every sample: 20.6% of timed OFF measurements exceed their targets. Account for score-floor censoring if modeling the observed-score likelihood. Keep target supervision and model selection aligned with squared error.
- For missing intake delays, integrate the observation likelihood over a training-fitted conditional delay distribution, initially using 8–16 quadrature values or empirical draws per score type and cohort. Missing times should not become precise median times. This incorporates the README's planned marginalization idea into a jointly supervised model rather than recommending it alone.
- Preserve raw masks, genuine observed counts, and gaps to genuine measurements before imputation. LEDD may condition priors or effect size, but daily LEDD should not be treated as a known single-dose pharmacokinetic input.

Fit global response parameters and population priors using training targets and observed measurements. A concrete joint training objective is target MSE plus a weighted observed-measurement negative log likelihood, a weighted availability-state negative log likelihood, and the negative log population prior on patient parameters. Normalize the first three terms by their respective numbers of visits or observed measurements so missingness does not accidentally change their relative scale. Standardize target/measurement units using training statistics, and initially restrict the measurement and availability weights to a small grid such as `{0.1, 1}`; compare availability weight zero as the planned ablation. Select prior strength inside grouped training rather than allowing free patient fits. For a new patient, infer the posterior over trajectory coefficients and amplitude from their observed scores, masks, times and covariates. A penalized maximum-a-posteriori fit is an economical prototype; a Laplace approximation or a small posterior sample can provide uncertainty. Use posterior mean severity for squared-error prediction when feasible.

Export a small set of aligned scalar features: inferred severity at the visit, its approximate uncertainty, response amplitude, and measurement-reconstruction error or evidence count. Append these to the current HistGB input alongside the existing embeddings. Their physical meanings remain consistent across folds, unlike arbitrary encoder coordinates. The model supplies a complementary estimate; HistGB can learn when it is useful.

**Where it would connect.** Preserve genuine measurements and masks alongside the preprocessing definitions in `features_selection.ipynb`. Add the measurement-model branch around the fold-preparation/evaluation logic in `opt.ipynb`, cells 12 and 19. Its inputs should use the raw table rather than treating imputed ON/OFF values as observed likelihood terms. The existing `VisitEncoder` can stay fixed initially, and the new branch's features can be cached across encoder seeds.

**Leakage control is essential.** The global measurement model and its priors are supervised by `y`. Therefore generate its outer-training features with inner patient-group cross-fitting: learn population parameters on inner-training patients, then infer features for held-out patients using only their X. Fit population parameters on all outer-training patients for outer-validation inference. No held-out patient's true trajectory may be used to estimate their amplitude, offset, uncertainty or reconstructed severity. Fit preprocessing, delay distributions and stopping rules within the same permitted training boundary.

**Controlled validation.** On one fixed outer split and seed, compare the existing Encoder–HistGB against the same pipeline plus the cross-fitted scalar features. As a cheap control, also measure the effect of adding genuine observation counts/gaps without the latent model; the mechanistic branch should demonstrate value beyond that representation correction. Internally ablate the observation-process likelihood and delay marginalization to identify where any gain comes from. Select likelihood weights and prior strengths only inside training; do not open another large encoder search.

For an initial screen, use three inner patient folds, the shared response shape, one random amplitude and a strongly pooled trajectory. Exclude a freely varying patient offset or extra time constants initially: these can be confounded with severity. Assess the branch's own held-out MSE and, more importantly, whether its features improve the complete 7.48-reference architecture. Correlation with baseline errors helps assess complementarity, but is not itself proof of an ensemble gain.

**Critical assessment.** This has potentially greater upside than output smoothing because it can change the inferred level of an entire patient trajectory, not just its local coherence. It uses supervised correction targets that the earlier response fit did not exploit. It also gives an explicit role to sparse observations and uncertainty. It adds modeling capability rather than new external information; the existing encoder already has access to many of the same raw signals.

The main risk is identifiability: severity, medication amplitude, bias and unobserved delay can explain similar measurements. Four-visit histories cannot support many unconstrained parameters. Priors and supervised population learning are therefore central. Additional risks include a misspecified response family, a biased delay distribution learned from observed times, double-counting dependent evidence, an unstable missingness mechanism, and failure to beat an encoder that already approximates the needed inference. Predictive usefulness should determine retention; inferred parameters should not be interpreted as validated physiology.

Implementation difficulty is moderate to high. The prototype involves small patient-level latent problems rather than another large attention architecture. Its cost should be manageable on the existing CPU/MPS setup, but grouped cross-fitting multiplies fitting work and must be budgeted explicitly. Wall-clock estimates require a prototype. Expected improvement potential is meaningful but substantially less certain than the direction of the monotonicity correction; no percentage gain can be justified yet.

**How to decide whether either result is robust.**

Freeze the winning encoder parameters, reference HistGB parameters, patient splits and seeds before screening. For an inexpensive comparison, reuse the existing splits so each gain is paired. For stronger subsequent evidence, use a predeclared grouped outer evaluation with all smoothing strengths, prior strengths, likelihood weights and feature choices selected inside its training partitions. Future predictor visits can contribute within a patient's available history, but never future labels.

Record predictions and squared errors for every patient, fold and seed. Calculate the pooled visit-level MSE difference and its RMSE counterpart. Bootstrap **patients**, retaining their visits together, rather than treating visits as independent. With repeated splits, resample each patient's entire set of validation appearances together and describe the interval as conditional on the fitted models. Report seed sensitivity separately; three seeds do not create three independent patient samples. Include equal-patient error as a secondary diagnostic while retaining the challenge's visit-weighted metric.

Check whether gains survive across sequence lengths, cohorts, missing-score patterns, diagnosis availability and target ranges. These are diagnostics, not permission to select a favorable subgroup. For a retained method, require a consistent overall direction across most paired splits and seeds, a patient-cluster interval supporting improvement, and no large unexplained subgroup failure. Predeclare the limited experiment budget to reduce repeated-selection bias.

There is no historically pristine internal holdout: the project has reused validation splits, and this investigation examined global target shapes. New splits and nested evaluation strengthen internal evidence but cannot erase that history. An eventual hidden challenge evaluation remains valuable external confirmation. I have not verified leaderboard performance or a published winning solution.

**The comparison and test order are:**

| Rank | Method | Main limitation addressed | Expected potential | Difficulty and incremental cost | Smallest meaningful experiment |
|---:|---|---|---|---|---|
| **1** | Smooth monotone reconstruction of final predictions | Visit-level incoherence and excess trajectory roughness | Modest for ordering alone; potentially more useful with smoothness | Low difficulty; negligible prediction-time cost | One patient split and seed; retain one baseline prediction vector and compare unchanged, isotonic, and one fixed smooth projection |
| **2** | Supervised, partially pooled measurement model with informative missingness | Patient-level severity/medication ambiguity and lost observation structure | Potentially larger; lower confidence because of identifiability | Moderate/high difficulty; small latent fits multiplied by grouped cross-fitting | One patient split and seed; three inner folds; shared response curve, one random amplitude, pooled trajectory; append cross-fitted scalar features |

**Experimental roadmap.** First, cache visit-level validation predictions from one fixed run of the current optimized pipeline and perform the three-way reconstruction screen. Advance smoothing only after an inner-training choice improves held-out patients. Second, build the smallest measurement-model branch and compare its additional features against both the original reference and the genuine-observation-summary control. Only after an individual method survives grouped confirmation should both be combined: infer improved patient levels with the measurement branch, then reconstruct the final predictions smoothly. No new 40-trial architecture search is needed to validate either hypothesis.
