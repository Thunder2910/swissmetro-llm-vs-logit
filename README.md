# Can LLMs Simulate Travellers?

Comparing LLM-simulated mode choice with a multinomial logit model on Swissmetro.

**Veeraj Jain** — undergraduate, Civil Engineering, Indian Institute of Technology Delhi.
October 2026.

This repository asks a narrow question. If an open-weights language model is given a traveller's attributes and the time, cost, and headway of the train, Swissmetro, and car, does it choose the way a multinomial logit fitted to the same stated-preference survey does? The models are Qwen2.5-Instruct at 1.5B, 3B, and 7B, same prompt, same 420 tasks.

They do not. On those tasks the logit is accurate on 63.1% and its mean probabilities match the observed shares. The 1.5B model is accurate on 34.3% and puts most of its probability on the car. The 3B model is accurate on 57.4%, which is exactly the rule that always picks Swissmetro, and it puts 99.95% of its probability there. The 7B model is accurate on 53.8% and puts 93.3% on Swissmetro. The car prior disappears as the model gets larger. It is replaced by a Swissmetro prior. Log-likelihood gets worse, not better: -693, -1,973, -2,819. The 1.5B model notices an annual public-transport pass. The 3B and 7B models do not. None of the three yields a value of time. Every number below is written out by `src/run_all.py` into `results/`.

## Motivation

Transport models used in appraisal are estimated on surveys. A recent claim is that a language model can stand in for those respondents: give it a persona and a scenario, and read its choice as a simulated traveller. That claim is easy to overstate and easy to test on a dataset with a published benchmark.

Swissmetro (Bierlaire, Axhausen, and Abay, 2001) is a stated-preference survey of intercity travel in Switzerland, including a hypothetical maglev. The Biogeme example `b01a` is a four-parameter multinomial logit with a known log-likelihood. This study reproduces that model, adds one richer specification, and scores three sizes of one local instruct model on the same held-out people. The logit is fitted to other respondents from this survey. The language models see none of their choices. Losing to the logit is the expected direction. Whether that gap shrinks with model size, and what replaces it, are the result.

## Data

The file `data/swissmetro.dat` is the public release from the Transport and Mobility Laboratory at EPFL (`https://transp-or.epfl.ch/data/swissmetro.dat`). SHA-256 `27432693cf052985d79a950b4b888be3efca798fc89b0d3ffefe40608ede00f2`. The code checks the checksum and downloads the file only if it is missing.

The estimation sample uses the Biogeme filters: `PURPOSE` in {1, 3} (commute or business) and `CHOICE != 0`. That leaves **6,768 tasks from 752 people, nine tasks each**. The excluded count is 3,960. Choices are train 908, Swissmetro 4,090, car 1,770. The car is unavailable for 129 people (5,607 tasks have a car). The chosen alternative is always available.

Time and cost enter the logit divided by 100, as in the Biogeme example. For a holder of the General Abonnement, `TRAIN_CO` in the raw file is the **annual** pass price (median 2,900 CHF), not the fare of the trip (median rail fare without the pass: 84 CHF). The payable train cost and the payable Swissmetro cost are set to zero for pass holders. The car cost is unchanged. The prompt is told about the pass in a sentence, and it is shown the zero fares. It is not shown the annual price.

The filtered sample is uneven: 577 of 752 people are on a business trip, and 589 are men. One hundred people hold the pass. Those facts bound every share in this repository.

## Models

### Multinomial logit

Swissmetro's alternative-specific constant is fixed at 0. The base systematic utilities are

```
V_train = ASC_TRAIN + B_TIME * (train time / 100) + B_COST * (train cost / 100)
V_SM    =                 B_TIME * (SM time / 100)    + B_COST * (SM cost / 100)
V_car   = ASC_CAR   + B_TIME * (car time / 100)   + B_COST * (car cost / 100)
```

with unavailable alternatives dropped from the logit denominator. The code maximises the log-likelihood by Newton–Raphson with the analytical gradient and Hessian. Classical standard errors are the inverse Hessian. The tables below use cluster-robust standard errors, one cluster per respondent, because the nine tasks of one person are not nine people. Observation-level robust standard errors are also stored; they match the Biogeme robust standard errors. p-values use the standard normal approximation. There is no small-sample correction.

The richer specification adds seven parameters, nested in the base model:

- business-trip shifts of the train and car constants (commute is the base)
- a pass-holder shift for the train, and one for Swissmetro
- headway, in the same per-100-minutes units, on the train and Swissmetro (zero for the car)
- an extra time coefficient for age 55 or over (`AGE` in {4, 5})
- an extra cost coefficient for income above 100,000 CHF/year (`INCOME == 3`)

Value of time, in CHF per hour, is `60 * B_TIME / B_COST`. The hundreds cancel because both attributes are scaled by 100. The standard error is the delta method with the cluster covariance.

### Split

`numpy` Generator seed 42 shuffles the 752 respondent IDs. The test set is 20% of people: **150 people, 1,350 tasks**. The training set is **602 people, 5,418 tasks**. Models used for prediction are estimated on the training people only. The published Biogeme comparison uses the full 6,768.

### Language models

Three instruct models, each quantised **Q4_K_M** and run locally with `llama-cpp-python`. No API key. Four CPU threads, context 1,024. The Llama seed is 42. Temperature is not used, because nothing is sampled.

| Key | Weights | Bytes |
| --- | --- | ---: |
| 1.5B | `Qwen2.5-1.5B-Instruct` | 1,117,320,736 |
| 3B | `Qwen2.5-3B-Instruct` | 2,104,932,768 |
| 7B | `Qwen2.5-7B-Instruct`, two GGUF shards | 3,993,201,344 + 689,872,288 |

llama.cpp loads the 7B pair when it is given the first shard. The files live in `models/` and are gitignored. `src/model_specs.py` checks the byte counts.

The prompt is `prompts/system.txt` plus `prompts/user_template.txt`, and it is the same text for all three models. It states the purpose, age band, gender, income band, who pays, luggage, and whether the person holds a General Abonnement, then the door-to-door time, trip cost, and headway of each available mode. The requested reply is a single digit: 1 train, 2 Swissmetro, 3 car. Choice probabilities are the next-token distribution, restricted to those digit tokens and renormalised over modes that exist in the task. A prefix cache keeps the system turn fixed.

The wording was not selected by test-set accuracy. A 24-task pilot on training respondents checked that the top token was a digit (format rate 1.000 for each model). Eight further tasks were scored only to time the machine; three of them fell in the test people and are excluded from the evaluation. The cache key changes if the prompt files change. Each model has its own cache file, so the 1.5B scores are not recomputed when a larger model is added.

The reported sample is **420 tasks from 147 test respondents**, at most four tasks each, drawn with seed 49 (`42 + 7`). Observed choices: train 58, Swissmetro 241, car 121. Elasticities for the 1.5B and 3B models use **84 of those tasks (71 people)**, seed 53, at most two per person. Cost elasticities use the **76** tasks in that subset where the person does not hold a pass, because a percentage change of a zero fare does nothing. Shocks are −50, −20, −10, +10, +20, and +50 percent, applied to train and Swissmetro time and cost. A separate ±10 minute and ±10 CHF pair, on tasks where the shift does not hit zero (time at least 20 minutes and cost at least 10 CHF), is the arc value of time.

The 7B model takes about 3.72 seconds per prompt on this machine, against 0.82 and 1.70 for the smaller two. Scoring the 84-task shock grid would have added about three hours after the 420 base tasks. Its time and cost shocks therefore use **36 of those 84 tasks (36 people)**, seed 61, at most one task per person. Of those 36, 31 are unclipped for the arc value of time and 31 pay a fare. Accuracy, market shares, the pass split, and the equalised-attribute probe still use the same 420 and the same 84 as the smaller models. `results/implied_vot_common.csv` repeats the arc value of time for all three models on that 31-task subset.

## Results

### The published logit

On the full filtered sample the base model matches Biogeme example `b01a`.

| Parameter | Estimate | Cluster s.e. | Cluster t | Robust t |
| --- | ---: | ---: | ---: | ---: |
| ASC train | −0.701 | 0.183 | −3.82 | −8.49 |
| ASC car | −0.155 | 0.129 | −1.20 | −2.66 |
| Time, per 100 min | −1.278 | 0.238 | −5.38 | −12.26 |
| Cost, per 100 CHF | −1.084 | 0.161 | −6.72 | −15.89 |

Log-likelihood −5,331.252. Null log-likelihood (equal probability among available modes) −6,964.663. ρ² = 0.235. ρ̄² = 0.234. The null is coefficients at zero, not constants only.

Time and cost have the expected negative signs. The car constant is significant in the robust (task-level) calculation that Biogeme reports, and it is not significant once the nine tasks of each person are clustered. Both numbers are in `results/mnl_summary.json`. The cluster figure is the one used here for inference.

Value of time: **70.7 CHF/hour**, cluster delta-method standard error **13.8**. A 95% interval is about 44 to 98 CHF/hour. This is the trade-off inside this stated-preference design. It is not an appraisal value of time.

At equal time and cost, the train constant is worth −64.7 CHF relative to Swissmetro (cluster s.e. 20.7). The car constant is worth −14.3 CHF (s.e. 12.1), consistent with the insignificant cluster t.

### The richer logit

Eleven parameters. Log-likelihood −4,889.826. ρ² = 0.298. ρ̄² = 0.296. Likelihood-ratio statistic against the base model: 882.85 on 7 degrees of freedom. That p-value treats tasks as independent, so it overstates how much new information the extra parameters add. The cluster t-statistics are the ones to read.

| Parameter | Estimate | Cluster s.e. | Cluster t |
| --- | ---: | ---: | ---: |
| ASC train | −1.584 | 0.246 | −6.45 |
| ASC car | −0.972 | 0.193 | −5.05 |
| Time, per 100 min | −1.523 | 0.217 | −7.02 |
| Cost, per 100 CHF | −1.411 | 0.348 | −4.05 |
| Train ASC, business vs commute | 0.855 | 0.208 | 4.11 |
| Car ASC, business vs commute | 0.814 | 0.183 | 4.44 |
| GA holder, train | 2.169 | 0.494 | 4.39 |
| GA holder, Swissmetro | 0.074 | 0.490 | 0.15 |
| Headway, per 100 min | −0.663 | 0.114 | −5.84 |
| Extra time coefficient, age 55+ | 0.768 | 0.253 | 3.04 |
| Extra cost coefficient, income >100k | 0.460 | 0.342 | 1.35 |

For a person who is under 55 and not in the top income band, the value of time is **64.8 CHF/hour** (s.e. 16.7). For age 55+ it falls to **32.1** (s.e. 15.9); the interaction itself has cluster t = 3.04, so older respondents are less time-sensitive in this file. The high-income point estimate is **96.1 CHF/hour** (s.e. 18.1) with cluster t = 1.35 on the cost interaction. That income result is not established. Headway is disliked about 0.436 times as much as in-vehicle time, which is **28.2 CHF per hour of headway**. The pass-holder shift on the train is worth 153.8 CHF (s.e. 55.5) on top of the free fare. People who like the train are the ones who buy the pass; the coefficient is not a causal effect of handing someone a pass. The Swissmetro pass shift is indistinguishable from zero.

![Base and richer coefficients with cluster-robust 95% intervals](figures/coefficients.png)

### Held-out people, before the language model

Predicted by models fitted on the 602 training people, scored on all 1,350 test tasks.

| Model | Accuracy | Log-likelihood | Mean per task | Mean probability, train / SM / car |
| --- | ---: | ---: | ---: | --- |
| Majority class (always Swissmetro) | 57.3% | — | — | hard share 0 / 100 / 0 |
| Constants only | 57.3% | −1,221.204 | −0.905 | 13.0 / 60.0 / 27.0 |
| Base MNL | 64.3% | −1,138.100 | −0.843 | 13.4 / 57.9 / 28.7 |
| Richer MNL | 67.4% | −1,060.671 | −0.786 | 13.0 / 57.4 / 29.7 |

Observed shares on these tasks: **13.9% / 57.3% / 28.8%**. The base model's mean probabilities sit on those shares. Its hard prediction (the mode with the highest probability) selects the train on **zero** of the 1,350 tasks. Train is often second. Of 188 train choosers, 178 are predicted as Swissmetro and 10 as car. Accuracy therefore misses a model whose probabilities are in the right place. The constants-only model also always predicts Swissmetro, because both constants are negative and Swissmetro is always available, so it ties the majority rule. The 7.0 point accuracy gain of the base model over that rule is the contribution of time and cost.

The richer model does sometimes predict the train (hard share 5.3%, train recall 28.7% on this full test set). Quote the log-likelihood and the probability shares together with the accuracy.

### Language models against the same 420 tasks

The comparison uses the 420-task subset, not all 1,350. The logit rows below are re-scored on those 420 so the models share a denominator. Respondent bootstrap: 1,000 resamples of the 147 people. `results/model_size.csv` is the same table in machine-readable form.

| Model | Accuracy | 95% bootstrap interval | Log-likelihood | Mean per task |
| --- | ---: | --- | ---: | ---: |
| Majority / constants (hard) | 57.4% | 51.4 to 64.1 | −381.03 (constants) | −0.907 |
| Base MNL | 63.1% | 56.8 to 69.6 | −357.22 | −0.851 |
| Richer MNL | 67.1% | 61.3 to 73.4 | −324.01 | −0.771 |
| Qwen2.5-1.5B, Q4 | 34.3% | 28.1 to 40.7 | −693.20 | −1.650 |
| Qwen2.5-3B, Q4 | 57.4% | 51.4 to 64.1 | −1,973.34 | −4.698 |
| Qwen2.5-7B, Q4 | 53.8% | 47.6 to 60.1 | −2,819.35 | −6.713 |

Accuracy differences, same bootstrap:

| Contrast | Percentage points | 95% interval |
| --- | ---: | --- |
| Base MNL − 1.5B | +28.8 | +18.8 to +39.5 |
| Base MNL − 3B | +5.7 | +0.7 to +11.0 |
| Base MNL − 7B | +9.3 | +3.2 to +15.3 |
| Base MNL − majority | +5.7 | +0.7 to +11.0 |
| 1.5B − majority | −23.1 | −35.5 to −11.0 |
| 3B − majority | 0.0 | 0.0 to 0.0 |
| 7B − majority | −3.6 | −6.8 to −0.7 |
| Richer MNL − base MNL | +4.0 | −0.2 to +7.9 |

On these 420 tasks the richer model has a clearly better log-likelihood. Its accuracy gain over the base model has an interval that includes zero. Every language model is worse than the base logit by a margin the bootstrap does not cover. The 3B model ties the majority rule because it hard-predicts Swissmetro on every task. The 7B model is reliably worse than that rule.

Mean probabilities and hard shares on the same 420 tasks. Observed shares: **13.8% train, 57.4% Swissmetro, 28.8% car**.

| Model | Mean probability | Hard prediction |
| --- | --- | --- |
| Base MNL | 13.4 / 58.0 / 28.7 | 0.0 / 82.1 / 17.9 |
| Richer MNL | 12.9 / 57.3 / 29.8 | 5.5 / 73.1 / 21.4 |
| 1.5B | 29.6 / 12.7 / 57.7 | 20.0 / 1.4 / 78.6 |
| 3B | 0.05 / 99.95 / 0.00 | 0 / 100 / 0 |
| 7B | 6.7 / 93.3 / 0.04 | 6.4 / 93.6 / 0 |

Mean probability on the chosen mode: 0.495 for the base MNL, 0.344 for 1.5B (Brier 0.907), 0.574 for 3B (Brier 0.851), 0.536 for 7B (Brier 0.905). The base MNL Brier score is 0.519. Correlation of task-level probabilities with the base MNL, train / Swissmetro / car: 0.38 / −0.04 / 0.36 at 1.5B, 0.27 / −0.03 / 0.17 at 3B, and 0.43 / −0.15 / 0.10 at 7B. Hard predictions agree with the base MNL on 16.0% of tasks at 1.5B, 82.1% at 3B, and 76.2% at 7B. The 3B agreement is the base MNL's own Swissmetro hard share, because the 3B model always picks Swissmetro.

The top token is a digit on every one of the 420 tasks for all three models. At 1.5B that digit is an available mode on 96.0% of tasks. At 3B and 7B it is an available mode on every task. Predictions renormalise over available modes, so an unavailable car is not chosen. Recall, train / Swissmetro / car: 51.7% / 0.8% (2 of 241) / 92.6% at 1.5B; 0 / 100 / 0 at 3B; 8.6% (5 of 58) / 91.7% (221 of 241) / 0 at 7B. The base logit's train recall on this subset is zero. The 1.5B model is the only one of the three that recovers the train more often than the logit does, and it does so by over-predicting the car.

![Accuracy and probability shares by model size](figures/model_size.png)

![Observed shares and mean predicted probabilities](figures/market_shares.png)

### Elasticities and value of time

Own elasticities use the mean predicted probability of the mode whose attribute changed: `((share_new − share_base) / share_base) / (percent / 100)`. Intervals are a respondent bootstrap. All **24** base-MNL elasticities in `results/elasticities.csv` are negative, and each interval excludes zero. The 1.5B and 3B rows in that file are on the 84-task sample. The 7B rows are on the 36-task subset. An elasticity with a near-zero base share is not a useful number; the change in the probability is.

The 1.5B model does not follow the logit. Every train-time elasticity is positive: a slower train receives more train probability. The intervals at −50%, +20%, and +50% lie entirely above zero.

| Shock | Sample | MNL probability | MNL elasticity | 1.5B probability | 1.5B elasticity |
| --- | --- | --- | ---: | --- | ---: |
| Train time, −50% | 84 | 12.5% → 29.1% | −2.68 | 28.0% → 25.8% | +0.16 (0.03 to 0.28) |
| Train time, +50% | 84 | 12.5% → 5.4% | −1.14 | 28.0% → 30.1% | +0.16 (0.04 to 0.31) |
| Swissmetro time, +50% | 84 | 60.0% → 48.4% | −0.38 | 13.5% → 12.6% | −0.13 (−0.27 to +0.01) |
| Swissmetro cost, +50% | 76 paying | 58.2% → 45.8% | −0.42 | 14.0% → 12.9% | −0.16 (−0.27 to −0.04) |

The last row is the one 1.5B cost response whose interval excludes zero and has the expected sign. The share moves by about one percentage point. A +20% train-fare shock has the wrong sign and an interval above zero (elasticity +0.34, interval 0.01 to 0.70). The other 1.5B cost intervals include zero.

The 3B model is saturated. On the 84-task sample its Swissmetro probability stays above 99.3% under every ±50% shock. Making the train 50% faster moves the train probability from about 0 to 1.6%. A ±10 minute or ±10 CHF shift moves the Swissmetro probability by less than a hundredth of a percentage point.

The 7B model does move, on the 36-task subset, when the shock is large. Making the train 50% faster raises the train probability from 0.16% to 24.5%. Making Swissmetro 50% slower lowers its probability from 99.8% to 80.0% (elasticity −0.40, interval −0.63 to −0.19). Making Swissmetro 50% more expensive lowers it from 99.8% to 97.5%, and that interval includes zero. The elasticity on the train's near-zero base share is not reported: the level change is the result.

A central difference of ±10 minutes and ±10 CHF, on 76 tasks where neither shift is clipped at zero, recovers **72.6 CHF/hour** from the base MNL for the train and again **72.6** for Swissmetro. That matches `60 * B_TIME / B_COST` on the training respondents (72.6, cluster s.e. 16.2). The same arc on the 31 unclipped tasks inside the 7B subset is also 72.6. The bootstrap interval of this arc ratio is only a few thousandths of a franc wide, because the ratio is pinned by two coefficients. The cluster standard error on the coefficient ratio is the uncertainty that belongs in a sentence.

No language-model row has `interpretable: true`. The code requires a negative time derivative, a negative cost derivative, and a ±10 CHF movement of at least half a percentage point. The 1.5B derivatives have the wrong sign. The 3B cost movement is far below that floor. At 7B the train's ±10 CHF shift does move the share by about 3.0 percentage points and in the right direction, but the ±10 minute shift has the wrong sign, so the ratio is negative and is not a value of time. The 7B Swissmetro cost shift is about 0.05 percentage points; dividing by it produces a ratio near 1,430 CHF/hour, which is the artefact the rule is there to refuse.

![Own probability when one attribute is rescaled](figures/elasticities.png)

### Where the predictors diverge

**Annual pass.** On the 41 pass-holder tasks in the 420 (16 people), the observed shares are 73.2% train, 24.4% Swissmetro, 2.4% car. Mean train probability: 78.3% at 1.5B, 0.002% at 3B, 0.04% at 7B, and 16.6% for the base MNL. Without the pass (379 tasks, 131 people) the observed train share is 7.4%, and the train probabilities are 24.3%, 0.06%, 7.4%, and 13.0%. The pass gap in the train share is +65.8 percentage points in the data, +54.0 at 1.5B, about zero at 3B, −7.3 at 7B, and +3.6 in the base logit, which has no pass coefficient beyond the zero fare.

Saying the words "General Abonnement" matters, and the sign depends on the model. On those same 41 tasks, with the fares still at zero but the pass not mentioned, the 1.5B train probability falls from 78.3% to 60.1% (mean absolute change in the three probabilities 0.122). The 3B model stays on Swissmetro (mean absolute change 0.00009). The 7B train probability rises from 0.04% to 70.2%, and Swissmetro falls from 99.96% to 29.8% (mean absolute change 0.468). The sentence that names the pass is what takes the 7B model off the train. Sixteen people is a small base for anything sharper than the sign of that change.

**Equal attributes.** On the 84 elasticity tasks, time is set to 90 minutes, cost to 50 CHF, and headway to 30 for every available mode. The car stays unavailable where it was unavailable. Mean probabilities, train / Swissmetro / car: base MNL 22.6 / 44.7 / 32.7; 1.5B 25.9 / 27.4 / 46.7; 3B 11.5 / 88.3 / 0.2; 7B 28.5 / 71.5 / 0.0. Hard predictions on this probe: 69.0% car at 1.5B, 92.9% Swissmetro at 3B, and 75% Swissmetro / 25% train at 7B. Swissmetro is the fast mode in the raw file (mean door-to-door time 84.5 minutes, against 166.1 for the train and 148.7 for the car when the car exists). The car prior is specific to the 1.5B model. At 3B and 7B the prior is Swissmetro, and equalising the attributes does not remove it.

A +10 CHF fare imposed on pass holders in the 84-task subsample covers 8 tasks. The 7B shock subset contains 5 pass-holder tasks, below the threshold of 8, so that check is not repeated for 7B. The 8-task check is stored in `results/summary.json` under `ga_plus_10chf` and is too small to interpret.

![Pass-holder gap and equal-attribute probe](figures/bias_checks.png)

Prompt times on four CPU threads, after the prefix cache was warm: about 0.82 seconds at 1.5B, 1.70 seconds at 3B, and 3.72 seconds at 7B. The 1.5B grid was scored in 43.9 minutes. The 7B pass, with the 36-task shock grid, took 102 minutes. A second `bash run.sh` reloads the caches.

## Reproduce

```bash
bash run.sh
```

The script creates `.venv`, installs `requirements.txt`, downloads the Swissmetro file if the checksum fails, downloads any missing GGUF into `models/` (gitignored; about 1.1 GB, 2.0 GB, and 4.4 GB across the two 7B shards), and runs `python -m src.run_all`. A completed cache under `results/cache/` is reused, so a second run recomputes the logit and reloads predictions instead of rescoring. Flags:

```bash
bash run.sh --skip-llm          # logit, split, and coefficient figure only
bash run.sh --pilot-only        # format check on 24 training respondents
bash run.sh --models 1.5b,3b,7b # the default
bash run.sh --llm-n 420 --elast-n 84
```

Tests, which do not load the language-model weights:

```bash
.venv/bin/python -m unittest tests.test_mnl tests.test_prompts
```

Stack: Python 3.12, NumPy, pandas, SciPy, Matplotlib, llama-cpp-python. No GPU.

## Repository

| Path | Contents |
| --- | --- |
| `src/data_utils.py` | Filters, General Abonnement costs, respondent split, shocks |
| `src/mnl.py` | Logit likelihood, Newton fit, cluster standard errors, value of time |
| `src/llm_choice.py` | Local model, digit-token probabilities |
| `src/prompts.py` | Persona and scenario text |
| `src/evaluate.py` | Accuracy, bootstrap, elasticities |
| `src/run_all.py` | One pipeline, writes `results/` and `figures/` |
| `prompts/` | The wording that was scored |
| `results/summary.json` | Machine-readable source for the tables above |
| `results/mnl_coefficients.csv` | Every coefficient and standard error |
| `results/elasticities.csv` | Every shock |
| `src/model_specs.py` | 1.5B, 3B, and 7B weight files and cache names |
| `results/model_size.csv` | Accuracy, shares, and the gap versus the logit |
| `results/implied_vot_common.csv` | Arc value of time on the shared 31-task subset |
| `results/llm_predictions.csv` | 1.5B rows; `llm_predictions_3b.csv` and `llm_predictions_7b.csv` are the others |
| `docs/explainer.md` | The same material in interview language |
| `report.md` | Short write-up of the same run |

If a number in this file disagrees with `results/summary.json`, the JSON wins. Rounding in the tables is three decimals for coefficients, one decimal for percentages and for CHF/hour, and the log-likelihoods are printed at the precision shown.

## Limitations

- The language models are Qwen2.5-Instruct at 1.5B, 3B, and 7B, and the weights are 4-bit. A fine-tune on the training respondents, or a different family, could behave differently. This run does not speak for "LLMs" as a class. Size, inside this family and this prompt, did not close the gap.
- The data are stated preference about a mode that does not exist. Sensitivity to time on the form can exceed sensitivity on a real Tuesday.
- The filtered sample is mostly business trips and mostly men.
- The language models are scored on 420 of 1,350 test tasks. Elasticities use 84 tasks for 1.5B and 3B, and 36 tasks for 7B. The pass-wording check uses 41 tasks from 16 people.
- The logit sees 602 respondents from this survey. The language models see a zero-shot prompt. The comparison is informative about simulation without fitting. It is an easy win for the logit, and a larger Qwen does not take the win away.
- Tasks within a person are dependent. Cluster standard errors and the respondent bootstrap are there for that reason. McNemar p-values in the JSON treat tasks as independent and are anti-conservative; they are not the intervals quoted above.
- The pass coefficient is not causal. The base model never hard-predicts the train, so accuracy understates it.
- One prompt wording. Origin, destination, first class, ticket type, and group size are not in the prompt.
- Percentage cost shocks do not move a zero fare. The arc value of time drops tasks that would clip at zero.
- The logit assumes independence of irrelevant alternatives. A nested logit for train and Swissmetro is the obvious relaxation, and it was left out so the baseline would stay the published model.
- The survey is public and widely taught. The equal-attribute probe is the check on a label prior. It shows a car prior at 1.5B and a Swissmetro prior at 3B and 7B.

## What I would do next

I would keep this prompt and this 420-task file frozen, and change one thing at a time.

1. A mixed logit with a person-level random coefficient on time, which is what the cluster standard errors are asking for, and a nested logit with train and Swissmetro in one nest.
2. The same frozen prompt on a model from a different family, or a fine-tune of one of these three on the training respondents, scored once on these 420 tasks. The Qwen size comparison is the experiment this repository already ran.
3. A small set of prompt variants scored only on training respondents. The test people get touched once, after the wording is locked.
4. A model fine-tuned on the training respondents' choices, evaluated on the held-out people, reported next to this zero-shot baseline so the gain from fitting is visible.
5. A reliability diagram for both predictors. Accuracy will keep hiding the base logit's train probabilities.
6. A revealed-preference dataset, where Swissmetro is not one of the labels and the annual-pass coding can be checked against actual tickets.

I would not put 70.7 CHF/hour into a cost-benefit spreadsheet, and I would not replace a mode-choice model with this language model.

## Licence and citation

Code and write-up: MIT, copyright Veeraj Jain, 2026. See `LICENSE`.

The Swissmetro file is not covered by that licence. Cite:

Bierlaire, M., Axhausen, K., and Abay, G. (2001). The acceptance of modal innovation: The case of Swissmetro. *1st Swiss Transport Research Conference*.

Variable codes follow the EPFL description, `https://transp-or.epfl.ch/documents/technicalReports/CS_SwissmetroDescription.pdf`. The logit benchmark is the Biogeme Swissmetro example `b01a`.
