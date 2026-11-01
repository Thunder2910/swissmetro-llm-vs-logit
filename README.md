# Can LLMs Simulate Travellers?

**Veeraj Jain** — undergraduate, Civil Engineering, Indian Institute of Technology Delhi.

October 2026.

Can a language model stand in for a traveller? Give it the person and the alternatives, and read its reply as a mode choice. This repository tests that idea on one public survey, against the multinomial logit usually taught with that survey.

The language models are Qwen2.5-Instruct at 1.5B, 3B, and 7B parameters. The prompt is frozen. Nothing is sampled. The same 420 held-out tasks are scored for every size. The logit was fitted to other people from the same survey. The language models never see those choices.

## The question

A person is offered a trip and up to three ways to make it: the existing train, a proposed maglev called Swissmetro, and a car. They pick one. The practical question is whether a local open-weights model, given the same attributes a logit sees, reproduces the logit's choices, market shares, and trade-off between time and money.

## Data

The survey is Swissmetro (Bierlaire, Axhausen, and Abay, 2001), a stated-preference study of intercity travel in Switzerland. The file `data/swissmetro.dat` is the public release from the Transport and Mobility Laboratory at EPFL:

<https://transp-or.epfl.ch/data/swissmetro.dat>

SHA-256 `27432693cf052985d79a950b4b888be3efca798fc89b0d3ffefe40608ede00f2`. `bash run.sh` checks that checksum and downloads the file only if it is missing.

The estimation sample follows the Biogeme filters: commute or business trips, and a stated choice. That is 6,768 tasks from 752 people. For someone who holds the General Abonnement, the raw train "cost" is the annual pass price, not the fare of the trip. Train and Swissmetro trip costs are set to zero for those people, which is what the Biogeme example does, and the prompt says so in words.

**The Swissmetro file is not covered by the MIT licence on this code.** It is redistributed unchanged so the study can be re-run. Cite:

> Bierlaire, M., Axhausen, K., and Abay, G. (2001). The acceptance of modal innovation: The case of Swissmetro. *1st Swiss Transport Research Conference*.

Variable codes follow the EPFL description, <https://transp-or.epfl.ch/documents/technicalReports/CS_SwissmetroDescription.pdf>.

## What is compared

**Multinomial logit.** The baseline is Biogeme specification `b01a`: one time coefficient, one cost coefficient, and alternative-specific constants for train and car. Swissmetro's constant is fixed at zero. On the full sample the fit matches the published example (log-likelihood −5,331.252). A richer logit adds the annual pass, headway, trip purpose, age, and income. People are split by respondent, seed 42: 602 people to estimate, 150 people held out. Standard errors are clustered by person.

**Language models.** Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, and Qwen2.5-7B-Instruct, each as a 4-bit `Q4_K_M` GGUF, run locally with llama.cpp. No API key. The reply is a single digit: 1 train, 2 Swissmetro, 3 car. The probability of each mode is the next-token probability of that digit, renormalised over modes that exist in the task. The scored sample is 420 tasks from 147 held-out people. The 7B model is slow on CPU (about 3.7 seconds per prompt), so its time and cost shocks use 36 of the 84 shock tasks. Accuracy, shares, the pass check, and the equal-attribute probe use the same tasks at every size.

## Results

On the same 420 tasks. Intervals are a respondent bootstrap.

| Model | Accuracy | Log-likelihood | Mean probability (train / Swissmetro / car) |
| --- | ---: | ---: | --- |
| Always Swissmetro | 57.4% | −381.03 | hard prediction is always Swissmetro |
| Base logit | 63.1% | −357.22 | 13.4% / 58.0% / 28.7% |
| Qwen2.5-1.5B | 34.3% | −693.20 | 29.6% / 12.7% / 57.7% |
| Qwen2.5-3B | 57.4% | −1,973.34 | 0.05% / 99.95% / 0.00% |
| Qwen2.5-7B | 53.8% | −2,819.35 | 6.7% / 93.3% / 0.04% |

Observed shares on these tasks are 13.8% train, 57.4% Swissmetro, 28.8% car. The logit's probabilities sit on those shares. Every number is written by `src/run_all.py` into `results/summary.json`. If this page and that file disagree, the JSON wins.

## Findings

**The car bias at 1.5B flips to a Swissmetro bias at 3B and 7B.** The 1.5B model puts 57.7% of its probability on the car, and 46.7% even after time, cost, and headway are set equal. The 3B model puts 99.95% on Swissmetro and hard-predicts it on every task, which is why its accuracy equals "always Swissmetro". The 7B model puts 93.3% on Swissmetro and never selects the car. With attributes equalised, Swissmetro still gets 88.3% at 3B and 71.5% at 7B, against 44.7% for the logit. Accuracy moves toward the majority rule. Log-likelihood gets worse, because the larger models give almost no probability to people who chose the train or the car.

**There is no usable language-model value of time.** A ±10 minute and ±10 CHF shift recovers 72.6 CHF/hour from the logit, matching its coefficients. The same shift fails a simple rule for all three Qwen models: both time and cost must move the share in the expected direction, and the cost shift must be large enough to divide by. The 7B model does react to a large shock — making Swissmetro 50% slower cuts its probability from 99.8% to 80.0% — but that is not a value of time.

**The annual-pass effect vanishes with size.** Among 41 pass-holder tasks the observed train share is 73.2%. The 1.5B model puts 78.3% of its probability on the train for those people, and much less for everyone else. The 3B and 7B models put about 0% on the train for pass holders. At 7B, deleting the sentence that names the pass (and leaving the fares at zero) raises the train probability from 0.04% to 70.2%. The sentence that helped the small model pushes the larger one off the train. Sixteen people is a thin sample for anything sharper than that.

## Limitations

- Three 4-bit sizes of one model family, and one prompt. This is not a result about language models in general.
- The models are not fitted to the survey. A fine-tune on the training respondents is a different experiment.
- The data are stated choices about a maglev that was never built. Most respondents are men on business trips.
- The language models are scored on 420 of 1,350 held-out tasks. Time and cost shocks for 7B use 36 tasks, not the 84 used for 1.5B and 3B.
- Tasks from the same person are not independent. The intervals above resample people. Tests in the JSON that treat tasks as independent overstate precision.

## Reproduce

```bash
bash run.sh
```

That creates a virtual environment, installs `requirements.txt`, checks the Swissmetro file, downloads any missing GGUF into `models/`, and runs the pipeline. A finished cache in `results/cache/` is reused, so a second run does not rescore the language models.

The weight files are not in git. They are about 1.1 GB, 2.0 GB, and 4.4 GB (the 7B model is two shards). `bash run.sh` downloads them from Hugging Face. URLs and expected byte counts are in `src/model_specs.py` and `models/README.md`. No API key is required. The run is CPU-only.

```bash
bash run.sh --skip-llm            # logit only
bash run.sh --models 1.5b,3b,7b   # the default
.venv/bin/python -m unittest tests.test_mnl tests.test_prompts
```

Stack: Python 3.12, NumPy, pandas, SciPy, Matplotlib, llama-cpp-python.

## Where the detail lives

| Path | What it is |
| --- | --- |
| `report.md` / `report.pdf` | The write-up |
| `docs/explainer.md` | The same study in interview language |
| `results/summary.json` | Every number the code wrote |
| `results/model_size.csv` | The table above |
| `figures/` | Confusion matrices, shares, elasticities, model size |

## Licence

Code and write-up: MIT. Copyright Veeraj Jain, 2026. See `LICENSE`.

The Swissmetro data file is not covered by that licence. Cite Bierlaire, Axhausen, and Abay (2001).
