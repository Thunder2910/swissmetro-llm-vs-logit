# Mode choice, in plain language

This note is for a second-year civil engineering student who may have to
explain the repository in an interview. It uses the numbers this code
actually estimated. The LLM comparison sample is summarised at the end and
in `report.md`; if those two disagree, `results/summary.json` wins.

## 1. What decision is being modelled?

A person is offered a trip between two Swiss cities and up to three ways to
make it: the existing train, a hypothetical high-speed maglev called
Swissmetro, and a car. They pick one. The data are *stated preference*:
the alternatives were constructed by a survey, not observed from tickets
or traffic counts. People can answer strategically, or can be more
sensitive to time on a form than they are on a Tuesday morning. That is a
limitation of the whole file, not only of the LLM.

Each person faces nine such tasks. Those nine answers are not nine
independent people. Standard errors that pretend they are come out too
small. The tables therefore report *cluster-robust* standard errors, with
one cluster per respondent.

## 2. Utility

Write the attractiveness of an alternative as a number, \(V\). Higher is
better. For the base model, with Swissmetro as the zero of the scale:

\[
\begin{align*}
V_{\text{train}} &= \text{ASC}_{\text{train}} + \beta_t \frac{T_{\text{train}}}{100} + \beta_c \frac{C_{\text{train}}}{100} \\
V_{\text{SM}} &= \beta_t \frac{T_{\text{SM}}}{100} + \beta_c \frac{C_{\text{SM}}}{100} \\
V_{\text{car}} &= \text{ASC}_{\text{car}} + \beta_t \frac{T_{\text{car}}}{100} + \beta_c \frac{C_{\text{car}}}{100}
\end{align*}
\]

\(T\) is door-to-door time in minutes. \(C\) is what the person would pay
for *this trip*, in francs. Dividing by 100 is the scaling in the Biogeme
example. It does not change the model. It changes the units of
\(\beta_t\) and \(\beta_c\): each is the utility of a 100-minute or
100-franc change.

The alternative-specific constants (ASCs) pick up everything that is not
time or cost: comfort, reliability, the fact that Swissmetro is new, the
fact that a car is parked at the end. One ASC has to be fixed, or the
model cannot tell "all utilities up by 5" from the original utilities.
Swissmetro's ASC is fixed at 0. A negative train ASC means that, at equal
time and equal cost, the train is less attractive than Swissmetro.

On the full filtered sample the base estimates are:

| Parameter | Estimate | Cluster s.e. | Cluster \(t\) |
| --- | ---: | ---: | ---: |
| ASC train | −0.701 | 0.183 | −3.82 |
| ASC car | −0.155 | 0.129 | −1.20 |
| Time (per 100 min) | −1.278 | 0.238 | −5.38 |
| Cost (per 100 CHF) | −1.084 | 0.161 | −6.72 |

Time and cost have the expected negative signs. The car ASC is the number
everyone quotes from Biogeme as "significant". It is significant if each
task is treated as a separate person (robust \(t = -2.66\)). It is not
significant once the nine tasks of each person are clustered
(\(t = -1.20\)). That is a real result of re-estimating the model, not a
failure to match Biogeme. The observation-level robust standard errors
match the published ones; see `results/mnl_summary.json`.

## 3. From utility to a probability

People do not always pick the alternative with the highest \(V\). The logit
model says the probability of alternative \(i\), among the ones that are
actually available, is

\[
P_i = \frac{e^{V_i}}{\sum_{j \text{ available}} e^{V_j}}.
\]

If only the train and Swissmetro are available, the car is left out of the
denominator. In this filtered file the train and Swissmetro are always
available; the car is unavailable for 129 of the 752 people.

Two properties are worth being able to say out loud:

- Only *differences* in \(V\) matter. Adding 10 to every utility does not
  change \(P\).
- The ratio \(P_i / P_j\) does not depend on the attributes of a third
  mode. That is the independence of irrelevant alternatives. It is a
  convenience and often a bad description of trains and a maglev, which
  are closer substitutes than either is to the car. A nested logit would
  relax it. This study stays with the multinomial logit because that is
  the published baseline.

## 4. How the coefficients are estimated

The likelihood of the sample is the product of the probabilities of the
chosen alternatives. One maximises the log:

\[
LL = \sum_n \log P_{n, \text{chosen}}.
\]

For a linear-in-parameters logit this surface is concave, so Newton steps
from the origin reach the same maximum Biogeme reports. The check is
numerical, not a claim from memory: final log-likelihood \(-5331.252\) on
6,768 tasks, against a null log-likelihood of \(-6964.663\) in which every
available mode is equally likely. Null here means "coefficients at zero",
not "constants only".

Rho-squared is

\[
\rho^2 = 1 - \frac{LL}{LL_0} = 0.235, \qquad
\bar\rho^2 = 1 - \frac{LL - K}{LL_0} = 0.234
\]

with \(K = 4\) estimated parameters. \(\bar\rho^2\) charges the model for
the number of parameters. Neither number is an \(R^2\) you can describe as
"percent of variance explained". A value around 0.23 against equal shares
is ordinary for this specification.

The \(t\)-statistic is the estimate divided by its standard error. Under
the usual normal approximation, \(|t| > 1.96\) is the 5% two-sided
threshold. Prefer the cluster \(t\) when you talk about what is known.

## 5. Value of time

A minute and a franc both change utility. The value of time is the franc
amount that compensates for one minute, so utility stays the same:

\[
\frac{\partial V}{\partial \text{minute}} = \frac{\beta_t}{100}, \qquad
\frac{\partial V}{\partial \text{CHF}} = \frac{\beta_c}{100}.
\]

The hundreds cancel:

\[
\text{VOT} = \frac{\beta_t}{\beta_c} \text{ CHF per minute}
= 60 \times \frac{\beta_t}{\beta_c} \text{ CHF per hour}.
\]

Both betas are negative, so the ratio is positive.
\(60 \times 1.27786 / 1.08379 = 70.7\) CHF per hour. The cluster-robust
delta-method standard error is 13.8, so a 95% interval is about 44 to 98
CHF/hour.

That is a high number next to many appraisal values. It is the trade-off
*inside this stated-preference design*, which offers large time savings on
a mode that does not exist. It is not a national value of time, and it
should not be dropped into a cost-benefit spreadsheet without saying where
it came from.

In the richer model the same ratio for a non-senior person who does not
have a high income is 64.8 CHF/hour (s.e. 16.7). People aged 55 or over
have a less negative time coefficient, and the implied value of time falls
to about 32 CHF/hour. The interaction itself has cluster \(t = 3.04\). The
high-income cost interaction has the expected sign (less cost-sensitive,
higher value of time, point estimate about 96 CHF/hour) but cluster
\(t = 1.35\). Do not describe that income result as established.

Headway (minutes between departures) enters the richer model on the same
per-100-minutes scale. Its coefficient is about 0.44 times the in-vehicle
time coefficient: a minute of headway is disliked a bit under half as much
as a minute on board, in this sample.

## 6. The annual pass

A General Abonnement is a yearly ticket for the Swiss railway and much of
local public transport. In the raw file, `TRAIN_CO` for a pass holder is
the price of the *year*, often a few thousand francs. Using that number as
the price of one trip would be a mistake. The Biogeme specification sets
the train trip cost, and the Swissmetro trip cost, to zero. This code does
the same.

The richer model also allows a pass holder to prefer the train, or
Swissmetro, beyond that free fare. The train shift is large and precise
(cluster \(t = 4.39\)). The Swissmetro shift is about zero
(cluster \(t = 0.15\)). A careful sentence is: *given a fare of zero, pass
holders still choose the existing train more than other people do*. It is
not a causal effect of handing someone a pass. People who already like the
train are the ones who buy the pass. The survey does not separate those
stories.

## 7. Why the split is by person

A random split of *rows* puts some of a person's nine answers in the
training set and some in the test set. The model then partly sees that
person's taste and looks better than it will on a new person. The split
here holds out 150 of 752 respondents (1,350 tasks) and estimates on the
other 602 (5,418 tasks). Seed 42.

On that held-out set the base model, estimated only on the training
people, has:

- accuracy 64.3%
- mean log-likelihood −0.843 per task
- mean predicted probabilities 13.4% train, 57.9% Swissmetro, 28.7% car

The observed shares on the same tasks are 13.9%, 57.3%, 28.8%. The
probabilities are close to the shares. The *hard* prediction (pick the
mode with the highest probability) is not: it predicts the train on
**zero** of the 1,350 tasks. Train is often second. Accuracy therefore
punishes a model whose probabilities are in the right place whenever the
top alternative is Swissmetro and the truth is the train. Quote the
log-likelihood and the probability shares together with the accuracy, or
you will mis-describe the model.

The constants-only model (ASCs, no time or cost) predicts Swissmetro every
time, because both estimated ASCs are negative and Swissmetro is always
available. Its accuracy equals the majority-class rule: 57.3% on the test
people, which is just the Swissmetro share. The 7.0 point gain of the base
model over that rule is exactly the contribution of time and cost. The
richer model reaches 67.4%.

## 8. What the LLM is doing

The language model is not estimated on these choices. For each task it is
shown a short persona (purpose, age, gender, income, who pays, luggage,
whether the person holds a pass) and the time, cost, and headway of each
available mode. It is asked to reply with the digit 1, 2, or 3.

Nothing is sampled. The next-token probabilities of the digit tokens are
renormalised over the modes that exist in that task. Temperature does not
enter. Repeating the run with the same weights and the same prompt file
gives the same distribution. The weights are Qwen2.5-1.5B-Instruct, stored
as a 4-bit GGUF (`Q4_K_M`) and run locally. Quantisation is a further
approximation on top of an already small model.

This is a hard comparison for the LLM and an easy one for the logit. The
logit is fitted to 602 other respondents from this same survey. The LLM
sees none of their choices. Beating the logit would be surprising. Losing
to it is not, by itself, evidence that "LLMs cannot represent travellers".
It is evidence about this model, this prompt, and this sample.

Read the probabilities, not only the argmax, for the same reason as in
section 7. An elasticity is a change in the mean predicted probability
when one attribute is rescaled, divided by the proportional change. A
negative own elasticity means the share of that mode falls when it gets
slower or more expensive. The value of time implied by a pair of shifts is

\[
60 \times \frac{\Delta S / \Delta \text{minutes}}{\Delta S / \Delta \text{CHF}},
\]

using a central difference of ±10 minutes and ±10 francs on tasks where
neither shift hits zero. If the cost shift barely moves the share, do not
divide. A huge ratio would be an artefact of a near-zero denominator, not
a value of time.

## 9. Questions worth being able to answer

**Why fix one ASC?**
Utilities are only identified up to an additive constant. Three free ASCs
plus a full set of probabilities is one parameter too many.

**Why is your car ASC "insignificant" when Biogeme's table says otherwise?**
The coefficient matches Biogeme (−0.155). Their robust \(t\) treats 6,768
tasks as independent. Clustering the nine tasks of each person roughly
doubles the standard error. Say both numbers and which one you trust for
inference.

**Is 70 CHF/hour the Swiss value of time?**
No. It is \(60 \beta_t / \beta_c\) in this stated-preference file, under a
generic time coefficient and a generic cost coefficient. Quote the
standard error. Say it is not an appraisal value.

**Why is the pass holder's train cost zero rather than the large number in the file?**
Because that large number is the annual price. The marginal trip costs
nothing. The description of the data says this explicitly.

**Your base model never predicts the train. Is the model useless?**
Its mean probabilities track the train share (about 13% predicted, 14%
observed on the test people). The highest probability is still usually
Swissmetro, so the 0-1 prediction misses every train chooser. Accuracy is
the wrong single summary. The richer model, which knows who holds a pass,
does predict the train for some of them.

**Why not a nested or mixed logit?**
Both are the right next models. Nested logit would let the train and
Swissmetro share an unobserved component. Mixed logit, with a person-level
random coefficient, is the natural response to the cluster standard
errors. They were left out so the baseline would stay the published
four-parameter model.

**Does the LLM "know" Swissmetro?**
The dataset is public and widely used in teaching. A model could in
principle have seen the aggregate shares during pretraining. On the 84
tasks where time, cost, and headway were set equal, the base MNL still
put 44.7% probability on Swissmetro and the LLM put 46.7% on the car
and 27.4% on Swissmetro. The label prior in this run favours the car.
A 1.5B model is also weak enough that memorising a 1990s survey is not
the most likely explanation of its errors. The numbers are in section 10.

**What would you do if a reviewer said the LLM prompt is arbitrary?**
Agree that one wording is one wording. The files `prompts/system.txt` and
`prompts/user_template.txt` are the wording, and the cache key changes
when they change. The training-person pilot was used only to check that
the reply is a digit. It was not used to pick a wording that scored well.
A next step is a small pre-registered set of prompt variants, scored on
the training people, with the test people touched once.

## 10. What the comparison actually found

These numbers are from `results/summary.json` for one local model:
Qwen2.5-1.5B-Instruct, 4-bit `Q4_K_M`, run with llama.cpp. Nothing was
sampled. The next-token probabilities of the digits 1, 2, and 3 were
renormalised over the modes available in that task. The same prompt file
and the same weights give the same distribution.

The evaluation is 420 tasks from 147 held-out people (at most four tasks
each). Eight tasks had already been used to check that the reply was a
digit and to time the machine; three of those eight were in the test
people, and those three are not among the 420. A 24-person pilot on the
training respondents had a format rate of 1.000 and was not used to
change the wording. Elasticities use 84 of the 420 tasks (71 people).
Cost shocks use the 76 of those 84 in which the person pays a fare.
Percentage changes of a zero fare, which is what a pass holder faces,
would not be a real change.

On those 420 tasks:

| Rule | Accuracy | Mean log-likelihood | Mean probability, train / Swissmetro / car |
| --- | ---: | ---: | --- |
| Always Swissmetro | 57.4% | −0.907 (constants-only) | hard prediction is 100% Swissmetro |
| Base MNL | 63.1% | −0.851 | 13.4% / 58.0% / 28.7% |
| Richer MNL | 67.1% | −0.771 | 12.9% / 57.3% / 29.8% |
| LLM | 34.3% | −1.650 | 29.6% / 12.7% / 57.7% |

Observed shares on the same tasks: 13.8% / 57.4% / 28.8%. A respondent
bootstrap (1,000 resamples of the 147 people) puts the LLM accuracy in
28.1% to 40.7%, and the gap "base MNL minus LLM" in 18.8 to 39.5
percentage points.

Read that table in both directions.

The logit probabilities sit on the observed shares. Its hard prediction
still never selects the train, for the reason in section 7. Accuracy
alone makes the base MNL look like a model that has forgotten the train.
The probabilities have not.

The LLM's hard prediction selects the car on 78.6% of tasks and
Swissmetro on 1.4% (6 of 420). It gets 2 of the 241 Swissmetro choices
right. Its mean probability on the chosen alternative is 0.344, and its
log-likelihood is far below the constants-only logit. Correlation with
the base MNL is 0.38 for the train probability, −0.04 for Swissmetro,
and 0.36 for the car. They agree on the hard prediction 16.0% of the time.

The annual pass is a place where the LLM follows the data and the base
logit does not. Among the 41 pass-holder tasks, the observed train share
is 73.2% and the LLM's mean train probability is 78.3%. Among the 379
other tasks the observed train share is 7.4% and the LLM's is 24.3%.
The base MNL, which only knows that the fare is zero, puts 16.6% on the
train for pass holders and 13.0% for everyone else. Removing the sentence
that names the pass, while leaving the fares at zero, drops the LLM's
train probability on those 41 tasks from 78.3% to 60.1%. The wording
matters, and so do the zeros. Sixteen people is a thin basis for a
stronger claim.

The trade-off check goes the other way. On the 84-task shock sample, every
base-MNL own elasticity for train and Swissmetro time and cost is
negative, and the bootstrap interval excludes zero. Making the train 50%
slower moves the MNL train probability from 12.5% to 5.4% (elasticity
−1.14). The LLM probability moves from 28.0% to 30.1% (elasticity +0.16,
interval +0.04 to +0.31). Making the train 50% faster also lowers the
LLM's train probability. A central difference of ±10 minutes and ±10 CHF
recovers a value of time of 72.6 CHF/hour from the MNL, the same number
as \(60 \beta_t / \beta_c\) on the training respondents. The same
difference applied to the LLM moves the share in the wrong direction for
both time and cost, so the ratio is not reported as a value of time.

A sentence you can defend: this 1.5B model, with this prompt, reproduces
the pass-holder's tilt toward the existing train and does not reproduce
the time and cost trade-off, or the Swissmetro share, that the multinomial
logit was built to capture. That is a result about one small model and
one wording. It is the result this repository measured.
