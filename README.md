# Math 189 Project: Forum Sentiment and Player Counts

**Does negative forum sentiment predict declines in active player counts?**  
MATH 189: Data Analysis & Inference — UC San Diego.

This document summarizes the **mathematical framework** of the project. The full analysis, code, and results live in `project(1).ipynb`.

### Running the notebook

The notebook needs several Python packages. Install them first:

```bash
pip install -r requirements.txt
```

Then open and run `project(1).ipynb` in Jupyter (Lab or Notebook), VS Code, or Colab. If you see `ModuleNotFoundError: No module named 'numpy'` (or similar), you're in an environment where the dependencies aren't installed—run the command above in that environment, or use a kernel that already has the packages.

**Optional (for full functionality):** RoBERTa sentiment and Steam scraping require:

```bash
pip install transformers torch requests beautifulsoup4
```

---

## 1. Formal Panel Model

The main object of inference is a **two-way fixed effects** panel regression:

$$
Y_{it} = \alpha_i + \gamma_t + \beta\, S_{i,t-k} + X_{it}\,\theta + \varepsilon_{it}, \qquad \varepsilon_{it} \sim (0, \sigma^2)
$$

| Symbol | Meaning |
|--------|--------|
| $Y_{it}$ | Log active players for game $i$ at week $t$ |
| $\alpha_i$ | Game fixed effect (baseline popularity, time-invariant) |
| $\gamma_t$ | Time fixed effect (platform-wide shocks, e.g. Steam Sales) |
| $S_{i,t-k}$ | Aggregate negative sentiment for game $i$, lagged by $k$ weeks |
| $X_{it}$ | Controls (update indicators, DLC, seasonal dummies) |
| $\beta$ | **Parameter of interest**: effect of lagged negativity on log players |

Interpretation: a one-unit increase in $S_{i,t-k}$ is associated with an approximate $100\beta\%$ change in player counts at $t$, holding $\alpha_i$, $\gamma_t$, and $X_{it}$ fixed. We test $H_0: \beta = 0$ vs $H_a: \beta < 0$ and construct 95% confidence intervals for $\beta$; if the interval excludes zero, we reject $H_0$ at level $\alpha = 0.05$.

---

## 2. Sentiment Construction and Aggregation

### 2.1 Weekly sentiment score

Let $\mathcal{P}_{it}$ be the set of forum posts for game $i$ in week $t$. Each post $j$ receives a **negativity score** $s_j \in [0,1]$ from a RoBERTa-based classifier. The weekly aggregate is the mean:

$$
S_{it} = \frac{1}{|\mathcal{P}_{it}|} \sum_{j \in \mathcal{P}_{it}} s_j.
$$

So $S_{it} \in [0,1]$ is the average probability of negativity over that week. Using a continuous predictor (rather than a binary flag) allows standard OLS inference: confidence intervals for $\beta$ and interpretable effect sizes.

### 2.2 Central Limit Theorem (CLT) justification

$S_{it}$ is an average over many post-level scores. Even when individual scores are skewed or non-normal, the **Lindeberg–Lévy CLT** gives:

$$
\sqrt{|\mathcal{P}_{it}|}\,\frac{S_{it} - \mu_{it}}{\sigma_{it}} \xrightarrow{d} \mathcal{N}(0, 1)
$$

as $|\mathcal{P}_{it}| \to \infty$, where $\mu_{it} = \mathbb{E}[s_j]$ and $\sigma_{it}^2 = \mathrm{Var}(s_j)$ within that game-week. So weekly aggregates are approximately normal for inference (t-tests, F-tests, CIs).

### 2.3 Poisson model for negative post volume

Beyond *intensity* $S_{it}$, we model the **count** of negative posts per game-week as Poisson:

$$
N_{\mathrm{neg}} \sim \mathrm{Poisson}(\lambda).
$$

We estimate$\hat{\lambda} = \bar{X}$(MLE = sample mean) separately for “decline” vs “normal” weeks and test$H_0: \lambda_{\mathrm{decline}} = \lambda_{\mathrm{normal}}$via a **likelihood ratio test**:

$$
-2\,\ln\frac{\mathcal{L}_0}{\mathcal{L}_1} \sim \chi^2_1.
$$

---

## 3. Stationarity: Augmented Dickey–Fuller (ADF)

Before Granger tests or panel regression, we check that series are (at least weakly) stationary to avoid **spurious regression**.

ADF tests the autoregressive representation $\Delta y_t = \phi\, y_{t-1} + \sum_{j=1}^{p} \psi_j \Delta y_{t-j} + \varepsilon_t$. The null is a **unit root** (non-stationarity):

$$
H_0: \phi = 0 \quad \text{(unit root)} \qquad \text{vs} \qquad H_a: \phi < 0 \quad \text{(stationary)}.
$$

Test statistic has a **Dickey–Fuller distribution** (non-standard); we use the reported p-value. If $\log(\mathrm{players})$ is non-stationary, we work with **first differences** $\Delta y_{it} = y_{it} - y_{i,t-1}$ and re-test.

---

## 4. Granger Causality

**Granger causality** is predictive precedence: does lagged sentiment help predict $Y_t$ beyond lagged $Y$ alone? It does *not* imply structural causation.

### 4.1 VAR setup

- **Restricted:** $Y_t$ regressed only on lags of $Y$:
 $$
  Y_t = \sum_{j=1}^{k} \phi_j Y_{t-j} + \varepsilon_t.
 $$
- **Unrestricted:** add lags of sentiment $S$:
 $$
  Y_t = \sum_{j=1}^{k} \phi_j Y_{t-j} + \sum_{j=1}^{k} \psi_j S_{t-j} + \varepsilon_t.
 $$

Null hypothesis: $H_0: \psi_1 = \psi_2 = \cdots = \psi_k = 0$. We use the usual **nested F-test**:

$$
F = \frac{(\mathrm{SSR}_{\mathrm{restr}} - \mathrm{SSR}_{\mathrm{unrestr}})\,/\,k}{\mathrm{SSR}_{\mathrm{unrestr}}\,/\,(T - 2k - 1)} \sim F(k, T - 2k - 1).
$$

Rejecting$H_0$means “sentiment Granger-causes player counts” at the chosen lag$k$. We also run a **permutation test** for Granger causality to avoid reliance on asymptotic normality.

---

## 5. Fixed Effects Estimation

The model is $Y_{it} = \alpha_i + \gamma_t + \mathbf{x}_{it}'\boldsymbol{\beta} + \varepsilon_{it}$. The **within (demeaning) transformation** removes $\alpha_i$ and $\gamma_t$:

- Entity-demean: $\tilde{Y}_{it} = Y_{it} - \bar{Y}_i - \bar{Y}_t + \bar{Y}$, and similarly for regressors.
- OLS on the transformed data gives the **within estimator** $\hat{\boldsymbol{\beta}}$; this is equivalent to LSDV (least squares dummy variables) with dummies for each $i$ and $t$.

**Hausman test:** compare fixed effects (FE) vs random effects (RE). Under $H_0$: RE is consistent and efficient. Test statistic:

$$
H = (\hat{\boldsymbol{\beta}}_{\mathrm{FE}} - \hat{\boldsymbol{\beta}}_{\mathrm{RE}})' \big(\widehat{\mathrm{Var}}(\hat{\boldsymbol{\beta}}_{\mathrm{FE}}) - \widehat{\mathrm{Var}}(\hat{\boldsymbol{\beta}}_{\mathrm{RE}})\big)^{-1} (\hat{\boldsymbol{\beta}}_{\mathrm{FE}} - \hat{\boldsymbol{\beta}}_{\mathrm{RE}}) \sim \chi^2_{\dim(\boldsymbol{\beta})}.
$$

Rejecting $H_0$ favors FE. We also check **first-differencing**: $\Delta Y_{it} = \Delta \mathbf{x}_{it}'\boldsymbol{\beta} + \Delta\varepsilon_{it}$; significance of $\hat{\beta}$ here supports the FE result.

---

## 6. Block Bootstrap for $\beta$

Asymptotic CIs assume large $N,T$ and normal approximations. **Block bootstrap** (resampling by *game*, preserving time series within each unit):

1. Draw $N$ games with replacement from $\{1,\ldots,N\}$.
2. For each bootstrap sample, stack the selected games’ observations and re-estimate the fixed effects model to get $\hat{\beta}^{*}_b$.
3. Repeat $B$ times. The **bootstrap 95% CI** is $[\hat{\beta}^{*}_{0.025},\, \hat{\beta}^{*}_{0.975}]$.

Bootstrap SE: $\widehat{\mathrm{SE}}_{\mathrm{boot}}(\hat{\beta}) = \sqrt{\frac{1}{B-1}\sum_{b=1}^{B}(\hat{\beta}^{*}_b - \bar{\hat{\beta}}^{*})^2}$. If this CI and the asymptotic CI both exclude zero, inference is robust.

---

## 7. Statistical Power

**Power** $= 1 - P(\mathrm{Type\ II\ error}) = P(\mathrm{reject\ } H_0 \mid H_a \mathrm{\ true})$. For a two-sided test on $\beta$ at level $\alpha$, under $H_a: \beta = \beta_a \neq 0$:

$$
Z = \frac{\hat{\beta} - \beta_a}{\widehat{\mathrm{SE}}(\hat{\beta})} \approx \mathcal{N}\left(\frac{|\beta_a|}{\mathrm{SE}}, 1\right), \qquad \text{Power}(|\beta|) = P\bigl(|Z| > z_{\alpha/2}\bigr).
$$

We plot power as a function of true $|\beta|$ and report the **minimum detectable effect** (smallest $|\beta|$ for which power $\geq 0.80$ at $\alpha = 0.05$).

---

## 8. Markov Chain State Transitions

We discretize each game-week into three states from $\Delta\log(\mathrm{players})$: **Growing**, **Stable**, **Declining**. Under a **first-order Markov chain**:

$$
P(X_{t+1} = j \mid X_t = i,\, X_{t-1},\, \ldots) = P(X_{t+1} = j \mid X_t = i) = p_{ij}.
$$

The **transition matrix** is $\mathbf{P} = (p_{ij})$. We estimate $\hat{p}_{ij}$ by counting transitions, separately in “high-negativity” vs “low-negativity” regimes. If $\hat{p}_{\mathrm{Declining},\cdot}$ is larger in the high-negativity regime, that aligns with “negative sentiment precedes decline” in a Markov framing.

---

## 9. LOESS (Nonparametric Check)

**Locally weighted scatterplot smoothing (LOESS):** at each $x_0$, fit a weighted least squares polynomial using a kernel $w_i = W\bigl(\frac{|x_i - x_0|}{h}\bigr)$. Typical choice: tricube kernel, span $\alpha \in (0,1]$. No global functional form is assumed.

If the LOESS curve of $Y$ on $S$ is approximately **linear**, the linear panel specification is justified; curvature suggests nonlinear terms (e.g. polynomials or splines).

---

## 10. Martingale Property and Variance Ratio

A process $\{Y_t\}$ is a **martingale** with respect to filtration $\mathcal{F}_t$ if

$$
\mathbb{E}[Y_{t+1} \mid \mathcal{F}_t] = Y_t.
$$

Our question: does **sentiment break the martingale property**? If $\mathbb{E}[\Delta Y_{t+1} \mid S_t] \neq 0$, then the martingale difference $d_t = Y_{t+1} - Y_t$ is predictable from $S_t$.

**Tests:**

1. **Martingale difference regression:** Regress $d_t$ on $S_{t-1}$. Under martingale, slope $= 0$. Significant slope ⇒ violation.
2. **Lo–MacKinlay variance ratio:** Under a martingale, $\mathrm{Var}(Y_{t+k} - Y_t) = k\,\mathrm{Var}(Y_{t+1} - Y_t)$, so
  $$
   \mathrm{VR}(k) = \frac{\mathrm{Var}(Y_{t+k} - Y_t)\,/\,k}{\mathrm{Var}(Y_{t+1} - Y_t)} = 1.
  $$
   Deviation from 1 indicates predictability (e.g. mean reversion or momentum).

---

## 11. Binary and Multinomial Logistic Regression

### 11.1 Binary: probability of decline

Define $Y_{i,t+1} = \mathbb{1}\{\Delta\log(\mathrm{players})_{i,t+1} < 0\}$. **Logistic regression** models the log-odds:

$$
\log \frac{P(Y=1 \mid \mathbf{x})}{1 - P(Y=1 \mid \mathbf{x})} = \mathbf{x}'\boldsymbol{\beta} \quad \Rightarrow \quad P(Y=1 \mid \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{x}'\boldsymbol{\beta}}} = \sigma(\mathbf{x}'\boldsymbol{\beta}).
$$

The **odds ratio** for a one-unit increase in $x_j$ is $e^{\beta_j}$. We estimate via **maximum likelihood**; no closed form, so use iterative (e.g. Newton–Raphson or IWLS). Evaluation: ROC curve and **AUC**.

### 11.2 Multinomial: three states (Growing / Stable / Declining)

With $K = 3$ states and reference category $K$, the **multinomial logit** specifies $K-1$ equations:

$$
\log \frac{P(Y=k \mid \mathbf{x})}{P(Y=K \mid \mathbf{x})} = \mathbf{x}'\boldsymbol{\beta}_k, \qquad k = 1,\, \ldots,\, K-1.
$$

Probabilities:

$$
P(Y=k \mid \mathbf{x}) = \frac{e^{\mathbf{x}'\boldsymbol{\beta}_k}}{1 + \sum_{j=1}^{K-1} e^{\mathbf{x}'\boldsymbol{\beta}_j}}, \qquad P(Y=K \mid \mathbf{x}) = \frac{1}{1 + \sum_{j=1}^{K-1} e^{\mathbf{x}'\boldsymbol{\beta}_j}}.
$$

This generalizes the binary logit and complements the Markov transition analysis with a parametric model for how sentiment shifts the distribution over states.

---

## 12. Conditional Probability and$\chi^2$Independence

Define “decline” as$\Delta\log(\mathrm{players}) < 0$in the following week. We compare:

$$
P(\mathrm{decline} \mid S_{it} > \mathrm{median}) \quad \mathrm{vs} \quad P(\mathrm{decline} \mid S_{it} \leq \mathrm{median}).
$$

Under **independence** of “decline” and “high vs low sentiment”, the $2 \times 2$ contingency table satisfies $H_0: p_{ij} = p_{i\cdot} p_{\cdot j}$. The **Pearson chi-squared statistic**:

$$
\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}} \sim \chi^2_1 \quad \text{(under } H_0\text{)}.
$$

Rejecting $H_0$ supports that sentiment and decline are associated.

---

## 13. Cross-Correlation and Placebo

- **Cross-correlation function (CCF)** between $\Delta\log(\mathrm{players})$ and $\mathrm{negSentiment}$ at lags $k$: a significant negative correlation at $k > 0$ suggests sentiment *leads* player drops.
- **Placebo test:** Regress *current* player change on *future* sentiment. Under a causal (or predictive) story, future sentiment should not predict current change; significant coefficient would suggest reverse causation or confounding.

---

## Summary

| Component | Main math |
|-----------|-----------|
| Panel model | Two-way FE: $Y_{it} = \alpha_i + \gamma_t + \beta S_{i,t-k} + X_{it}\theta + \varepsilon_{it}$; test $H_0: \beta = 0$. |
| Sentiment | $S_{it} = \frac{1}{\lvert\mathcal{P}_{it}\rvert}\sum_{j \in \mathcal{P}_{it}} s_j$; CLT for normality of $S_{it}$. |
| Granger | Nested VAR; F-test $H_0: \psi_1 = \cdots = \psi_k = 0$. |
| Inference | Asymptotic + block bootstrap CIs for $\beta$; power and MDE. |
| Robustness | ADF stationarity; Hausman (FE vs RE); first-difference; LOESS; Markov; martingale; logistic/multinomial; placebo; $\chi^2$. |

For data pipeline, EDA, and full results, see **`project(1).ipynb`**.
