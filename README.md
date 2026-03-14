# Math 189 Project: Forum Sentiment and Player Counts

**Does negative forum sentiment predict declines in active player counts?**
MATH 189: Data Analysis & Inference — UC San Diego, Winter 2026.

Authors: Jiho Lee, Kevin Xu, Viki Shi, Sharon Tey, Favio Espejo, Ankita Inamti

Full analysis and results: `final_draft.ipynb` | Report: `report/report.pdf`

---

## Running the Project

```bash
pip install -r requirements.txt
# Optional: RoBERTa sentiment + Steam scraping
pip install transformers torch requests beautifulsoup4

# Regenerate all figures
python3 report/generate_figures.py
```

---

## 1. Formal Panel Model

The main object of inference is a **two-way fixed effects** panel regression on first-differenced log player counts:

$$
\Delta\log Y_{it} = \alpha_i + \gamma_t + \beta\, S_{i,t-1} + X_{it}\,\theta + \varepsilon_{it}
$$

| Symbol | Meaning |
|--------|---------|
| $\Delta\log Y_{it}$ | Week-over-week change in log active players for game $i$ at week $t$ |
| $\alpha_i$ | Game fixed effect (time-invariant heterogeneity: genre, studio quality, baseline popularity) |
| $\gamma_t$ | Time fixed effect (platform-wide shocks: Steam sales, holiday seasons) |
| $S_{i,t-1}$ | Aggregate negative sentiment for game $i$, lagged one week |
| $X_{it}$ | Controls (seasonal sale indicator) |
| $\beta$ | **Parameter of interest**: effect of lagged negativity on log player change |

**Hypotheses:** $H_0: \beta = 0$ vs $H_1: \beta < 0$.

---

## 2. Sentiment Construction and the CLT

### 2.1 Weekly Sentiment Score

Let $\mathcal{P}_{it}$ be the set of Steam reviews for game $i$ in week $t$. Each review $j$ is classified by a RoBERTa transformer, yielding a negativity score $s_j \in [0,1]$. The weekly aggregate is:

$$
S_{it} = \frac{1}{|\mathcal{P}_{it}|} \sum_{j \in \mathcal{P}_{it}} s_j \in [0,1]
$$

### 2.2 Theorem (Lindeberg–Lévy CLT)

**Theorem.** Let $s_1, s_2, \ldots, s_n \overset{\text{i.i.d.}}{\sim} F$ with $\mathbb{E}[s_j] = \mu$ and $\text{Var}(s_j) = \sigma^2 < \infty$. Then:

$$
\sqrt{n}\,\frac{\bar{s}_n - \mu}{\sigma} \xrightarrow{d} \mathcal{N}(0,1) \quad \text{as } n \to \infty
$$

**Application.** Even though individual review scores are skewed and non-normal, the weekly average $S_{it}$ is approximately normal for large $|\mathcal{P}_{it}|$. This justifies using $S_{it}$ in OLS regressions with standard $t$- and $F$-based inference (t-tests, confidence intervals, F-tests for Granger causality).

---

## 3. Stationarity: Augmented Dickey–Fuller

### 3.1 The ADF Test

To avoid spurious regression, we require stationarity. The ADF model is:

$$
\Delta y_t = \alpha + \delta\, y_{t-1} + \sum_{j=1}^{p} \phi_j\, \Delta y_{t-j} + \varepsilon_t
$$

**Hypotheses:** $H_0: \delta = 0$ (unit root, non-stationary) vs $H_1: \delta < 0$ (stationary).

**Key property.** Under $H_0$, the ADF $t$-statistic does **not** follow a standard normal or $t$-distribution. It follows the **Dickey–Fuller distribution**, which is left-skewed with more negative critical values than $\mathcal{N}(0,1)$ (e.g., the 5% critical value is approximately $-2.86$ for a model with intercept, vs. $-1.645$ for a one-sided normal test).

**Result.** Of 30 series tested (2 variables $\times$ 15 games), 5 are stationary at $\alpha = 0.05$. First-differencing $\log(\text{players})$ removes the unit root for the regression panel.

---

## 4. Granger Causality

**Definition (Granger Causality).** Series $\{S_t\}$ *Granger-causes* $\{Y_t\}$ if lagged values of $S$ improve the mean-squared prediction of $Y$ beyond what is achievable using lagged $Y$ alone.

### 4.1 Nested F-Test

**Restricted model** (own lags of $Y$ only):

$$
Y_t = c + \sum_{j=1}^{p} \phi_j Y_{t-j} + \varepsilon_t^R
$$

**Unrestricted model** (own lags + $q$ sentiment lags):

$$
Y_t = c + \sum_{j=1}^{p} \phi_j Y_{t-j} + \sum_{j=1}^{q} \psi_j S_{t-j} + \varepsilon_t^U
$$

**Null hypothesis:** $H_0: \psi_1 = \psi_2 = \cdots = \psi_q = 0$.

**Theorem (F-test under $H_0$).** If $H_0$ holds and errors are i.i.d. normal:

$$
F = \frac{(\text{RSS}_R - \text{RSS}_U)/q}{\text{RSS}_U/(T - 2p - q - 1)} \sim F(q,\; T - 2p - q - 1)
$$

Rejection means "sentiment Granger-causes player counts" at lag $q$.

**Result.** 7 of 15 games reject $H_0$ at $\alpha = 0.05$; an 8th (Counter-Strike 2) is marginally significant at $\alpha = 0.10$.

### 4.2 Permutation Test (Distribution-Free)

**Procedure.** Under $H_0$, the temporal ordering of $\{S_{it}\}$ is uninformative. Randomly permuting $S_{it}$ within each game (breaking the temporal link while preserving marginal distributions) gives a reference distribution for $F$:

$$
p_{\text{perm}} = \frac{1}{B} \sum_{b=1}^{B} \mathbf{1}\!\left(F^{*(b)} \geq F_{\text{obs}}\right)
$$

**Lemma (Exact validity).** The permutation $p$-value is exact under $H_0$ in finite samples — no asymptotic approximation is required. This follows directly from the exchangeability of the permuted series under the null.

**Result.** 4 of 15 games survive permutation testing at $\alpha = 0.05$: For Honor ($p < 0.001$), PUBG ($p < 0.001$), GTA V ($p = 0.024$), Rainbow Six Siege ($p = 0.029$). Number of replications: $B = 2{,}000$.

---

## 5. Fixed Effects Estimation

### 5.1 The Within Estimator

**Theorem (Within/Demeaning Estimator).** For the two-way FE model $Y_{it} = \alpha_i + \gamma_t + \mathbf{x}_{it}'\boldsymbol{\beta} + \varepsilon_{it}$, define the within-transformed variables:

$$
\ddot{Z}_{it} = Z_{it} - \bar{Z}_{i\cdot} - \bar{Z}_{\cdot t} + \bar{Z}_{\cdot\cdot}
$$

where $\bar{Z}_{i\cdot} = T^{-1}\sum_t Z_{it}$, $\bar{Z}_{\cdot t} = N^{-1}\sum_i Z_{it}$, $\bar{Z}_{\cdot\cdot} = (NT)^{-1}\sum_{i,t} Z_{it}$.

The within estimator is OLS on the demeaned data:

$$
\hat{\boldsymbol{\beta}}_{\text{FE}} = \left(\sum_{i,t} \ddot{\mathbf{x}}_{it}\ddot{\mathbf{x}}_{it}'\right)^{-1} \sum_{i,t} \ddot{\mathbf{x}}_{it}\ddot{Y}_{it}
$$

This is algebraically equivalent to OLS with $N + T - 1$ dummy variables (LSDV), but computationally more efficient.

**Corollary (Consistency).** Under strict exogeneity $\mathbb{E}[\varepsilon_{it} \mid \mathbf{x}_{i1},\ldots,\mathbf{x}_{iT},\alpha_i,\gamma_t] = 0$ and standard regularity conditions, $\hat{\boldsymbol{\beta}}_{\text{FE}}$ is consistent as $N \to \infty$ or $T \to \infty$.

**Corollary (Asymptotic Normality).** Under consistency conditions:

$$
\sqrt{N}\,(\hat{\boldsymbol{\beta}}_{\text{FE}} - \boldsymbol{\beta}) \xrightarrow{d} \mathcal{N}\!\left(\mathbf{0},\; \boldsymbol{\Sigma}^{-1}\boldsymbol{\Omega}\boldsymbol{\Sigma}^{-1}\right)
$$

where $\boldsymbol{\Sigma} = \mathbb{E}[\ddot{\mathbf{x}}_{it}\ddot{\mathbf{x}}_{it}']$ and $\boldsymbol{\Omega}$ is the long-run variance accounting for within-game serial correlation. The sandwich estimator $\hat{\boldsymbol{\Sigma}}^{-1}\hat{\boldsymbol{\Omega}}\hat{\boldsymbol{\Sigma}}^{-1}$ gives the clustered standard errors used throughout.

**Key results (Model 2, two-way FE):**
- $\hat{\beta} = -0.649$, clustered SE $= 0.316$, $p = 0.040$
- 95% CI $= [-1.268,\,-0.030]$
- Multi-lag cumulative effect (Model 3): $\hat{\beta}_1 + \hat{\beta}_2 + \hat{\beta}_3 = -0.152 - 0.017 - 0.611 = -0.780$

### 5.2 Hausman Specification Test

**Theorem (Hausman, 1978).** Let $\hat{\boldsymbol{\beta}}_{\text{FE}}$ and $\hat{\boldsymbol{\beta}}_{\text{RE}}$ be the fixed and random effects estimators. Under $H_0$ (RE correctly specified, i.e., $\text{Cov}(\alpha_i, \mathbf{x}_{it}) = \mathbf{0}$), both are consistent; under $H_1$, only FE is consistent. The Hausman statistic:

$$
H = (\hat{\boldsymbol{\beta}}_{\text{FE}} - \hat{\boldsymbol{\beta}}_{\text{RE}})'\!\left[\widehat{\text{Var}}(\hat{\boldsymbol{\beta}}_{\text{FE}}) - \widehat{\text{Var}}(\hat{\boldsymbol{\beta}}_{\text{RE}})\right]^{-1}\!(\hat{\boldsymbol{\beta}}_{\text{FE}} - \hat{\boldsymbol{\beta}}_{\text{RE}}) \xrightarrow{d} \chi^2_k \quad \text{under } H_0
$$

**Result.** $\chi^2(2) = 1.774$, $p = 0.412$ — fail to reject $H_0$. Despite this, FE is preferred throughout because the RE assumption $\text{Cov}(\alpha_i, \mathbf{x}_{it}) = 0$ is implausible: game-level characteristics (genre, studio reputation, community size) are likely correlated with sentiment levels.

---

## 6. Clustered Standard Errors

### 6.1 Why Clustering?

Two diagnostic tests motivate clustered SEs:
- **Breusch–Pagan** (heteroscedasticity): LM $= 30.537$, $p < 0.001$
- **Ljung–Box** (serial correlation): $Q(20) = 31{,}353$, $p < 0.001$

### 6.2 Theorem (Clustered Variance Estimator)

When observations are independent across clusters $i = 1,\ldots,N$ but arbitrarily correlated within clusters, the cluster-robust sandwich estimator:

$$
\hat{V}_{\text{clust}} = (X'X)^{-1} \left(\sum_{i=1}^{N} \mathbf{X}_i' \hat{\boldsymbol{\varepsilon}}_i \hat{\boldsymbol{\varepsilon}}_i' \mathbf{X}_i\right) (X'X)^{-1}
$$

is consistent for $\text{Var}(\hat{\boldsymbol{\beta}})$ as $N \to \infty$, regardless of the within-cluster correlation structure ($\mathbf{X}_i$, $\hat{\boldsymbol{\varepsilon}}_i$ are the stacked regressors and residuals for cluster $i$).

### 6.3 Breusch–Pagan LM Test

Regress $\hat{\varepsilon}_{it}^2$ on regressors $\mathbf{x}_{it}$. Under $H_0$ (homoscedasticity):

$$
\text{LM} = n \cdot R^2_{\text{aux}} \xrightarrow{d} \chi^2_k
$$

### 6.4 Ljung–Box Portmanteau Test

For residual autocorrelations $\hat{\rho}_k$ at lags $1,\ldots,m$:

$$
Q(m) = T(T+2)\sum_{k=1}^{m} \frac{\hat{\rho}_k^2}{T-k} \xrightarrow{d} \chi^2_m \quad \text{under } H_0 \text{ (no serial correlation)}
$$

---

## 7. Block Bootstrap Confidence Intervals

### 7.1 Procedure

**Bootstrap procedure** (2,000 replications, seed = 189):
1. Draw $N = 15$ games **with replacement** from $\{1,\ldots,15\}$.
2. Stack all time series observations for the selected games into a bootstrap panel.
3. Re-estimate the entity-FE model on the bootstrap panel; record $\hat{\beta}^{*(b)}$.
4. Repeat $B = 2{,}000$ times. Report the percentile CI $[\hat{\beta}^{*}_{(0.025)},\, \hat{\beta}^{*}_{(0.975)}]$.

**Theorem (Bootstrap Percentile CI).** Under standard bootstrap consistency conditions, the percentile CI is asymptotically valid. It requires no parametric distributional assumption on $\varepsilon_{it}$.

**Lemma (Why game-level resampling?).** Resampling entire game panels preserves the within-game temporal dependence $\{\varepsilon_{it}\}_{t=1}^T$. Row-level resampling would treat observations as i.i.d. and understate uncertainty when serial correlation is present (as confirmed by the Ljung–Box test).

**Bootstrap SE:** $\widehat{\text{SE}}_{\text{boot}} = \sqrt{\frac{1}{B-1}\sum_{b=1}^{B}(\hat{\beta}^{*(b)} - \bar{\hat{\beta}}^*)^2}$

**Result.** Mean $= -0.512$, SE $= 0.268$, 95% CI $= [-1.093,\,-0.050]$. Asymptotic CI $= [-1.004,\,-0.019]$. Both exclude zero.

---

## 8. Statistical Power and Minimum Detectable Effect

### 8.1 Power Formula

For a two-sided test $H_0: \beta = 0$ at level $\alpha$:

**Theorem.** Under $H_1: \beta = \beta_a \neq 0$ with $\hat{\beta} \approx \mathcal{N}(\beta_a, \text{SE}^2)$:

$$
\text{Power}(\beta_a) = 1 - \Phi\!\left(z_{1-\alpha/2} - \frac{|\beta_a|}{\text{SE}}\right) + \Phi\!\left(-z_{1-\alpha/2} - \frac{|\beta_a|}{\text{SE}}\right)
$$

where $\Phi$ is the standard normal CDF and $z_{1-\alpha/2} = 1.960$ for $\alpha = 0.05$.

### 8.2 Minimum Detectable Effect (MDE)

**Definition.** The MDE at power level $1-\kappa$ is the smallest $|\beta^*|$ achieving that power:

$$
|\beta^*| = (z_{1-\alpha/2} + z_{1-\kappa}) \cdot \text{SE}(\hat{\beta})
$$

**Derivation.** Set Power $= 1 - \kappa$, ignoring the negligible second tail:

$$
\Phi\!\left(\frac{|\beta^*|}{\text{SE}} - z_{1-\alpha/2}\right) = 1 - \kappa \implies \frac{|\beta^*|}{\text{SE}} - z_{1-\alpha/2} = z_{1-\kappa} \implies |\beta^*| = (z_{1-\alpha/2} + z_{1-\kappa}) \cdot \text{SE}
$$

**Result.** With SE $= 0.251$, $\alpha = 0.05$, $1 - \kappa = 0.80$ ($z_{0.80} = 0.842$):

$$
\text{MDE} = (1.960 + 0.842) \times 0.251 = 0.707
$$

Power at observed $|\hat{\beta}| = 0.511$: $0.535$ (below the 0.80 threshold). The design is underpowered.

---

## 9. Residual Diagnostics

### 9.1 Shapiro–Wilk Normality Test

**Test statistic:**

$$
W = \frac{\left(\sum_{i=1}^n a_i\, x_{(i)}\right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}
$$

where $x_{(i)}$ are order statistics and $a_i = \frac{m_i'}{\sqrt{m'(V^{-1})'V^{-1}m}}$ with $m_i$ the expected values of standard normal order statistics and $V$ their covariance matrix. $W \in (0,1]$; values close to 1 indicate normality.

**Result.** $W = 0.923$, $p < 0.001$: residuals are non-normal. With $N = 3{,}630$, inference remains valid by the CLT, but motivates the non-parametric checks.

### 9.2 Variance Inflation Factor

$$
\text{VIF}_j = \frac{1}{1 - R^2_j}
$$

where $R^2_j$ is the $R^2$ from regressing $x_j$ on all other regressors. VIF $> 10$ signals severe multicollinearity.

**Result.** VIF $\approx 1.000$ for all regressors — no multicollinearity.

---

## 10. LOESS Nonparametric Regression

**Definition.** At each evaluation point $x_0$, LOESS fits a local weighted polynomial:

$$
\hat{m}(x_0) = \arg\min_{\mathbf{b}} \sum_{i=1}^{n} K\!\left(\frac{|x_i - x_0|}{h(x_0)}\right) \!\left(y_i - b_0 - b_1(x_i - x_0)\right)^2
$$

where $K$ is the tricube kernel $K(u) = (1 - |u|^3)^3\,\mathbf{1}(|u| \leq 1)$ and $h(x_0)$ is the bandwidth set so that a fraction $f$ (span) of the data falls within the window. No global functional form is assumed.

**Use.** If the LOESS curve tracks the OLS line, the linear panel specification is validated. If it shows curvature, polynomial or spline terms may be needed.

**Result.** LOESS (span $= 0.3$) closely tracks the OLS line over all 3,630 observations — linearity supported.

---

## 11. Cross-Correlation Analysis

The **cross-correlation function (CCF)** between sentiment $\{S_t\}$ and log players $\{Y_t\}$ at lag $k \geq 0$:

$$
\hat{\rho}_{SY}(k) = \frac{\sum_{t=1}^{T-k}(S_t - \bar{S})(Y_{t+k} - \bar{Y})}{\sqrt{\sum_t (S_t - \bar{S})^2\,\sum_t (Y_t - \bar{Y})^2}}
$$

$\hat{\rho}_{SY}(k) < 0$ at $k > 0$ means current sentiment negatively predicts future player counts — sentiment *leads* declines.

**Approximate 95% CI** under white-noise null: $\pm 1.96/\sqrt{T}$. With $T \approx 247$, CI $\approx \pm 0.125$.

The mean CCF (averaged across all 15 games) is reported. Bars outside the $\pm 0.125$ band are statistically significant on average.

---

## 12. Conditional Probability and $\chi^2$ Independence

Define "decline" as $\Delta\log(\text{players})_{t+1} < 0$. Split weeks at the median of $S_{it}$.

**Null hypothesis:** $H_0$: sentiment level and next-week direction are independent ($p_{ij} = p_{i\cdot}\,p_{\cdot j}$).

**Theorem (Pearson $\chi^2$).** Under $H_0$:

$$
\chi^2 = \sum_{i=1}^{r}\sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}} \xrightarrow{d} \chi^2_{(r-1)(c-1)}, \quad E_{ij} = \frac{(\text{row}_i\text{ total})(\text{col}_j\text{ total})}{n}
$$

**Result.** $P(\text{decline} \mid \text{high sentiment}) = 0.550$ vs $P(\text{decline} \mid \text{low sentiment}) = 0.467$, difference $= 8.3$ pp. $\chi^2(1) = 25.19$, $p < 0.001$.

---

## 13. Placebo / Reverse Lag Test

To check for reverse causation, we reverse the lag direction: test whether *future* sentiment predicts *past* player changes. Under a genuine temporal ordering, the reversed test should yield few significant results.

**Result.** Original direction: 7/15 games significant at $\alpha = 0.05$. Reversed: 3/15 games. The asymmetry (7 vs. 3) partially rules out reverse causation, though 3 significant reversed results suggest some bidirectional relationship may exist.

---

## Summary of Methods and Results

| Method | Mathematical Core | Key Result |
|--------|------------------|------------|
| Panel regression (FE) | Within estimator; clustered sandwich SE | $\hat{\beta} = -0.649$, $p = 0.040$, CI $= [-1.268,\,-0.030]$ |
| Multi-lag model | $\sum_{k=1}^3 \beta_k S_{i,t-k}$ | Cumulative $= -0.780$ over 3 weeks |
| Hausman test | $\chi^2$ specification test (FE vs RE) | $\chi^2(2) = 1.774$, $p = 0.412$; FE preferred |
| Granger causality | Nested $F$-test on VAR | 7/15 games at $\alpha = 0.05$ |
| Permutation test | Exact permutation $p$-value | 4/15 games survive |
| Block bootstrap | Percentile CI, $B = 2{,}000$ | CI $= [-1.093,\,-0.050]$ |
| Power analysis | Normal-approximation formula | Power $= 0.535$, MDE $= 0.707$ (underpowered) |
| ADF stationarity | Dickey–Fuller distribution | 25/30 series have unit roots; first-differencing used |
| LOESS | Local weighted polynomial | Linear specification validated |
| Breusch–Pagan | LM test, $\sim \chi^2_k$ | LM $= 30.537$, $p < 0.001$ |
| Ljung–Box | Portmanteau test, $\sim \chi^2_m$ | $Q(20) = 31{,}353$, $p < 0.001$ |
| Shapiro–Wilk | Order-statistic test | $W = 0.923$, $p < 0.001$ |
| Conditional probability | Pearson $\chi^2$ independence | $\chi^2(1) = 25.19$, $p < 0.001$ |
| Cross-correlation | CCF at lags $\pm 12$ weeks | Negative at positive lags; mean CI $\approx \pm 0.125$ |
| Placebo test | Reversed-lag Granger | 3/15 vs. 7/15; temporal ordering supported |

For full data pipeline, EDA, and results see **`final_draft.ipynb`**. For the written report see **`report/report.pdf`**.
