# References

Annotated reading list for this project. PDFs are not redistributed here (copyright);
download them into `references/papers/` (git-ignored) using the links below.

| Paper | Role in this project | Read |
|---|---|---|
| Gatev, Goetzmann & Rouwenhorst (2006) | Benchmark / baseline pair-selection method | Methodology section only |
| Engle & Granger (1987) | Pair selection used in the main strategy | Concept only (textbook/blog is fine) |
| Avellaneda & Lee (2010) | Planned extension (PCA / factor residuals) | After the baseline works |

---

## 1. Gatev, Goetzmann & Rouwenhorst (2006)
**"Pairs Trading: Performance of a Relative-Value Arbitrage Rule."**
*Review of Financial Studies* 19(3), 797-827. Working-paper version: NBER WP 7032 (1999).
Links: https://www.nber.org/papers/w7032 · SSRN abstract 141615

**Method (the "distance method")**
1. *Formation period* (12 months): turn each stock's price into a cumulative total-return index starting at 1.
2. For each stock, pair it with the stock that minimises the sum of squared deviations (SSD) between the two normalised series.
3. *Trading period* (next 6 months): open a position when the normalised spread moves more than 2 formation-period standard deviations; close when the prices cross again (or at the end of the period).

**Why it matters here**
- It is the canonical academic baseline. Re-implementing it on the same universe gives a
  reference point against which the Engle-Granger strategy can be compared.
- Its formation/trading split is the template for this project's in-sample / out-of-sample design.
- Note what it does *not* do: it has no statistical test for mean reversion. SSD finds pairs that
  *co-moved*, not pairs whose spread is *stationary*. That gap is the motivation for cointegration.

**Questions to be able to answer in an interview**
- Why can SSD pairs fail to mean-revert even when the historical fit is tight?
- How does the paper handle trading costs and the bid-ask bounce, and how do the results change?

---

## 2. Engle & Granger (1987)
**"Co-integration and Error Correction: Representation, Estimation, and Testing."**
*Econometrica* 55(2), 251-276. (The original paper is not required.)

**What to understand**
- *Correlation* concerns co-movement of returns; *cointegration* concerns a stationary linear
  combination of non-stationary price levels. Two series can be highly correlated and not
  cointegrated, and vice versa.
- Two-step test: (i) regress `y_t = a + b x_t + e_t` by OLS; (ii) run an ADF test on the residuals `e_t`.
  Null hypothesis: **no** cointegration (unit root in the residual).
- **Critical values are not the standard Dickey-Fuller ones**, because the residuals come from an
  estimated regression. Use MacKinnon's response-surface values, which is what
  `statsmodels.tsa.stattools.coint` does. Running `adfuller` on the residuals and reading the usual p-value is a common mistake.
- The test is **asymmetric**: `coint(y, x)` and `coint(x, y)` can give different statistics. A convention (e.g. test both, keep the smaller statistic, or fix an ordering) must be chosen and documented.
- Testing ~1,225 pairs means ~5% false positives under the null (multiple testing); see the half-life filter discussion in the project design notes.

---

## 3. Avellaneda & Lee (2010)
**"Statistical Arbitrage in the U.S. Equities Market."**
*Quantitative Finance* 10(7), 761-782.
Link: https://www.math.nyu.edu/faculty/avellane/AvellanedaLeeStatArb071108.pdf (author's NYU page)

**Idea (planned extension, not part of the baseline)**
Instead of pairing two stocks, regress each stock's returns on common factors (PCA eigenportfolios
or sector ETFs) and trade the mean-reverting *residual*, modelled as an Ornstein-Uhlenbeck
process, using an s-score (standardised distance from equilibrium) for entry and exit.

**Fits this project's strengths**: the OU parameter estimation, the mean-reversion speed `kappa`
and the s-score are probability/numerical-methods topics, and a Kalman or MCMC treatment of the
OU parameters would be a natural extension.

---

## Further reading (optional)
- Vidyamurthy, *Pairs Trading: Quantitative Methods and Analysis* (2004), a practitioner treatment of cointegration-based pairs.
- Krauss (2017), "Statistical Arbitrage Pairs Trading Strategies: Review and Outlook", *Journal of Economic Surveys*, a survey comparing the distance, cointegration and stochastic-spread approaches.
- Do & Faff (2010), "Does Simple Pairs Trading Still Work?", on profit decay in the GGR strategy after costs.
