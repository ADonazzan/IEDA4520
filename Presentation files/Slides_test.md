---
marp: true
theme: default
---

# **Implementation of Least Squares Monte Carlo**
## Using Longstaff-Schwartz algorithm

---
## Longstaff-Schwartz algorithm
- Proposed in **Longstaff and Schwartz (2001)**.
- Computes price of options that can be exercised **before maturity**.
- Uses **least-squares regression** to estimate conditional expectations.
- Useful to price American-style options.
---
## American Options
Allow for **early exercise**: 

The value at each time step is the maximum between the payoff of exercising the option at that time step (early exercise value) and the expected value of holding the option to maturity (continuation value).

For an **American call** option:
$$
V_t^i = \max\left((S_t^i - K)^+,\: e^{-(T-t)r}(E[S_T|S_t^i] - K)^+ \right)
$$
---
## Motivation: Why do we need the LSMC algorithm?

- How can we evaluate the **continuation value** $E[S_T|S_t^i]$?

    Nested Monte Carlo simulations for every $S_t^i$ until maturity &rarr; **unfeasible** for large numbers
    
- Binomial Tree: high discretization error if used with long time steps.

    Will **underestimate** the number of **early exercise opportunities** as it only provides two outcomes for the value of the underlying.

    On the other hand, time complexity: $O(2^n)$
---
## LSMC algorithm
We take as example an **American call** option with 1 year maturity, exercisable at times 1,2 and 3. Furthermore: 

$$
S_0 = 1,\: K = 1.1,\: r = 0.1,\: \sigma = 0.2
$$
```python
GBM_paths = pm.BS_path(1, 0.1, 0.2, 1, nSteps = 3, nPaths = 8)
```
---
---
