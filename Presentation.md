---
marp: true
theme: default
---
*Andrea Donazzan, Amadeus Lars Linge*
# Empirical Comparison of Option Pricing Models
---
## Content
- Data collection
- Monte Carlo pricing models
- Calibration
- Pricing with Machine Learning
- Results
---

## Data Collection

**Goal:** build database with market price of options and information about underlying

Stock data: ```yfinance ```\
Option data: ```yahooquery```

```python
from yahooquery import Ticker
Ticker('AAPL').option_chain
```
![option](./Presentation%20files/Option_Chain.png)


---
## Modifying Option Chain
Add price of underlying, historical variablility, option maturity


```python
import Data
Data.GetData('2018,11,22', '2022,11,22', 252, False)
```

![data](./Presentation%20files/Data_df.png)

---

# Pricing Models
European Options: 
- **Black Scholes** Model (GMB) 
- Merton's **Jump Diffusion** (GMB with Poisson)

American Options: 
- **Cox-Ross-Rubinstein** Model (Binomial Tree)
- **Longstaff-Schwartz** Model (Least Squares Monte Carlo)
---
## Black Scholes
$$
S(t_{i+1})=S(t_i)\exp\left(\left(\mu - \frac{\sigma}{2}\right)\Delta t + \sigma \sqrt{\Delta t} Z_{i+1}\right)
$$
## Jump Diffusion
$$
S(t_{i+1})=S(t_i)\exp\left(\left(\mu - \frac{\sigma}{2}\right)\Delta t + \sigma \sqrt{\Delta t} Z_{i+1} + aN_{i+1}+b\sqrt{N_{i+1}}Z_{i+1}' \right)
$$
 

$$
\text{with } N_i \sim \text{Pois}(\lambda \Delta t)
$$
---

## American Options
Allow for **early exercise**

Value at each time step: maximum between early exercise value and continuation value.

For an **American call** option:
$$
O_t^i = \max\left((S_t^i - K)^+,\: e^{-(T-t)r}(E[S_T|S_t^i] - K)^+ \right)
$$

---

## Binomial Trees 1: Simulate Stock Value
Stock price can:
- Move up to $uS_{0}$ with probability $q$
- Move down to $uS_{0}$ with probability $(1-q)$
$$
u = e^{\sigma \sqrt{\Delta t}}
\qquad
d = e^{-\sigma \sqrt{\Delta t}} = \frac{1}{u}
\qquad
q = \frac{e^{r\Delta t}-d}{u-d}\\
$$
Fill the best outcome nodes  with $S_{i,0} = u S_{i-1,0}$ and the others with $S_{i,j}=d S_{i-1,j-1}$

<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>

![center width:650](./Presentation%20files/Stock_tree.png)

---

## Binomial Trees 2: Compute Option Price
Start from end of tree and move upwards:

At maturity, compute value as $\max(S_{T,j}-K, 0) $

For each previous node compute 
$$
\quad O_{i,j} = \max \left(\underbrace{e^{-r\Delta t}(qO_{(i+1),j}+(1-q)O_{(i+1),(j+1)})}_\text{continuation value},\; \underbrace{S_{i,j}-K}_\text{immediate exercise}\right)
$$

![center width:370px](./Presentation%20files/Option_tree.png)

---

## Longstaff-Schwartz algorithm
- Proposed in **Longstaff and Schwartz (2001)**.
- Computes price of options that can be exercised before maturity.
- Uses least-squares regression to estimate conditional expectations.

---

## Motivation: Why do we need the LSMC algorithm?

- How can we evaluate the **continuation value** $E[S_T|S_t^i]$?

    Nested Monte Carlo simulations for every $S_t^i$ until maturity &rarr; **unfeasible** for large numbers
    
- Binomial Tree: high discretization error if used with long time steps.

    Will **underestimate** the number of **early exercise opportunities** as it only provides two outcomes for the value of the underlying.

    On the other hand, time complexity: $O(2^n)$

---

# Calibration

--

## Jump Diffusion Calibration
Parameters:
- $\lambda$ &rarr; rate of Poisson process 
- $a$ and $b$ &rarr; size of the jump

Test different values for each parameter holding the other two constant 
Compute mean squared error of estimated prices
<table style="width:100%"><tr>
<td> <img src="./Presentation files/MJD_lambda_error.png"/> </td>
<td> <img src="./Presentation files/MJD_a_error.png"/> </td>
<td> <img src="./Presentation files/MJD_b_error.png" /> </td>
</tr></table>

$\lambda = 0.075, a = -0.1, b = 0.05$
---


<table style="width:100%">
    <tr>
        <img src="./Presentation files/MJD_lambda_error.png"  width="200">
        <img src="./Presentation files/MJD_a_error.png"  width="200">
        <img src="./Presentation files/MJD_b_error.png"  width="200">
    </tr>
</table>
---

## Interest rate and Standard Deviation calibration

