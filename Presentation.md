---
marp: true
theme: default
math: katex
---

# **Empirical Comparison of Option Pricing Models**

Andrea Donazzan, Amadeus Lars Linge
*IEDA 4520*


---
## Content
- Monte Carlo Pricing Models
- Pricing with Machine Learning
- Data Collection
- Calibration
- Results

--- 

# **Pricing Models**
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
O_t^i =\max\left(\underbrace{(S_t^i - K)^+}_\text{early exercise},\: \underbrace{e^{-(T-t)r}(E[S_T|S_t^i] - K)^+}_\text{continuation value} \right)
$$

---

## Binomial Trees 1: Simulate Stock Value
Stock price can:
- Move up to $uS_{0}$ with probability $q$
- Move down to $dS_{0}$ with probability $(1-q)$
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

![center width:400](./Presentation%20files/Stock_tree.png)

---

## Binomial Trees 2: Compute Option Price
Start from end of tree and move upwards:

At maturity, compute value as $\max(S_{T,j}-K, 0)$

For each previous node compute 
$$
\quad O_{i,j} = \max \left(e^{-r\Delta t}(qO_{(i+1),j}+(1-q)O_{(i+1),(j+1)}),\;S_{i,j}-K\right)
$$

![center width:400px](./Presentation%20files/Option_tree.png)

---

## Why do we need the Longstaff-Schwartz algorithm?

- How can we evaluate the **continuation value** $E[S_T|S_t^i]$?

    Nested Monte Carlo simulations &rarr; **unfeasible** for large numbers
    
- *Binomial Tree:* **discretization** error if used with long time steps.

    Will **underestimate** the number of **early exercise opportunities** as it only provides two outcomes for the value of the underlying.

    *Time complexity*: $O(2^n)$

---

## LSMC explanation: an example

We take as example an **American call** option with 1 year maturity, exercisable at times 1,2,3:

$$
S_0 = 1,\: K = 1.1,\: r = 0.1,\: \sigma = 0.2
$$

![bg right:40% 80%](./Presentation%20files/Stock_path.png) 

---

## Step 1
Setup **cash flow matrix**:
Determine expected payoff at **maturity**.
Since continuation value is zero = payoff of a vanilla European option
![bg right:40% 70%](./Presentation%20files/Payoff_maturity.png) 

---

## Step 2

- One time step back: consider the paths were the option is **in the money** at $t = T-1$.
- Discount the future **cash flow** of holding the option: $y_{t=2,i} = e^{-r}\pi_{t=3,i}$
- Get value of underlying at time T-1
![bg right:30% 70%](./Presentation%20files/Disc_payoff.png) 

---

## Step 3

- Regress $y_{t=2}$ on a set of basis functions of $S_{t=2}$ to obtain the **continuation value** 
$$
\hat{C}_{t,i} = \sum_{j=0}^{n} a_{t,j}B_j(S_{t,i})
$$

The parameters $a_t$ are obtained minimizing
$$
\frac{1}{I}\sum_{i=0}^I\left(y_{t,i}-\hat{C}_{t,i}\right)^2
$$
```python
X = np.column_stack([np.ones(M), S[:,i], S[:,i]**2, S[:,i]**3,S[:,i]**4,S[:,i]**5])
beta = np.linalg.lstsq(cond_x, Y, rcond=None)[0]
continue_val = np.dot(X, beta)
```
---

## Step 4
If $S_{t,i} > \hat{C}_{t,i}$ , fill cash flow matrix with resulting cash flow from this path.

Repeat until $t=0$.

![bg right:30% 70%](./Presentation%20files/Continue_payoff.png)

---

# Pricing with Machine Learning
---
# Splitting the data
4 sets of data in total for training:
- European Options: calls/puts
- American Options: calls/puts
```python
from sklearn.model_selection import train_test_split

X_train_calls, X_test_calls, y_train_calls, y_test_calls = train_test_split(X_calls, y_calls, test_size=0.25)
X_train_puts, X_test_puts, y_train_puts, y_test_puts = train_test_split(X_puts, y_puts, test_size=0.25)

Train calls: (27187, 5)
Test calls: (9063, 5)
Train puts: (18194, 5)
Test puts: (6065, 5)
```
---
# Models
     DTR - Decision Tree Regressor
     XGBr - Xtreme Gradient Booster

Supervised Learning Models
```python
(S0, K, T, sigma, r) --> option price
```
     
---
# DecisionTreeRegressor
```python
DTR_calls = DecisionTreeRegressor(max_depth=24, min_samples_leaf=1)
DTR_calls.fit(X_train_calls, y_train_calls)
DTR1_pred = DTR_calls.predict(X_test_calls)
```
![center width:700](./Presentation%20files/regressiontree.png)

---
# XGBr
<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>

- Advanced verison of GBM
- Ensemble of decision trees

![bg right 30% 70%](./Presentation%20files/grad_decent.png)


- Handles missing data
- Uses L1 and L2 regularization (Lasso- and Ridge-regularization)
- Tree pruning 
- Parallelized 

```python
XGBr = xg.XGBRegressor(learning_rate=0.1, gamma= 0.001, 
max_depth= 5, min_child_weight= 6, 
subsample= 1, n_estimators=900)

XGBr.fit(X_train_calls, y_train_calls)
XGBr_pred = XGBr.predict(X_test_calls)
```

---

## Pipeline
```python
def model(pipeline, parameters, X_train, y_train, X, y, figname):

    grid_obj = GridSearchCV(estimator = pipeline, param_grid = parameters, cv = 5, 
                            scoring = 'r2', verbose = 0, n_jobs = 1, refit = True)

    grid_obj.fit(X_train, y_train)

    print("Best Param:", grid_obj.best_params_)
    estimator = grid_obj.best_estimator_
    shuffle = KFold(n_splits = 5, shuffle = True, random_state = 0)
    cv_scores = cross_val_score(estimator, X, y.ravel(), cv=shuffle, scoring='r2')

    y_pred = cross_val_predict(estimator, X, y, cv = shuffle)
    
```
---
## XGBr optimal parameters
![center width:600](./Presentation%20files/CV_XGBr_calls_plot.png)
 ```python
 Best Param: {'xgb__colsample_bytree': 1, 'xgb__gamma': 0.01, 'xgb__max_depth': 5, 'xgb__min_child_weight': 3, 'xgb__subsample': 0.6}
 ```
---
## DecisionTreeRegressor optimal parameters
![center width:600](./Presentation%20files/CV_DTR_calls_plot.png)
```python
Best Param: {'dt__max_depth': 16, 'dt__min_samples_leaf': 1}
```

---

## **Data Collection**

**Goal:** build database with market price of options and information about underlying


Stock data: ```yfinance ``` $\qquad$ Option data: ```yahooquery```
```python
def BS(S0, K, T, sigma, r, type):
```
**Option Chain:**
![option](./Presentation%20files/Option_Chain.png)


---
## Modifying Option Chain
Add price of underlying, maturity, variablility, mean returns.

```python
import Data
Data.GetData('2018,11,22', '2022,11,22', 252, False)
```

![data](./Presentation%20files/Data_df.png)

---

## **Jump Diffusion Calibration**
Parameters:
- $\lambda$ &rarr; rate of Poisson process 
- $a$ and $b$ &rarr; size of the jump

Test different values for each parameter holding the other two constant 
Compute mean squared error of estimated prices
```python
df = df.sample(n)
lamb_values = np.linspace(0,0.4, iterations)
for i in range(iterations):
  lamb = lamb_values[i]
  errors.append(compute_errors(lamb, a, b))

```

---
![center](./Presentation%20files/MJD_calibration.png)
$\lambda = 0.15, a = -0.1, b = 0.05$

---

## Interest rate and Standard Deviation calibration

---
## Error given interest rate

![center width:900](./Presentation%20files/err_interest.png)

---
## Error given volatility

![center width:900](./Presentation%20files/err_volatility.png)

---

# Results
---

## Algorithm runtime
![center](./Presentation%20files/Execution_time.png)

---

## Option prices: Predicted against Market

![center](./Presentation%20files/Model_predictions.png)

---

## Distribution of errors
![center](./Presentation%20files/Model_errors.png)

---
<style>
img[alt~="top-right"] {
  position: absolute;
  top: 100px;
  right: 30px;
}
</style>

<style>
img[alt~="bottom-right"] {
  position: absolute;
  top: 130px;
  left: 60px;
}
</style>

![top-right](./Presentation%20files/Violinplot_errors.png)

![bottom-right width:500](./Presentation%20files/Summary_errors.png)


