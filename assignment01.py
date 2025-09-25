import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, RidgeCV

import matplotlib.pyplot as plt
import matplotlib.dates as mdates 

#Load CSV
returns = pd.read_csv('training_data.csv')

#Convert date to yyyy-mm-dd (timestamp) format
returns['Date'] = pd.to_datetime(returns['Date'],format='%Y%m%d')

#set date frame index
returns.set_index('Date', inplace=True)

#added: express percentage in decimals
returns = returns / 100.0

#Modularize the code
def prepare_regression_matrices(R):
   #p = number of assets
   p = R.shape[1]
   print("Line 17:", p)
   #w_ew = weightage for each of the assets in p
   w_ew = np.full((p, 1), 1.0/p)
   print("Line 20:", w_ew)
   #y = vector y, which is a matrix multiplication of R and w_ew, 
   # each element is the return at a given point of time for each data point in the portfolio
   y = R @ w_ew
   print("Line 24:", y)
   print("Line 24:",y.ravel())
   N = np.vstack([np.eye(p - 1), -np.ones((1, p - 1))])
   print("Line 27:", N)
   X = R @ N
   print("Line 29:", X)
   return y.ravel(), X

def regression_portfolio_weights(beta, N, w_ew):
    return w_ew - N @ beta.reshape(-1, 1)

def get_minvar_weights(train_data):
   #Minimum-variance portfolio block, 
   # calculating prtfolio weights using Lasso/Ridge done at line 85
   p = train_data.shape[1]
   sigma_hat = np.cov(train_data.T)
   try:
       w = np.linalg.solve(sigma_hat, np.ones(p))
       w /= w.sum()
       return w
   except:
       return np.full(p, 1.0/p)

#126 training days in a year, rolling window set for 6 months
WINDOW = 126
alphas = np.logspace(-8, 8, 21)
dates = returns.index
print(dates)
#Below start and end dates will be used for test data, 
#with the auumption that train data for 126 days prior is avilable in the training_data.csv file.
start = returns.index.get_loc('2025-02-03')
print("Line 53:", start)
end = returns.index.get_loc('2025-02-10')
print("Line 56:", end)

lasso_ret, ridge_ret, ew_ret, minvar_ret, eval_dates = [], [], [], [], []

for i in range(start, end + 1):
   print(i, start, end)
   t0 = i - WINDOW
   print("Line 61:", i)
   print("Line 61:", t0)
   if t0 < 0: 
      #Not enough past data to train, skip the iteration
      continue
   
   #Full window (126) available, train on returns[t0:1]
   #training data set, picks all rows which are within the range
   train = returns.iloc[t0:i].values
   #test data set,  picks one particular row
   test = returns.iloc[i].values
   #Fit model using train data and test using the test data, for train data it will only return NaN.

   print("Line 74:", train)
   print("Line 75:", test)

   #Call regression matrix with training data
   y, X = prepare_regression_matrices(train)

   # LASSO (fit_intercept handles mean automatically)
   lasso = LassoCV(alphas=alphas, cv=5, max_iter=15000).fit(X, y)
   ridge = RidgeCV(alphas=alphas, cv=5).fit(X, y)

   p = train.shape[1]
   print("Line 79:",p)
   w_ew = np.full((p, 1), 1.0/p)
   N = np.vstack([np.eye(p - 1), -np.ones((1, p - 1))])

   w_lasso = regression_portfolio_weights(lasso.coef_, N, w_ew)
   w_ridge = regression_portfolio_weights(ridge.coef_, N, w_ew)
   w_minvar = get_minvar_weights(train)

   #From here the TEST data is picked to evaluate against the model
   lasso_ret.append(np.dot(test, w_lasso.ravel()))
   ridge_ret.append(np.dot(test, w_ridge.ravel()))
   ew_ret.append(np.dot(test, w_ew.ravel()))
   minvar_ret.append(np.dot(test, w_minvar))
   eval_dates.append(dates[i])

results = pd.DataFrame({
   'LASSO': lasso_ret,
   'Ridge': ridge_ret,
   'EqualWeight': ew_ret,
   'MinVar': minvar_ret
}, index=pd.to_datetime(eval_dates))

print("Line 101:", results)

print(results.mean() / results.std())  # Sharpe ratios

#-------- added--------- 
#Plotting the Cumulative Returns
# Calculate the growth of $1 by taking the cumulative product of (1 + daily returns)
cumulative_returns = (1 + results).cumprod()

#--------added--------------------
#returns the last cumulative returns
last_date = cumulative_returns.index[-1]
last_values = cumulative_returns.iloc[-1]
print(f"\nCumulative returns on {last_date.date()}:")
print(last_values)
#----------------------------------


fig, ax = plt.subplots(figsize=(14, 8))

for model in cumulative_returns.columns:
    ax.plot(cumulative_returns.index, cumulative_returns[model], label=model, lw=2)

ax.set_title('Cumulative Return (Out-of-Sample)', fontsize=18, pad=20)
ax.set_xlabel('Date', fontsize=12)
# ax.set_ylabel('Growth of $1', fontsize=12)
ax.axhline(1.0, color='grey', linestyle='--', linewidth=1.0) # Break-even line
ax.legend(title='Portfolio Strategy', fontsize=11)
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Date formatting functions that require mdates
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
fig.autofmt_xdate() # Rotate date labels for better readability

plt.tight_layout()
plt.show()
print("Plot displayed.")
