import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split

def check_interaction(df_high,df_low,iv,dv,grouped_var):
    """
    Plots a scatter plot of the dependent variable [dv] as a function of the independent variable [iv]
    when iv is split into two groups, as denoted by df_high and df_low. These two dataframes must contain mutually
    exclusive data.

    Param df_high: [pandas DataFrame] First split (high values)
    Param df_low: [pandas DataFrame] First split (low values)
    Param iv: [str] the name of the independent variable exactly as it appears in the dataframes.
    Param dv: [str] the name of the dependent variable exactly as it appears in the dataframes.
    Param grouped_var: [str] the name of the variable that was used to split the iv into two dataframes. Must
                       be written exactly as it appears in the dataframe.
    """

    regression_1 = LinearRegression()
    regression_2 = LinearRegression()

    high_data = df_high[iv].values.reshape(-1, 1)
    low_data = df_low[iv].values.reshape(-1, 1)

    regression_1.fit(high_data, df_high[dv])
    regression_2.fit(low_data, df_low[dv])

    # Make predictions using the testing set
    pred_1 = regression_1.predict(high_data)
    pred_2 = regression_2.predict(low_data)

    # The coefficients
    print('regression coeff 1 is ', regression_1.coef_)
    print('regression coeff 2 is ', regression_2.coef_)

    # Plot outputs
    plt.figure(figsize=(10,6));

    plt.scatter(high_data, df_high[dv],  color='blue', alpha = 0.3, label = f'high {grouped_var}');
    plt.scatter(low_data, df_low[dv],  color='red', alpha = 0.3, label = f'low {grouped_var}');

    plt.plot(high_data, pred_1,  color='blue', linewidth=2);
    plt.plot(low_data, pred_2,  color='red', linewidth=2);

    plt.ylabel(dv)
    plt.xlabel(iv)
    plt.title(f'Interaction between {iv} and {grouped_var}')
    plt.legend();

def compare_fits(X,y,test_size=0.2, alpha=0.5):
    """
    Print out fit comparisons between linear, ridge, and lasso regressions.

    Param X: predictors
    Param y: outcome
    Param test_size: percentage of data set to be used as the test sample
    Param alpha: hyperparameter for ridge and lasso. Must be [0,1].
    """
    # Perform test train split
    X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

    # Build a Ridge, Lasso and regular linear regression model.
    # Note how in scikit learn, the regularization parameter is denoted by alpha (and not lambda)
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)

    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)

    lin = LinearRegression()
    lin.fit(X_train, y_train)

    # create predictions
    y_h_ridge_train = ridge.predict(X_train)
    y_h_ridge_test = ridge.predict(X_test)

    y_h_lasso_train = np.reshape(lasso.predict(X_train),(1279,1))
    y_h_lasso_test = np.reshape(lasso.predict(X_test),(320,1))

    y_h_lin_train = lin.predict(X_train)
    y_h_lin_test = lin.predict(X_test)

    # examine the residual sum of sq
    print('---------------------------------------')
    print('Train Error Ridge Model', np.sum((y_train - y_h_ridge_train)**2))
    print('Test Error Ridge Model', np.sum((y_test - y_h_ridge_test)**2))
    print('\n')

    print('Train Error Lasso Model', np.sum((y_train - y_h_lasso_train)**2))
    print('Test Error Lasso Model', np.sum((y_test - y_h_lasso_test)**2))
    print('\n')

    print('Train Error Unpenalized Linear Model', np.sum((y_train - lin.predict(X_train))**2))
    print('Test Error Unpenalized Linear Model', np.sum((y_test - lin.predict(X_test))**2))
