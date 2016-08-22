import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

class regressor_base():
    def __init__(self, X, Y, model_choice='ridge', degree=2, alpha=0.01):
        assert(X != None and Y != None)
        self.X = X
        self.Y = Y
        self.model_choice = model_choice
        self.degree = degree
        self.alpha = alpha

    # linear regression
    def lin_regression(self):
        regressor = None
        if self.degree > 1:
            poly = PolynomialFeatures(degree=self.degree)
            self.X = poly.fit_transform(self.X)
        if self.model_choice == 'ols':
            regressor = linear_model.LinearRegression()
        elif self.model_choice == 'ridge':
            regressor = linear_model.Ridge(alpha = self.alpha)
        elif self.model_choice == 'lasso':
            regressor = linear_model.Lasso(alpha = self.alpha)
        else:
            print 'Warning: no model specified'

        regressor.fit(self.X, self.Y)

        return regressor.predict(self.X)
