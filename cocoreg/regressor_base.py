'''
  Copyright (c) 2016 University of Helsinki

  Permission is hereby granted, free of charge, to any person
  obtaining a copy of this software and associated documentation files
  (the "Software"), to deal in the Software without restriction,
  including without limitation the rights to use, copy, modify, merge,
  publish, distribute, sublicense, and/or sell copies of the Software,
  and to permit persons to whom the Software is furnished to do so,
  subject to the following conditions:

  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
  ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
'''

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
