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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import regressor_base
from sklearn.decomposition import PCA

'''Note: this is the implementation of cocoreg algorithm in Python and it was originally in R,
the performance and behavior are not confirmed yet to be the same between both implementations. So please
refer to the original implementation in R [1] if you have any doubts. Also please refer to the original paper [2] for how it works.

[1] https://github.com/bwrc/cocoreg-r
[2] Using regression makes extraction of shared variation in multiple datasets easy,
Jussi Korpela, Andreas Henelius, Lauri Ahonen, Arto Klami, Kai Puolamäki, Data Mining and Knowledge Discovery,
2016. (URL: http://dx.doi.org/10.1007/s10618-016-0465-y)
'''

class cocoreg_base():
    def __init__(self, X):
        # first check if there are more than one signals input
        assert(len(X) > 1)
        for i in range(len(X)-1):
            assert(len(X[i]) == len(X[i+1]))
        self.X = X
        self.subtract_mean()
        self.n_regressors = len(X) - 1
        self.path_list = []
        self.outputs = []

    def subtract_mean(self):
        for i in range(0, len(self.X)):
            self.X[i] -= np.mean(self.X[i])

    def pick_path(self, cur_node_list, path):
        if len(cur_node_list) == 0:
            self.path_list.append(path)
            return
        for node in cur_node_list:
            cur_node_list_next = list(cur_node_list)
            cur_node_list_next.remove(node)
            path_next = list(path)
            path_next.append(node)
            self.pick_path(cur_node_list_next, path_next)

    def chain_of_regressors(self):
        # generate path for each node first
        full_node_list = tuple(np.arange(0, len(self.X))) # immutable list
        for node in full_node_list:
            self.path_list = []
            node_list = list(full_node_list)
            path = [node]
            node_list.remove(node)
            self.pick_path(node_list, path)

            # now the path list starting with 'node' is contained in self.path_list
            mean_output = 0.0
            n = 0.0
            for path in self.path_list:
                input = self.X[path[0], :]
                # if it is 1-D vector, np.shape(input) will be sth. like (D, ), reshape it to (D, 1)
                if len(np.shape(input)) == 1:
                    input = input.reshape(np.shape(input)[0], 1)
                for i in range(self.n_regressors):
                    output = self.X[path[i+1], :]
                    output = output.reshape(np.shape(output)[0], 1)
                    regressor = regressor_base.regressor_base(input, output)
                    input = regressor.lin_regression()
                    if len(np.shape(input)) == 1:
                        input = input.reshape(np.shape(input)[0], 1)
                n += 1.0
                mean_output = (n - 1.0) / n * mean_output + 1.0 / n * input

            self.outputs.append(mean_output[:, 0])
        self.outputs = np.array(self.outputs)
    def do_PCA(self):
        pca = PCA(n_components=1)
        return pca.fit_transform(self.outputs.T)
