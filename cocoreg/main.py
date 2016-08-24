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
import cocoreg_base
import matplotlib.pyplot as plt

'''In main() we demonstrate how to calculate shared signals of some artificial data with cocoreg algorithm'''

'''generate 3 sets of artificial time-series data of [n_samples] samples with sine waves'''
def gen_artif_data(n_samples):
    t = np.arange(n_samples)
    X = np.zeros([3, n_samples])

    sin = 0.6 * np.sin(2 * np.pi * 0.01 * t)
    sin2 = 0.6 * np.sin(2 * np.pi * 0.02 * (t - 5))
    sin3 = 0.75 * np.sin(2 * np.pi * 0.05 * (t - 10))

    X[0, :] = sin
    X[1, :] = sin2
    X[2, :] = sin3

    return t, X

def main():

    t, X = gen_artif_data(n_samples = 250)
    cocoreg = cocoreg_base.cocoreg_base(X)
    output = cocoreg.chain_of_regressors()
    pca_output = cocoreg.do_PCA()

    for label, (x, x_cocoreg) in enumerate(zip(cocoreg.X, cocoreg.outputs), 1):
        plt.figure(1)
        plt.plot(t, x, label=label)
        plt.title('Original Signals')
        plt.xlabel('time index')
        plt.legend()
        plt.figure(2)
        plt.title('Cocoreg outputs')
        plt.plot(t, x_cocoreg, label='cocoreg_output_%d'%label)
        plt.xlabel('time index')
        plt.legend()

    plt.figure(1)
    ymin, ymax = plt.ylim()
    plt.figure(2)
    plt.ylim([ymin, ymax])
    plt.plot(t, pca_output, label='pca_cocoreg')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
