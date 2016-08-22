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
