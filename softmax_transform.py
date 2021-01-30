import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    num_of_sample = 10000
    v = np.random.rand(num_of_sample)
    v_Gauss = np.random.normal(loc=0, scale=1, size=2000)
    v_tensor = torch.tensor(v).cuda()
    v_tensor_softmaxed = F.softmax(v_tensor, dim=0)
    v_Gauss_tensor_softmaxed = F.softmax(torch.tensor(v_Gauss), dim=0)
    v_tensor_exponent = torch.exp(v_tensor)
    v_ndarray_softmaxed = v_tensor_softmaxed.cpu().numpy()
    v_ndarray_exponent = v_tensor_exponent.cpu().numpy()
    bins = 100
    # Softmax change the uniform distribution to logarithmic distribution, not Boltzmann distribution.
    plt.subplot(411)
    plt.hist(v, bins)
    plt.subplot(412)
    plt.hist(v_ndarray_softmaxed, bins)
    plt.subplot(413)
    plt.hist(v_ndarray_exponent, bins)
    plt.subplot(414)
    # logarithmic normal distribution, similar to Boltzmann distribution.
    plt.hist(v_Gauss_tensor_softmaxed.numpy(), bins)
    plt.show()
