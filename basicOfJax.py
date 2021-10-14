from typing_extensions import ParamSpec
from jax._src.numpy.lax_numpy import ravel_multi_index
from jax._src.random import normal
from jax.experimental.maps import hide_mapped_axes, hide_units
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
from jax import random
from numpy import arange


key = random.PRNGKey(1)


def ReLU(x):
    return np.maximum(0, x)


jitReLU = jit(ReLU)


def FinitDiffGrad(x):
    """
    docstring
    """
    return np.array((ReLU(x + 1e-3) - ReLU(x - 1e-3)) / (2 * 1e-3))


print("Jax Grad:", jit(grad(jit(ReLU)))(2.0))

batchDim = 32
featureDim = 100
hiddenDim = 512


X = random.normal(key, (batchDim, featureDim))

params = [
    random.normal(key, (hiddenDim, featureDim)),
    random.normal(key, (hiddenDim,))
]


def relu_layer(params, x):
    """
    docstring
    """
    return ReLU(np.dot(params[0], x) + params[1])


def batchVersionReLULayer(params, x):
    """
    docstring
    """
    return ReLU(
        np.dot(x, params[0].T) + params[1]
    )


def vmapReLULayer(params, x):

    return jit(
        vmap(relu_layer, in_axes=(None, 0), out_axes=0)
    )


out = np.stack(
    [relu_layer(params, X[i:]) for i in range(X.shape[0])]
)

out = batchVersionReLULayer(params, X)

out = vmapReLULayer(params, X)
