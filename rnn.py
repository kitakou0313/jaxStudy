from jax._src.api import value_and_grad, vmap
from jax._src.numpy.lax_numpy import array
from jax.scipy.special import logsumexp
from jax.experimental import optimizers
from jax import random, jit
import jax.numpy as np

import numpy as onp


def generate_ou_process(batch_size, num_dims, mu, tau, sigma, noise_std, dt=0.1):
    """
    テスト用データ生成
    """
    ou_x = onp.zeros((batch_size, num_dims))
    ou_x[:, 0] = onp.random.random(batch_size)

    for t in range(1, num_dims):
        dx = -(ou_x[:, t-1]-mu)/tau * dt + sigma*onp.sqrt(2/tau) * \
            onp.random.normal(0, 1, batch_size)*onp.sqrt(dt)
        ou_x[:, t] = ou_x[:, t-1] + dx

    ou_x_noise = ou_x + onp.random.multivariate_normal(
        onp.zeros(num_dims),
        noise_std*onp.eye(num_dims),
        batch_size
    )

    return ou_x, ou_x_noise


x_0, mu, tau, sigma, dt = 0, 1, 2, 0.5, 0.1
noise_std = 0.1
num_dims, batch_size = 100, 50

x, x_tilde = generate_ou_process(
    batch_size=batch_size,
    num_dims=num_dims,
    mu=mu,
    tau=tau,
    sigma=sigma,
    noise_std=noise_std,
    dt=dt
)
