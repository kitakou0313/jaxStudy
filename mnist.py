from jax._src.api import vmap
from jax.scipy.special import logsumexp
from jax.experimental import optimizers
from jax import random
import jax.numpy as np


from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

import time

if __name__ == "__main__":
    key = random.PRNGKey(1)
    batch_size = 100

    train_loader = DataLoader(
        datasets.MNIST(
            "./data", train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,)
                )
            ])
        ), batch_size=batch_size, shuffle=True
    )

    test_loader = DataLoader(
        datasets.MNIST(
            "./data", train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,)
                )
            ])
        ), batch_size=batch_size, shuffle=True
    )


def initialize_mlp(sizes, key):
    keys = random.split(
        key, len(sizes)
    )

    def initialize_layer(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(
            w_key, (n, m)
        ), scale * random.normal(
            b_key, (n,)
        )
    return [
        initialize_layer(m, n, k) for m, n, k in zip(
            sizes[:-1], sizes[1:], keys
        )
    ]


layer_sizes = [784, 512, 512, 10]

params = initialize_mlp(
    layer_sizes, key
)


def forward_pass(params, in_array):
    """
    forward pass
    """
    def relu_layer(params, x):
        def ReLU(x):
            return np.maximum(0, x)
        return ReLU(np.dot(params[0], x) + params[1])

    activations = in_array

    for w, b in params[:-1]:
        activations = relu_layer([w, b], activations)

    final_w, final_b = params[-1]
    logits = np.dot(
        final_w, activations
    ) + final_b

    return logits - logsumexp(logits)


batch_forward = vmap(
    forward_pass, in_axes=(None, 0), out_axes=0
)
