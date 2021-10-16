from jax._src.api import value_and_grad, vmap
from jax._src.numpy.lax_numpy import array
from jax.scipy.special import logsumexp
from jax.experimental import optimizers
from jax import random, jit
import jax.numpy as np


from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

import time

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


def one_hot(x, k, dtype=np.float32):
    """
    docstring
    """
    return np.array(
        x[:, None] == np.arange(k), dtype
    )


def loss(params, in_array, targets):

    preds = batch_forward(
        params, in_array
    )

    return -np.sum(preds * targets)


num_classes = 10


def accuracy(params, data_loader):
    acc_total = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        images = np.array(data).reshape(
            data.size(0), 28*28
        )
        targets = one_hot(
            np.array(target), num_classes
        )

        target_class = np.argmax(
            targets, axis=1
        )

        predicted_class = np.argmax(
            batch_forward(params, images), axis=1
        )

        acc_total += np.sum(
            predicted_class == target_class
        )

    return acc_total/len(data_loader.dataset)


step_size = 1e-3
opt_init, opt_update, get_params = optimizers.adam(step_size)
opt_state = opt_init(params)


@jit
def update(params, x, y, opt_state):
    """
    Compute the gradient for batch and update the params
    """
    value, grads = value_and_grad(loss)(params, x, y)
    opt_state = opt_update(
        0, grads, opt_state
    )

    return get_params(
        opt_state
    ), opt_state, value


num_epochs = 10


def run_mnist_training_loop(num_epochs, opt_state, net_type="MLP"):
    """
    traing loop
    """
    log_acc_train, log_acc_test, train_loss = [], [], []

    params = get_params(opt_state)

    train_acc = accuracy(params, train_loader)
    test_acc = accuracy(params, test_loader)
    log_acc_train.append(train_acc)
    log_acc_test.append(test_acc)

    for epoch in range(num_epochs):
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            if net_type == "MLP":
                x = np.array(data).reshape(
                    data.size(0), 28*28
                )

            y = one_hot(
                np.array(target), num_classes
            )

            params, opt_state, loss = update(
                params, x, y, opt_state
            )

            train_loss.append(loss)

        epoch_time = time.time() - start_time
        train_acc = accuracy(params, train_loader)
        test_acc = accuracy(params, test_loader)
        log_acc_train.append(train_acc)
        log_acc_test.append(test_acc)
        print("Epoch {} | T: {:0.2f} | Train A: {:0.3f} | Test A: {:0.3f}".format(
            epoch+1, epoch_time, train_acc, test_acc))
    return train_loss, log_acc_train, log_acc_test


train_loss, train_log, test_log = run_mnist_training_loop(
    num_epochs, opt_state, net_type="MLP"
)
