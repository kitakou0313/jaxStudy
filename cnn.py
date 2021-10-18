from jax._src.lax.lax import conv
from jax.experimental import stax
import jax.numpy as np
from jax.experimental import optimizers

from jax._src.api import value_and_grad, vmap

import time

from jax import random, jit
from jax.experimental.stax import (
    BatchNorm, Conv, Dense, Flatten, Relu, LogSoftmax
)


from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms


num_classes = 10
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

init_fun, conv_net = stax.serial(
    Conv(32, (5, 5), (2, 2), padding="SAME"),
    BatchNorm(), Relu,
    Conv(32, (5, 5), (2, 2), padding="SAME"),
    BatchNorm(), Relu,
    Conv(10, (3, 3), (2, 2), padding="SAME"),
    BatchNorm(), Relu,
    Conv(10, (3, 3), (2, 2), padding="SAME"), Relu,
    Flatten,
    Dense(num_classes),
    LogSoftmax
)

key = random.PRNGKey(1)

_, params = init_fun(
    key, (batch_size, 1, 28, 28)
)


def one_hot(x, k, dtype=np.float32):
    """
    docstring
    """
    return np.array(
        x[:, None] == np.arange(k), dtype
    )


def accuracy(params, data_loader):
    """
    docstring
    """
    acc_total = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        images = np.array(data)
        targets = one_hot(np.array(target), num_classes)

        target_class = np.argmax(
            targets, axis=1
        )

        predicted_class = np.argmax(
            conv_net(params, images),
            axis=1
        )

        acc_total += np.sum(
            predicted_class == target_class
        )

    return acc_total / len(data_loader.dataset)


def loss(params, images, targets):
    """
    docstring
    """
    preds = conv_net(params, images)
    return -np.sum(preds*targets)


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

            elif net_type == "CNN":
                # No flattening of the input required for the CNN
                x = np.array(data)

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


num_epochs = 10
train_loss, train_log, test_log = run_mnist_training_loop(
    num_epochs, opt_state, net_type="CNN"
)
