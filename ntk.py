from jax._src.prng import _bit_stats
from jax.experimental.stax import Dense
import neural_tangents as nt
from neural_tangents import stax
import matplotlib.pyplot as plt
import jax.numpy as np

from jax import random
from jax.experimental import optimizers
from jax import jit, grad, vmap

import functools

from IPython.display import set_matplotlib_formats
import matplotlib
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".95"})
sns.set()


# Define train datas
train_points = 5
test_points = 50
noise_scale = 1e-1


def target_fn(x): return np.sin(x)


key = random.PRNGKey(11)
key, x_key, y_key = random.split(key=key, num=3)

# generate test data

train_xs = random.uniform(x_key, (train_points, 1),
                          minval=-np.pi, maxval=np.pi)
train_ys = target_fn(train_xs)
train_ys += noise_scale * random.normal(
    y_key, (train_points, 1)
)

train = (train_xs, train_ys)

test_xs = np.linspace(
    -np.pi, np.pi, test_points
)
test_xs = np.reshape(
    test_xs, (test_points, 1)
)

test_ys = target_fn(test_xs)
test = (test_xs, test_ys)


def plot_fn(train, test, *fs):
    """
    データプロット用関数
    """
    train_xs, train_ys = train
    plt.plot(
        train_xs, train_ys, "ro", markersize=10, label="train"
    )
    if test != None:
        test_xs, test_ys = test
        plt.plot(test_xs, test_ys, 'k--', linewidth=3, label='$f(x)$')

    for f in fs:
        plt.plot(test_xs, f(test_xs), '-', linewidth=3)

    plt.xlim([-np.pi, np.pi])
    plt.ylim([-1.5, 1.5])
    plt.xlabel("x")
    plt.ylabel("f(x)")


plt.figure(figsize=(6, 4))
plot_fn(train, test)
plt.legend(loc="upper left")
plt.savefig("./plot/TrainDatas")


init_fn, apply_fn, kernel_fun = stax.serial(
    stax.Dense(512, W_std=1.5, b_std=0.05), stax.Erf(),
    stax.Dense(512, W_std=1.5, b_std=0.05), stax.Erf(),
    stax.Dense(1, W_std=1.5, b_std=0.05)
)

apply_fn = jit(apply_fn)
kernel_fun = jit(kernel_fun, static_argnums=(2,))
