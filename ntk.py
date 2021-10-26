from jax._src.prng import _bit_stats
from jax._src.tree_util import tree_structure
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
from neural_tangents.utils.utils import named_tuple_factory
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".95"})
sns.set()


def loss_fn(predict_fn, ys, t, xs=None):
    mean, cov = predict_fn(t=t, get='ntk', x_test=xs, compute_cov=True)
    mean = np.reshape(mean, mean.shape[:1] + (-1,))
    var = np.diagonal(cov, axis1=1, axis2=2)
    ys = np.reshape(ys, (1, -1))

    mean_predictions = 0.5 * np.mean(ys ** 2 - 2 * mean * ys + var + mean ** 2,
                                     axis=1)

    return mean_predictions


# Define train datas
train_points = 5
test_points = 50
noise_scale = 1e-1


def target_fn(x): return np.sin(x)


key = random.PRNGKey(397)
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

prior_draws = []

for _ in range(10):
    key, net_key = random.split(key)
    _, params = init_fn(net_key, (-1, 1))
    prior_draws += [apply_fn(
        params, test_xs
    )]

plot_fn(train, test)

for p in prior_draws:
    plt.plot(test_xs, p, linewidth=3, alpha=0.7)
    plt.legend(['train', '$f(x)$', 'random draw'], loc='upper left')

plt.savefig("./plot/firstOutput")

kernel = kernel_fun(test_xs, test_xs, "nngp")
std_dev = np.sqrt(np.diag(kernel))

plot_fn(train, test)

plt.fill_between(np.reshape(test_xs, (-1,)), 2 *
                 std_dev, -2 * std_dev, alpha=0.4)

for p in prior_draws:
    plt.plot(test_xs, p, linewidth=3, alpha=0.5)
plt.savefig("./plot/nngpOutput")

# NTK
perdict_fun = nt.predict.gradient_descent_mse_ensemble(
    kernel_fun, train_xs, train_ys, diag_reg=1e-4
)

ntk_mean, ntk_covariance = perdict_fun(
    x_test=test_xs, get='ntk', compute_cov=True
)
print(ntk_mean)
ntk_mean = np.reshape(ntk_mean, (-1,))
print(ntk_mean)
ntk_std = np.sqrt(np.diag(ntk_covariance))

plot_fn(train, test)

plt.plot(test_xs, ntk_mean, 'b-', linewidth=3)
plt.fill_between(
    np.reshape(test_xs, (-1)),
    ntk_mean - 2 * ntk_std,
    ntk_mean + 2 * ntk_std,
    color='blue', alpha=0.2)

plt.xlim([-np.pi, np.pi])
plt.ylim([-1.5, 1.5])

plt.legend(['Train', 'f(x)', 'Bayesian Inference', 'Gradient Descent'],
           loc='upper left')

plt.savefig("./plot/ntk_prediction")

# inference finite time
ts = np.arange(
    0, 10 ** 3, 10**-1
)

print(ts)

ntk_train_loss_mean = loss_fn(perdict_fun, train_ys, ts)
ntk_test_loss_mean = loss_fn(perdict_fun, test_ys, ts, test_xs)

plt.subplot(1, 2, 1)

plt.loglog(ts, ntk_train_loss_mean, linewidth=3)
plt.loglog(ts, ntk_test_loss_mean, linewidth=3)

plt.legend(['Infinite Train', 'Infinite Test'])

plt.subplot(1, 2, 2)

plot_fn(train, None)

plt.plot(test_xs, ntk_mean, 'b-', linewidth=3)
plt.fill_between(
    np.reshape(test_xs, (-1)),
    ntk_mean - 2 * ntk_std,
    ntk_mean + 2 * ntk_std,
    color='blue', alpha=0.2)

plt.legend(
    ['Train', 'Infinite Network'],
    loc='upper left')

plt.savefig("./plot/train_log")

# comopare to train finite network
learning_rate = 0.1
training_steps = 10000

opt_init, opt_update, get_params = optimizers.sgd(
    learning_rate
)
opt_update = jit(opt_update)
loss = jit(lambda params, x, y: 0.5 * np.mean((apply_fn(params, x) - y)**2))
grad_loss = jit(lambda state, x, y: grad(loss)(get_params(state), x, y))

train_losses = []
test_losses = []

opt_state = opt_init(params)

for i in range(training_steps):
    opt_state = opt_update(i, grad_loss(
        opt_state, *train
    ), opt_state)

    train_losses += [loss(get_params(opt_state), *train)]
    test_losses += [loss(get_params(opt_state), *test)]

plt.subplot(1, 2, 1)

plt.loglog(ts, ntk_train_loss_mean, linewidth=3)
plt.loglog(ts, ntk_test_loss_mean, linewidth=3)

plt.loglog(ts, train_losses, 'k-', linewidth=2)
plt.loglog(ts, test_losses, 'k-', linewidth=2)

plt.legend(['Infinite Train', 'Infinite Test', 'Finite'])

plt.subplot(1, 2, 2)

plot_fn(train, None)

plt.plot(test_xs, ntk_mean, 'b-', linewidth=3)
plt.fill_between(
    np.reshape(test_xs, (-1)),
    ntk_mean - 2 * ntk_std,
    ntk_mean + 2 * ntk_std,
    color='blue', alpha=0.2)

plt.plot(test_xs, apply_fn(get_params(opt_state), test_xs), 'k-', linewidth=2)

plt.legend(
    ['Train', 'Infinite Network', 'Finite Network'],
    loc='upper left')

plt.savefig("./plot/compare_finite_network")
