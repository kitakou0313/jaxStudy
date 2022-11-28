import neural_tangents as nt
from neural_tangents import stax

import jax.numpy as np
from jax import random
from jax.example_libraries.optimizers import l2_norm
from jax import jit, grad, vmap

def model_fn(x_train, x_test, y_train, kernel_fn, t=None,diag_reg=1e-4):
    ntk_train_train = kernel_fn(x_train, x_train, 'ntk')
    ntk_test_train = kernel_fn(x_test, x_train, 'ntk')

    predict_fn = nt.predict.gradient_descent_mse(ntk_train_train, y_train, diag_reg=diag_reg)
    fx = predict_fn(t, 0., 0., ntk_test_train)[1]

    return fx

def adv_loss(x_train, x_test, y_train, y_test, kernel_fn, loss='mse', t=None, targeted=False, diag_reg=1e-4, x_train_regs=None, alpha=0., beta=0.):
    """
    adv loss + ベクトルに関する制約項
    """

    # ADV loss
    ntk_train_train = kernel_fn(x_train, x_train, 'ntk')
    ntk_test_train = kernel_fn(x_test, x_train, 'ntk')

    predict_fn = nt.predict.gradient_descent_mse(ntk_train_train, y_train, diag_reg=diag_reg)
    fx = predict_fn(t, 0., 0., ntk_test_train)[1]

    l2_norm_loss = l2_norm(fx-y_test)

    # regularization loss
    regularization_loss = 0
    for x, X_trg in zip(x_train, x_train_regs):
        regularization_loss += np.sum(np.exp(-((np.linalg.norm(X_trg - x, ord=2, axis=1)**2) / beta)))

    print(l2_norm_loss, regularization_loss)
    return l2_norm_loss + alpha*regularization_loss

if __name__ == "__main__":
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(5, W_std=1.5, b_std=0.05), stax.Erf(),
        stax.Dense(1, W_std=1.5, b_std=0.05)
    )

    # kernel_fn = (kernel_fn)
    # grads_fn = (grad(adv_loss, argnums=0))
    kernel_fn = jit(kernel_fn,static_argnums=(2,))
    grads_fn = jit(grad(adv_loss, argnums=0),static_argnums=(4, 5, 7))

    x_train = np.array(
        [np.array([1,1,1,1,1], dtype=np.float32),
        np.array([1,1,1,1,1], dtype=np.float32)],
        dtype=np.float32)
    
    x_train_ori = x_train.copy()

    y_train = np.array(
        [5,5], dtype=np.float32
    )
    y_train = y_train.reshape((len(y_train),1))

    x_test = np.array(
        [[5,5,5,5,5],
        [6,6,6,6,6],
        [7,7,7,7,7]],dtype=np.float32
    )
    y_test = np.array(
        [8,9,1], dtype=np.float32
    )
    y_test = y_test.reshape((len(y_test), 1))

    x_1_trg = np.array(
        [[1.5, 1.5, 1.5, 1.5, 1.5],
         [0.5, 0.5, 0.5, 0.5, 0.5]], dtype=np.float32)

    x_2_trg = np.array(
        [[1.9, 1.9, 1.9, 1.9, 1.9],
         [2.5, 2.5, 2.5, 2.5, 2.5], 
         [3., 3., 3., 3., 3.]], dtype=np.float32)

    alpha = 7.0
    beta = 5.0


    # print(np.linalg.norm(x_1_trg - X[0],ord=2, axis=1))
    X_trg = [
        x_1_trg, x_2_trg
    ]

    print(adv_loss(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, kernel_fn=kernel_fn, x_train_regs=X_trg, alpha=alpha, beta=beta))
    print(grads_fn(x_train, x_test=x_test, y_train=y_train, y_test=y_test, kernel_fn=kernel_fn, x_train_regs=X_trg,alpha=alpha, beta=beta))

    loss_list = []
    for iteration in range(30):
        d_x = grads_fn(x_train, x_test=x_test, y_train=y_train, y_test=y_test, kernel_fn=kernel_fn, x_train_regs=X_trg,alpha=alpha, beta=beta)
        x_train += 0.1*d_x

        print(iteration+1)
        # print(d_x)

        loss =adv_loss(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, kernel_fn=kernel_fn, x_train_regs=X_trg, alpha=alpha, beta=beta) 
        loss_list.append(float(loss))
        # print(loss)
        # print(model_fn(x_train, x_test, y_train, kernel_fn))

        print()

    print(loss_list)

    print("Alpha", alpha, "beta", beta)
    print(x_train_ori)
    print(x_train)
