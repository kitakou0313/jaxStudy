import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
from jax import random


def loss_func(X: np.ndarray, X_trg_list: list, Y_trg:np.ndarray) -> np.ndarray:
    """
    trg_xの各行との距離にexpした値を返す
    """
    

    trg_loss = np.sum(np.linalg.norm(X-Y_trg, ord=2, axis=1)**2)

    regularization_loss = 0
    for x, X_trg in zip(X, X_trg_list):
        regularization_loss -= np.sum(np.exp(-((np.linalg.norm(X_trg - x, ord=2, axis=1)**2))))

    regularization_loss *= 15
    print(trg_loss, regularization_loss)

    return trg_loss+regularization_loss


def loss_func_for(x:np.ndarray, a_arr:list) -> np.ndarray:
    """
    docstring
    """
    res = 0
    for a in a_arr:
        res += a*x
    return res

if __name__ == "__main__":
    # 到達できる候補ベクトル候補周辺に遷移するような損失関数を定義して微分可能か検証

    key = random.PRNGKey(1)

    X = np.array(
        [
            [1.0,1.0,1.0,1.0,1.0],
            [2.0,2.0,2.0,2.0,2.0]
        ],dtype=np.float32)

    x_1_trg = np.array(
        [[1.5, 1.5, 1.5, 1.5, 1.5],
         [0.5, 0.5, 0.5, 0.5, 0.5]], dtype=np.float32)

    x_2_trg = np.array(
        [[1.9, 1.9, 1.9, 1.9, 1.9],
         [2.5, 2.5, 2.5, 2.5, 2.5], 
         [3., 3., 3., 3., 3.]], dtype=np.float32)


    # print(np.linalg.norm(x_1_trg - X[0],ord=2, axis=1))
    X_trg = [
        x_1_trg, x_2_trg
    ]

    Y_trg = np.array(
        [[3.,3.,3.,3.,3.],
        [0.,0.,0.,0.,0.,]]
    ,dtype=np.float32)

    grad_loss_func = jit(grad(
        loss_func, argnums=0
    ))

    for iteration in range(30):
        print("Iteraion", iteration)
        (loss_func(X, X_trg, Y_trg))
        d_X = grad_loss_func(X, X_trg, Y_trg)
        X += -0.1*d_X
        print(X)
        print()

    print("Trg vecs")
    print(Y_trg)