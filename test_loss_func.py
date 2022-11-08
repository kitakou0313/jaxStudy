import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
from jax import random


def loss_func(x: np.ndarray, trg_x: np.ndarray) -> np.ndarray:
    """
    trg_xの各行との距離にexpした値を返す
    """
    return np.sum(np.exp(-(np.linalg.norm(trg_x - x, ord=2, axis=1)**2)))


if __name__ == "__main__":
    # 到達できる候補ベクトル候補周辺に遷移するような損失関数を定義して微分可能か検証

    key = random.PRNGKey(1)
    x_1 = np.array([
        1, 1, 1, 1, 1
    ], dtype=np.float32)

    data = random.normal(
        key=key, shape=(5, 5), dtype=np.float32
    )
    data = np.array(
        [[1.5, 1.5, 1.5, 1.5, 1.5],
         [1.5, 1.5, 1.5, 1.5, 1.5],
         [1.5, 1.5, 1.5, 1.5, 1.5],
         [1.5, 1.5, 1.5, 1.5, 1.5]], dtype=np.float32)

    grad_loss_func = jit(grad(
        loss_func, argnums=0
    ))

    print(loss_func(x_1, data))
    print(x_1)
    print(grad_loss_func(x_1, data))

    for epoch in range(10):
        dx_1 = grad_loss_func(x_1, data)
        x_1 = x_1 + 0.1*dx_1

        print("Epoch", epoch, "Loss", loss_func(x_1, data))
        print(x_1)
