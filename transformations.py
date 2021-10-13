from jax import grad
import jax.numpy as jnp

def tanh(x):
    """
    docstring
    """
    y = jnp.exp(-2.0*x)
    return (1.0-y) / (1.0+y)

grad_tanh = grad(tanh)
print(grad_tanh(1.0))