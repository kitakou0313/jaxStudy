from jax._src.numpy.lax_numpy import block
import numpy as np

a = np.arange(12).reshape(3, 4)

print(a)

print(np.sum(a, axis=0))
print(
    a[0, :] + a[1, :] + a[2, :]
)

print(a[0, :])
print(a[0])

b = np.arange(24).reshape(2, 3, 4)
print(b)

print(np.sum(b, axis=(1, 2)))


print((1, None))
print((None, 1))
