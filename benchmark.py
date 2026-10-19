
import numpy as np
import time
import math

# Method 1: Normal Python loops
def distance_python(pos):
    N = len(pos)
    result = [[0.0] * N for _ in range(N)]

    for i in range(N):
        for j in range(N):
            dx = pos[i][0] - pos[j][0]
            dy = pos[i][1] - pos[j][1]
            dz = pos[i][2] - pos[j][2]
            result[i][j] = math.sqrt(dx*dx + dy*dy + dz*dz)

    return result


# Method 2: NumPy broadcasting
def distance_numpy(pos):
    diff = pos[:, None, :] - pos[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


# Method 3: Gram matrix
def distance_gram(pos):
    sq = np.sum(pos**2, axis=1)
    dist_sq = sq[:, None] + sq[None, :] - 2 * (pos @ pos.T)
    np.maximum(dist_sq, 0, out=dist_sq)
    return np.sqrt(dist_sq)


# Test all three methods
n = 200
pos = np.random.default_rng(42).uniform(-5, 5, (n, 3))

methods = [
    ("Python loops", distance_python),
    ("NumPy broadcasting", distance_numpy),
    ("Gram matrix", distance_gram)
]

print("PARTICLE DISTANCE BENCHMARK")
print("Number of particles:", n)
print()

for name, func in methods:
    start = time.perf_counter()
    distances = func(pos)
    elapsed = (time.perf_counter() - start) * 1000

    print(name, ":", round(elapsed, 4), "ms")
    print("Matrix shape:", distances.shape if isinstance(distances, np.ndarray) else (len(distances), len(distances[0])))
    print()
