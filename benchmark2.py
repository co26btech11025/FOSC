
import numpy as np
import time
import math
import matplotlib.pyplot as plt

# Settings
PARTICLE_COUNTS = [50, 200, 500, 1000]
REPEATS = 5


# Method 1: Pure Python loops
def distance_python(pos):
    n = len(pos)
    result = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            result[i][j] = math.sqrt(dx*dx + dy*dy)

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


methods = [
    ("Python loops", distance_python),
    ("NumPy broadcasting", distance_numpy),
    ("Gram matrix", distance_gram)
]

results = {name: [] for name, func in methods}
rng = np.random.default_rng(42)

print("PARTICLE DISTANCE PERFORMANCE BENCHMARK")
print("Coordinates: 2D")
print("Trials per method:", REPEATS)
print("Timing statistic: median")
print()

for n in PARTICLE_COUNTS:
    positions = rng.uniform(-5, 5, (n, 2))
    print("Particles:", n)

    for name, func in methods:
        # Warm-up run
        func(positions)

        times = []

        for _ in range(REPEATS):
            start = time.perf_counter()
            distances = func(positions)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        median = float(np.median(times))
        results[name].append(median)

        print(" ", name, "median:", round(median, 4), "ms")

    print()


# Calculate speedups and theoretical memory usage
print("SPEEDUP AND MEMORY ANALYSIS")
print()

speedup_numpy = []
speedup_gram = []
memory_float64 = []
memory_float32 = []

for i, n in enumerate(PARTICLE_COUNTS):
    py_time = results["Python loops"][i]
    np_time = results["NumPy broadcasting"][i]
    gram_time = results["Gram matrix"][i]

    s_numpy = py_time / np_time
    s_gram = py_time / gram_time

    speedup_numpy.append(s_numpy)
    speedup_gram.append(s_gram)

    # Broadcasting displacement tensor: (N, N, 2)
    mem64 = n**2 * 2 * 8 / (1024**2)
    mem32 = n**2 * 2 * 4 / (1024**2)

    memory_float64.append(mem64)
    memory_float32.append(mem32)

    print(
        f"N={n}: NumPy speedup={s_numpy:.2f}x, "
        f"Gram speedup={s_gram:.2f}x"
    )
    print(
        f"  Theoretical displacement tensor memory: "
        f"float64={mem64:.4f} MB, float32={mem32:.4f} MB"
    )

print()


# Save benchmark data to CSV
with open("benchmark_results.csv", "w", encoding="utf-8") as file:
    file.write(
        "Particles,Python loops ms,NumPy broadcasting ms,"
        "Gram matrix ms,NumPy speedup,Gram speedup,"
        "Float64 memory MB,Float32 memory MB\n"
    )

    for i, n in enumerate(PARTICLE_COUNTS):
        file.write(
            f"{n},"
            f"{results['Python loops'][i]},"
            f"{results['NumPy broadcasting'][i]},"
            f"{results['Gram matrix'][i]},"
            f"{speedup_numpy[i]},"
            f"{speedup_gram[i]},"
            f"{memory_float64[i]},"
            f"{memory_float32[i]}\n"
        )


# Graph 1: Log-log execution time
plt.figure(figsize=(8, 6))

for name, func in methods:
    plt.loglog(
        PARTICLE_COUNTS,
        results[name],
        marker="o",
        label=name
    )

# Reference O(N^2) curve, anchored to first Gram result
reference = [
    results["Gram matrix"][0] * (n / PARTICLE_COUNTS[0])**2
    for n in PARTICLE_COUNTS
]

plt.loglog(
    PARTICLE_COUNTS,
    reference,
    linestyle="--",
    label="Reference O(N^2)"
)

plt.xlabel("Number of particles (N)")
plt.ylabel("Median execution time (ms)")
plt.title("Execution Time Scaling")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("task4_execution_time.png", dpi=300)
plt.show()


# Graph 2: Speedup bar chart
x = np.arange(len(PARTICLE_COUNTS))
width = 0.35

plt.figure(figsize=(8, 6))
plt.bar(
    x - width/2, speedup_numpy, width,
    label="NumPy broadcasting"
)
plt.bar(
    x + width/2, speedup_gram, width,
    label="Gram matrix"
)

plt.xticks(x, [str(n) for n in PARTICLE_COUNTS])
plt.xlabel("Number of particles (N)")
plt.ylabel("Speedup relative to Python loops")
plt.title("Optimization Speedup")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("task4_speedup.png", dpi=300)
plt.show()


# Graph 3: Theoretical memory scaling
plt.figure(figsize=(8, 6))

plt.plot(
    PARTICLE_COUNTS, memory_float64,
    marker="o", label="float64"
)
plt.plot(
    PARTICLE_COUNTS, memory_float32,
    marker="s", label="float32"
)

plt.xlabel("Number of particles (N)")
plt.ylabel("Memory (MiB)")
plt.title("Theoretical Displacement Tensor Memory")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("task4_memory_scaling.png", dpi=300)
plt.show()


print("Done! Files saved:")
print("1. benchmark_results.csv")
print("2. task4_execution_time.png")
print("3. task4_speedup.png")
print("4. task4_memory_scaling.png")
