import random
import numpy as np
from time import time

# basic numpy matrix multiplication
def matmul(n):
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    start = time()
    C = np.matmul(A, B)
    latency = time() - start
    return latency

# openfaas event handler function
def handle(event):
    input = [10, 100, 1000]
    n = random.randint(0, 2)
    result = matmul(input[n])
    return result