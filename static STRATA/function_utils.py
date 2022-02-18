# function_utils: contains functions describing mathematical functions

from calendar import c
import random
import numpy as np

# uniform distribution
def distribution_uniform(min_val, max_val, param_a, param_b):
    # param A: unused
    # param B: unused

    val = random.random() * (max_val - min_val) + min_val
    return val

# gaussian distribution
def distribution_gaussian(min_val, max_val, param_a, param_b):
    # param A: mean
    # param B: stdev

    # sample until we get a point within the bounds
    while True:
        val = random.gauss(param_a, param_b)
        if min_val < val < max_val:
            break
    return val

# linear function: A * x + B
# note that this works for both singular values and numpy arrays
def function_linear(x, param_a, param_b, param_c):
    # param A: slope
    # param B: y shift
    # param C: unused

    val = x * param_a + param_b
    return val


# power function: A * (x ** B) + C
# note that this works for both singular values and numpy arrays
def function_power(x, param_a, param_b, param_c):
    # param A: x scalar
    # param B: x exponent
    # param C: y shift

    val = param_a * (x ** param_b) + param_c
    return val


# polynomial function: A1 * (x ** B1) + A2 * (x ** B2) + A3 * (x ** B3) + C
def function_polynomial(x, params_a, params_b, param_c):
    # params[] A: x scalars
    # params[] B: x exponents
    # param C: y shift

    # check if params are of equal length
    if len(params_a) != len(params_b):
        print("Error in function_polynomial: unequal parameter A/B lengths, A=", len(params_a), "B=", len(params_b))
        return 0

    val = param_c  # value is initially the y shift
    for i in range(len(params_a)):  # add each polynomial calculation
        val += params_a[i] * (x ** params_b[i])
    return val


