import numpy as np
import numba as nb

'''========== FAST INTEGRATION FUNCTIONS =========='''

@nb.njit(fastmath=True)
def integral(f, lower, upper, precision=10000):
    sign = 1
    if lower > upper:
        lower, upper = upper, lower
        sign = -1
    number_of_points = (upper - lower) * precision
    xs = np.linspace(lower, upper, int(number_of_points))
    integral = 0
    super_sum = 0
    sub_sum = 0
    for index in range(len(xs) - 1):
        delta = xs[index + 1] - xs[index]

        y1 = f(xs[index])
        sub_area = y1 * delta
        y2 = f(xs[index + 1])
        super_area = y2 * delta

        if 2 * delta == 0:
            continue
            
        area = (y2 + y1) / 2 * delta
        integral += area
        sub_sum += sub_area
        super_sum += super_area

    # error = super_sum - sub_sum
    return sign * integral

@nb.njit(fastmath=True)
def double_integral(f, x_limits, y_limits, precision=500):
    (a, b), (c, d) = x_limits, y_limits
    x_points, y_points = (b - a) * precision, (d - c) * precision
    xs, ys = np.linspace(a, b, int(x_points)), np.linspace(c, d, int(y_points))
    integral = 0
    sub_sum = 0
    super_sum = 0
    for i in range(len(xs) - 1):
        delta_x = xs[i + 1] - xs[i]
        for j in range(len(ys) - 1):
            delta_y = ys[j + 1] - ys[j]
            delta = delta_x * delta_y
            
            f1 = f(xs[i], ys[j])
            sub_area = f1 * delta
            f2 = f(xs[i + 1], ys[j + 1])
            super_area = f2 * delta

            if 2 * delta == 0:
                continue
            
            area = (f2 + f1) / 2 * delta
            integral += area
            sub_sum += sub_area
            super_sum += super_area

    # error = super_sum - sub_sum
    return integral

@nb.njit(fastmath=True)
def trap(f, n, lower_bound, upper_bound):
    h = (upper_bound - lower_bound) / float(n)
    intgr = 0.5 * h * (f(lower_bound) + f(upper_bound))
    for i in range(1, int(n)):
        intgr = intgr + h * f(lower_bound + i * h)
    return intgr

'''========== TEST INTEGRATION FUNCTIONS =========='''

@nb.njit(fastmath=True)
def simple_test(x):
    return np.sin(x) / x

result = integral(simple_test, -1, 1)
print(result)

@nb.njit(fastmath=True)
def double_gaussian(x, y):
    return np.exp(-(x ** 2 + y ** 2))

result = double_integral(double_gaussian, np.array([-500, 500]), np.array([-500, 500]), precision=50)
print(result)

@nb.njit(fastmath=True)
def f(x):
    return np.exp(-x ** 2)

lower = -10
upper = 10
n = 100
 
while(abs(trap(f, n, lower, upper) - trap(f, n * 4, lower * 2, upper * 2)) > 1e-6):
    n *= 4
    lower *= 2
    upper *= 2

print(trap(f, n, lower, upper))