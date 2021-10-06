import numpy as np

from numpy.linalg import norm, inv, eigvals, solve

n = 100  # 99 + 1
a = 1.
b = 10.
c = 1.
p = 1.

f = np.arange(1, n + 1)

# Let's fill matrix A

A = np.zeros((n, n))

A[0][0] = b
A[0][1] = c

for i in range(1, n - 1):
    A[i][i - 1] = a
    A[i][i] = b
    A[i][i + 1] = c

for j in range(n):
    A[n - 1][j] = p


# Let's define functions for our methods
def lambda_max(A, u, lambda_true, err):
    max_lambda = 0

    while True:

        max_lambda = (A.dot(u).dot(u)) / (u.dot(u))

        if abs(max_lambda - lambda_true) < err:
            break

        u = A.dot(u) / norm(A.dot(u), 2)

    return max_lambda


def gauss(A, f):
    n = len(A)

    # Creating augmented matrix of system - B = (A | f)
    B = np.zeros((n, n + 1))
    for i in range(n):
        for j in range(n):
            B[i][j] = A[i][j]
        B[i][n] = f[i]

    # Starting algorithm
    # In this part we want to get an upper triangular matrix
    for j in range(n):
        # Search for maximum in column j below row j
        max_element = abs(B[j][j])
        number_max_row = j
        for i in range(j + 1, n):
            if abs(B[i][j]) > max_element:
                max_element = abs(B[i][j])
                number_max_row = i

        # Swap maximum row with current row (column by column)
        for k in range(j, n + 1):
            B[number_max_row][k], B[j][k] = B[j][k], B[number_max_row][k]

        # Make all rows below current row (j) equal to 0 in current column (j)
        for i in range(j + 1, n):  # Iteration throw rows
            c = -B[i][j] / B[j][j]
            B[i][j] = 0

            for k in range(j + 1, n + 1):  # iteration throw columns
                B[i][k] += c * B[j][k]

    # Solve Ax=f for an upper triangular A (B = (A | f))
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = B[i][n] / B[i][i]
        for k in range(i - 1, -1, -1):
            B[k][n] -= B[k][i] * x[i]
    return x


def seidel(A, f, err, x, x_precise):
    n = len(A)
    x_new = np.zeros_like(x)
    step = 1
    while True:

        for i in range(n):
            x_new[i] = (f[i] - sum(A[i][j] * x_new[j] for j in range(i))
                        - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]

        x = np.copy(x_new)

        residual_norm = norm(f - A.dot(x), 2)

        error_norm = norm(x - x_precise, 2)

        if error_norm < err:
            break

        # printing residuals:
        if step % 5 == 0:
            print("||r^(" + str(step) + ")|| =", residual_norm)

        step += 1

    return x


# Calculating eigen values with numpy
eig_vals = eigvals(A)
eig_vals.sort()

print("lambda_min = " + str(min(eig_vals)) + "\nlambda_max = " + str(max(eig_vals)))

#  lambda_min = 0.8998885030048327
#  lambda_max = 12.007560699871817

# Let's calculate lambda_max with power iterations
u_0 = np.ones(n)

print("lambda_max calculated with power iterations:", lambda_max(A, u_0, max(eig_vals), 10e-8))

# lambda_max calculated with power iterations: 12.007560799820835, совпало с значением, вычисленным с помощью numpy


# Let's calculate mu
inv_A = inv(A)
mu = np.sqrt(norm(A.transpose().dot(A), 2) * norm(inv_A.transpose().dot(inv_A), 2))
print("mu =", mu)

#  mu = 22.458341939715567


# Gauss and Seidel methods:
exact_x = solve(A, f)

x_seidel = seidel(A, f, 10e-12, np.zeros(n), exact_x)
x_gauss = gauss(A, f)

differ_exact_gauss = norm(exact_x - x_gauss, 2)
differ_exact_seidel = norm(exact_x - x_seidel, 2)
differ_seidel_gauss = norm(x_seidel - x_gauss, 2)

print("Difference exact - gauss:", differ_exact_gauss)
print("Difference exact - seidel:", differ_exact_seidel)
print("Difference seidel - gauss:", differ_seidel_gauss)

# Результаты:
# Difference exact - gauss: 0.0
# Difference exact - seidel: 6.612886602892705e-12
# Difference seidel - gauss: 6.612886602892705e-12
# Метод Гаусса - точный метод, решение совпадает с истинным. У метода Зейделя погрешность меньше err, которую передаем
# в функцию, критерий остановки - когда норма ||x^(s) - x_точное|| < err.
