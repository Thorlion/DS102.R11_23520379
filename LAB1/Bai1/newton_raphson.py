import numpy as np


def f(x):
    """
    f(x, y) = x^4 + y^2 + 2xy + 1
    """
    return x[0]**4 + x[1]**2 + 2 * x[0] * x[1] + 1


def gradient(u):
    """
    Đạo hàm bậc 1
    dx = 4x^3 + 2y
    dy = 2y + 2x
    """
    x, y = u
    return np.array([
        4 * x**3 + 2 * y,
        2 * y + 2 * x
    ], dtype=float)


def hessian(u):
    """
    Đạo hàm bậc 2
    Hessian = [12 2]
              [ 2 2]
    """
    x, y = u
    return np.array([
        [12 * x**2, 2],
        [2, 2]
    ], dtype=float)


def newton_optimizer(u, tol=1e-8, max_iter=100):
    """ Thuật toán tìm kiếm điểm cực trị """
    u = np.array(u, dtype=float)

    for iter in range(max_iter):
        grad = gradient(u)
        hes = hessian(u)

        print(f"Step: {iter}")
        print(f" - Gradient: [{grad[0]:.3f} {grad[1]:.3f}]")
        
        if np.linalg.norm(grad) < tol:
            break

        step = np.linalg.solve(hes, grad)
        u = u - step

        print(f" - x: {u[0]:.3f}")
        print(f" - y: {u[1]:.3f}")
        print(f" - Value: {f(u):.3f}")
        
    return u, f(u)


if __name__ == "__main__":
    # Điểm bắt đầu
    x0 = np.array([2.0, 3.0])
    
    x_min, f_min = newton_optimizer(x0)

    print(f"\nMinimum position: [{x_min[0]:.3f} {x_min[1]:.3f}]")
    print(f"Minimum value: {f_min:.3f}")
