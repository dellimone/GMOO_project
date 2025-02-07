import numpy as np
from numpy.typing import NDArray

def sphere_function(x: NDArray) -> float:
    """
    Sphere Function
    Domain: xi in [-10, 10]
    Global Minimum: f(0,0,...,0) = 0
    """
    return float(np.sum(x ** 2))

def rastrigin_function(x: NDArray) -> float:
    """
    Rastrigin Function (Highly multimodal)
    Domain: xi in [-5.12, 5.12]
    Global Minimum: f(0,0,...,0) = 0
    """
    a = 10
    return a * len(x) + np.sum(x**2 - a * np.cos(2 * np.pi * x))

def ackley_function(x: NDArray) -> float:
    """
    Ackley Function
    Domain: xi in [-5, 5]
    Global Minimum: f(0,0,...,0) = 0
    """
    a, b, c = 20, 0.2, 2 * np.pi
    d = len(x)
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e

def rosenbrock_function(x: NDArray) -> float:
    """
    Rosenbrock Function (Ill-conditioned)
    Domain: xi in [-2, 2]
    Global Minimum: f(1,1,...,1) = 0
    """
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def griewank_function(x: NDArray) -> float:
    """
    Griewank Function (Complex landscape)
    Domain: xi in [-600, 600]
    Global Minimum: f(0,0,...,0) = 0
    """
    return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1

def schwefel_function(x: NDArray) -> float:
    """
    Schwefel Function (Deceptive global minimum)
    Domain: xi in [-500, 500]
    Global Minimum: f(420.9687, ..., 420.9687) = 0
    """
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def schaffer_function(x: NDArray) -> float:
    """
    Schaffer Function
    Domain: xi in [-100, 100]
    Global Minimum: f(0,0,...,0) = 0
    """
    term1 = np.sum(x**2)
    term2 = np.sum(np.sin(np.sqrt(term1))**2 - 0.5)
    term3 = (1 + 0.001 * term1)**2
    return 0.5 + term2 / term3

def levy_function(x: NDArray) -> float:
    """
    Levy Function (Steep valleys and ridges)
    Domain: xi [-10, 10]
    Global Minimum: f(1,1,...,1) = 0
    """
    w = 1 + (x - 1) / 4
    term1 = np.sin(np.pi * w[0])**2
    term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
    term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
    return term1 + term2 + term3

def zakharov_function(x: NDArray) -> float:
    """
    Zakharov Function
    Domain: xi in [-5, 10]
    Global Minimum: f(0,0,...,0) = 0
    """
    sum1 = np.sum(x**2)
    sum2 = np.sum(0.5 * np.arange(1, len(x) + 1) * x)
    return sum1 + sum2**2 + sum2**4

def dixon_price_function(x: NDArray) -> float:
    """
    Dixon-Price Function
    Domain: xi in [-10, 10]
    Global Minimum: f(2^{-((2^i - 2) / 2^i)}, ...) = 0
    """
    term1 = (x[0] - 1)**2
    term2 = np.sum((2 * np.arange(2, len(x) + 1) * x[1:]**2 - x[:-1])**2)
    return term1 + term2

def michalewicz_function(x: NDArray, m: float = 10) -> float:
    """
    Michalewicz Function  (Sharp local minima)
    Domain: xi in [0, pi]
    Global Minimum: Hard to determine analytically, varies with **m**
    """
    return -np.sum(np.sin(x) * (np.sin((np.arange(1, len(x) + 1) * x**2) / np.pi) ** (2 * m)))

def easom_function(x: NDArray) -> float:
    """
    Easom Function (Narrow global minimum)
    Domain: xi in [-100, 100]
    Global Minimum: f(pi, pi, ..., pi) = -1
    """
    return -np.prod(np.cos(x)) * np.exp(-np.sum((x - np.pi)**2))

def perm_function(x: NDArray, b: float = 0.5) -> float:
    """
    Perm Function (Difficult high-dimensional optimization)
    Domain: xi in [-d, d]
    Global Minimum: f(1, 1/2, 1/3, ..., 1/d) = 0
    """
    d = len(x)
    j = np.arange(1, d + 1)
    return np.sum([np.sum((j**k + b) * (x / j - 1))**2 for k in range(1, d + 1)])

def salomon_function(x: NDArray) -> float:
    """
    Salomon function: multimodal, non-separable function with global minimum at origin.
    Domain: xi in [-100, 100]
    Global Minimum: f(0, ..., 0) = 0
    """
    norm = np.sqrt(np.sum(x ** 2))
    return 1 - np.cos(2 * np.pi * norm) + 0.1 * norm

def xinsheyang1_func(x: np.ndarray) -> float:
    """
    Xin-She Yang's 1 Function (Nonconvex, Nonseparable)
    Domain: xi in [-2π, 2π]
    Global Minimum: f(0,0,...,0) = 0
    """
    abs_sum = np.sum(np.abs(x))
    exp_term = np.exp(-np.sum(np.sin(x**2)))
    return abs_sum * exp_term

def modxinsyang3_func(x: np.ndarray) -> float:
    """
    Modified Xin-She Yang's 3 Function (Nonconvex, Nonseparable)
    Domain: xi in [-20, 20]
    Global Minimum: f(0,0,...,0) = 0
    """
    exp_term1 = np.exp(-np.sum((x / 15) ** 10))
    exp_term2 = np.exp(-np.sum(x ** 2))
    cosine_term = np.prod(np.cos(x) ** 2)
    return 10**4 * (1 + (exp_term1 - 2 * exp_term2) * cosine_term)

def modxinsyang5_func(x: np.ndarray) -> float:
    """
    Modified Xin-She Yang's 5 Function (Nonconvex, Nonseparable)
    Domain: xi in [-100, 100]
    Global Minimum: f(0,0,...,0) = 0
    """
    sin_term = np.sum(np.sin(x) ** 2)
    exp_term1 = np.exp(-(np.sum(x ** 2)))
    exp_term2 = np.exp(-np.sum(np.sin(np.sqrt(np.abs(x))) ** 2))
    return 10**4 * (1 + (sin_term - exp_term1) * exp_term2)
