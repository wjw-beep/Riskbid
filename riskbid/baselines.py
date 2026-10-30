from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog, minimize

from .geometry import BoxSimplex
from .objective import EmpiricalOracle
from .optimizer import OptimizationResult


def _constant_result(method: str, allocation: NDArray[np.float64]) -> OptimizationResult:
    return OptimizationResult(method=method, allocation=allocation, evaluations=0, iterations=0, status="completed")


def equal_budget(geometry: BoxSimplex) -> OptimizationResult:
    return _constant_result("equal_budget", geometry.project(np.full(geometry.dimension, 1 / geometry.dimension)))


def mean_greedy(geometry: BoxSimplex, returns: NDArray[np.float64]) -> OptimizationResult:
    return _constant_result("mean_greedy", geometry.linear_minimizer(-returns.mean(axis=0)))


def _scipy_heuristic(
    name: str, geometry: BoxSimplex, target: Callable[[NDArray[np.float64]], float], settings: dict
) -> OptimizationResult:
    start = equal_budget(geometry).allocation
    optimum = minimize(
        target,
        start,
        method="SLSQP",
        bounds=list(zip(geometry.lower, geometry.upper)),
        constraints=[{"type": "eq", "fun": lambda x: float(np.sum(x) - 1)}],
        options={
            "maxiter": int(settings["solver_iterations"]),
            "ftol": float(settings["solver_tolerance"]),
        },
    )
    allocation = geometry.project(optimum.x)
    return _constant_result(name, allocation)


def sharpe_greedy(geometry: BoxSimplex, returns: NDArray[np.float64], settings: dict) -> OptimizationResult:
    def objective(x: NDArray[np.float64]) -> float:
        scenario = returns @ x
        return -float(scenario.mean() / max(scenario.std(), float(settings["volatility_floor"])))
    return _scipy_heuristic("sharpe_greedy", geometry, objective, settings)


def risk_parity(geometry: BoxSimplex, returns: NDArray[np.float64], settings: dict) -> OptimizationResult:
    covariance = np.cov(returns, rowvar=False, ddof=1)
    ridge = float(settings["covariance_regularization"])
    covariance += ridge * np.eye(geometry.dimension)
    def objective(x: NDArray[np.float64]) -> float:
        contribution = x * (covariance @ x)
        return float(np.sum(np.square(contribution - contribution.mean())))
    return _scipy_heuristic("risk_parity", geometry, objective, settings)


def cvar_allocation(
    geometry: BoxSimplex, returns: NDArray[np.float64], alpha: float, settings: dict
) -> OptimizationResult:
    n, dimension = returns.shape
    coefficients = np.concatenate([np.zeros(dimension), [-1.0], np.full(n, 1 / (alpha * n))])
    matrix = np.zeros((n, dimension + 1 + n))
    matrix[:, :dimension] = -returns
    matrix[:, dimension] = 1
    matrix[:, dimension + 1:] = -np.eye(n)
    rhs = np.zeros(n)
    equality = np.zeros((1, dimension + 1 + n))
    equality[0, :dimension] = 1
    variable_bounds = (
        list(zip(geometry.lower, geometry.upper))
        + [(None, None)]
        + [(0, None)] * n
    )
    solution = linprog(
        coefficients,
        A_ub=matrix,
        b_ub=rhs,
        A_eq=equality,
        b_eq=np.ones(1),
        bounds=variable_bounds,
        method=settings["linear_solver"],
    )
    if not solution.success:
        raise RuntimeError(f"CVaR optimization failed: {solution.message}")
    return _constant_result("cvar", geometry.project(solution.x[:dimension]))


def _finite_difference(
    oracle: EmpiricalOracle, geometry: BoxSimplex, x: NDArray[np.float64], radius: float
) -> tuple[float, NDArray[np.float64]]:
    basis = geometry.tangent_basis()
    center = oracle.evaluate(x)
    reduced = np.empty(geometry.dimension - 1)
    for index in range(geometry.dimension - 1):
        delta = radius * basis[:, index]
        if not geometry.contains(x + delta):
            raise ArithmeticError("A baseline finite-difference probe left the feasible region")
        reduced[index] = (oracle.evaluate(x + delta) - center) / radius
    return center, basis @ reduced


def zeroth_order(
    name: str,
    geometry: BoxSimplex,
    oracle: EmpiricalOracle,
    settings: dict,
    rng: np.random.Generator,
) -> OptimizationResult:
    algorithm = settings["zeroth_order"]
    radius = float(algorithm["query_radius"])
    low, high = geometry.tube(radius + oracle.smoothing_radius)
    initial = np.full(geometry.dimension, 1 / geometry.dimension)
    x = geometry.project(initial, low, high)
    best = x.copy()
    best_value = float("inf")
    step = float(algorithm["step_size"])
    trace = []
    status = "iteration_limit"
    cost = geometry.dimension * oracle.primitive_cost
    for iteration in range(int(algorithm["max_iterations"])):
        if oracle.budget is not None and oracle.count + cost > oracle.budget:
            status = "evaluation_limit"
            break
        value, gradient = _finite_difference(oracle, geometry, x, radius)
        if value < best_value:
            best_value = value
            best = x.copy()
        if name == "zo_pgd":
            candidate = geometry.project(x - step * gradient, low, high)
        elif name == "zo_sfw":
            vertex = geometry.linear_minimizer(gradient)
            candidate = geometry.project(x + step * (vertex - x), low, high)
        elif name == "zo_signsgd":
            sign = np.sign(gradient)
            sign -= sign.mean()
            candidate = geometry.project(x - step * sign, low, high)
        else:
            raise ValueError(f"Unknown zeroth-order baseline: {name}")
        trace.append({
            "iteration": iteration,
            "evaluations": oracle.count,
            "objective": float(value),
            "score": -float(value),
            "best_score": -best_value,
        })
        x = candidate
    return OptimizationResult(
        method=name,
        allocation=best if algorithm["return_policy"] == "best" else x,
        evaluations=oracle.count,
        iterations=len(trace),
        status=status,
        trace=trace,
    )


def random_search(
    geometry: BoxSimplex,
    oracle: EmpiricalOracle,
    settings: dict,
    rng: np.random.Generator,
) -> OptimizationResult:
    concentration = float(settings["random_search"]["concentration"])
    best = None
    best_value = float("inf")
    trace = []
    count = 0
    while oracle.budget is None or oracle.count + oracle.primitive_cost <= oracle.budget:
        if count == 0:
            candidate = equal_budget(geometry).allocation
        else:
            sample = rng.dirichlet(np.full(geometry.dimension, concentration))
            candidate = geometry.project(sample)
        value = oracle.evaluate(candidate)
        if value < best_value:
            best_value = value
            best = candidate.copy()
        count += 1
        if count % int(settings["random_search"]["logging_interval"]) == 0:
            trace.append({"iteration": count - 1, "evaluations": oracle.count, "best_score": -best_value})
        if count >= int(settings["random_search"]["max_candidates"]):
            break
    if best is None:
        raise ValueError("The random-search oracle budget is too small")
    return OptimizationResult(
        method="random_search",
        allocation=best,
        evaluations=oracle.count,
        iterations=count,
        status="completed",
        trace=trace,
    )


def run_baseline(
    name: str,
    geometry: BoxSimplex,
    oracle: EmpiricalOracle,
    settings: dict,
    rng: np.random.Generator,
) -> OptimizationResult:
    if name == "equal_budget":
        return equal_budget(geometry)
    if name == "mean_greedy":
        return mean_greedy(geometry, oracle.returns)
    if name == "sharpe_greedy":
        return sharpe_greedy(geometry, oracle.returns, settings)
    if name == "risk_parity":
        return risk_parity(geometry, oracle.returns, settings)
    if name == "cvar":
        return cvar_allocation(geometry, oracle.returns, oracle.alpha, settings)
    if name in ("zo_pgd", "zo_sfw", "zo_signsgd"):
        return zeroth_order(name, geometry, oracle, settings, rng)
    if name == "random_search":
        return random_search(geometry, oracle, settings, rng)
    raise ValueError(f"Unsupported baseline: {name}")
