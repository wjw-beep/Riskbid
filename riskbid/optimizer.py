from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .geometry import BoxSimplex
from .objective import EmpiricalOracle


@dataclass
class OptimizationResult:
    method: str
    allocation: NDArray[np.float64]
    evaluations: int
    iterations: int
    status: str
    trace: list[dict] = field(default_factory=list)


class RiskBid:
    def __init__(self, geometry: BoxSimplex, oracle: EmpiricalOracle, settings: dict):
        self.geometry = geometry
        self.oracle = oracle
        self.settings = settings
        self.basis = geometry.tangent_basis()
        self.query_radius = float(settings["query_radius"])
        self.smoothing_radius = float(settings["smoothing_radius"])
        self.safety_factor = float(settings["safety_factor"])
        self.max_iterations = int(settings["max_iterations"])
        self.stopping_tolerance = float(settings["stopping_tolerance"])
        self.min_step = float(settings["min_step"])
        self.return_policy = settings["return_policy"]
        if self.return_policy not in ("final", "best", "minimum_direction_norm"):
            raise ValueError("Unknown return policy")
        margin = self.query_radius + self.smoothing_radius
        self.tube_lower, self.tube_upper = geometry.tube(margin)
        if geometry.dimension < 2:
            raise ValueError("RiskBid requires at least two channels")
        if not (0 < self.safety_factor < 1):
            raise ValueError("The safety factor must lie strictly between zero and one")
        if self.oracle.smoothing_samples and self.oracle.smoothing_radius != self.smoothing_radius:
            raise ValueError("The oracle and optimizer must use the same smoothing radius")

    def step_size(self, iteration: int) -> float:
        schedule = self.settings["step_size"]
        initial = float(schedule["initial"])
        exponent = float(schedule["exponent"])
        kind = schedule["schedule"]
        if initial <= 0:
            raise ValueError("The nominal step must be positive")
        if kind == "constant":
            return initial
        if kind == "polynomial":
            return initial / (1 + iteration) ** exponent
        if kind == "inverse_sqrt":
            return initial / np.sqrt(1 + iteration)
        raise ValueError(f"Unknown step schedule: {kind}")

    def optimize(self, initial: NDArray[np.float64] | None = None) -> OptimizationResult:
        if initial is None:
            x = self.geometry.project(
                np.full(self.geometry.dimension, 1 / self.geometry.dimension),
                self.tube_lower,
                self.tube_upper,
            )
        else:
            x = np.asarray(initial, dtype=float).copy()
        if not self.geometry.contains(x, self.tube_lower, self.tube_upper):
            raise ValueError("The initial allocation must lie in the fixed query tube")
        if np.any(x <= self.geometry.lower) or np.any(x >= self.geometry.upper):
            raise ValueError("The initial allocation must be strictly interior")
        best = x.copy()
        min_norm_allocation = x.copy()
        best_value = float("inf")
        min_norm = float("inf")
        trace = []
        status = "iteration_limit"
        required = self.geometry.dimension * self.oracle.primitive_cost
        for iteration in range(self.max_iterations):
            if self.oracle.budget is not None and self.oracle.count + required > self.oracle.budget:
                status = "evaluation_limit"
                break
            center_value = self.oracle.evaluate(x)
            components = np.empty(self.geometry.dimension - 1, dtype=float)
            for column in range(self.geometry.dimension - 1):
                probe = x + self.query_radius * self.basis[:, column]
                if not self.geometry.contains(probe):
                    raise ArithmeticError("A finite-difference query left the feasible region")
                value = self.oracle.evaluate(probe)
                components[column] = (value - center_value) / self.query_radius
            lifted_gradient = self.basis @ components
            direction = self.geometry.riemannian_direction(x, lifted_gradient)
            inverse_metric = self.geometry.inverse_metric(x)
            norm = float(np.sqrt(np.sum(np.square(direction) / inverse_metric)))
            if center_value < best_value:
                best = x.copy()
                best_value = center_value
            if norm < min_norm:
                min_norm = norm
                min_norm_allocation = x.copy()
            maximum = self.geometry.max_safe_step(x, direction, self.tube_lower, self.tube_upper)
            step = min(self.step_size(iteration), self.safety_factor * maximum)
            if not np.isfinite(step):
                step = self.step_size(iteration)
            trace.append(
                {
                    "iteration": iteration,
                    "evaluations": self.oracle.count,
                    "objective": float(center_value),
                    "score": -float(center_value),
                    "best_score": -best_value,
                    "direction_norm": norm,
                    "nominal_step": self.step_size(iteration),
                    "realized_step": float(step),
                    "maximum_safe_step": float(maximum),
                    "minimum_margin": float(np.min(np.minimum(x - self.tube_lower, self.tube_upper - x))),
                }
            )
            if norm <= self.stopping_tolerance:
                status = "stationary_direction"
                break
            if step < self.min_step:
                status = "step_limit"
                break
            candidate = x + step * direction
            if not self.geometry.contains(candidate, self.tube_lower, self.tube_upper):
                raise ArithmeticError("The safeguard failed to preserve the query tube")
            x = candidate
        allocation = {
            "best": best,
            "final": x,
            "minimum_direction_norm": min_norm_allocation,
        }[self.return_policy]
        return OptimizationResult(
            method="riskbid",
            allocation=allocation.copy(),
            evaluations=self.oracle.count,
            iterations=len(trace),
            status=status,
            trace=trace,
        )
