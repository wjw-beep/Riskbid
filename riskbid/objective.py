from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import BoxSimplex


class OracleBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class ScoreComponents:
    score: float
    mean_return: float
    quantile: float
    violation_rate: float
    denominator: float


def score_components(
    outcomes: ArrayLike, alpha: float, kappa: float, rho: float
) -> ScoreComponents:
    values = np.asarray(outcomes, dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("The return scenarios must form a nonempty finite vector")
    if not (0 < alpha < 0.5 and kappa > 0 and rho > 0):
        raise ValueError("The score parameters are invalid")
    order = max(0, int(np.ceil(values.size * alpha)) - 1)
    quantile = float(np.partition(values, order)[order])
    mean = float(values.mean())
    denominator = max(rho - quantile, kappa * rho)
    return ScoreComponents(
        score=(mean - rho) / denominator,
        mean_return=mean,
        quantile=quantile,
        violation_rate=float(np.mean(values < rho)),
        denominator=denominator,
    )


class EmpiricalOracle:
    def __init__(
        self,
        returns: ArrayLike,
        geometry: BoxSimplex,
        alpha: float,
        kappa: float,
        rho: float,
        budget: int | None,
        smoothing_radius: float = 0.0,
        smoothing_samples: int = 0,
        rng: np.random.Generator | None = None,
        record_queries: bool = False,
        saturation_strength: float = 0.0,
        saturation_coefficients: ArrayLike | None = None,
    ) -> None:
        matrix = np.asarray(returns, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != geometry.dimension or matrix.shape[0] == 0:
            raise ValueError("The return matrix must have shape (n, channels)")
        if np.any(~np.isfinite(matrix)):
            raise ValueError("The return matrix contains nonfinite values")
        if budget is not None and budget < 0:
            raise ValueError("The oracle budget cannot be negative")
        if smoothing_radius < 0 or smoothing_samples < 0:
            raise ValueError("Smoothing parameters cannot be negative")
        self.returns = matrix
        self.geometry = geometry
        self.alpha = float(alpha)
        self.kappa = float(kappa)
        self.rho = float(rho)
        self.budget = budget
        self.smoothing_radius = float(smoothing_radius)
        self.smoothing_samples = int(smoothing_samples)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.basis = geometry.tangent_basis()
        self.count = 0
        self.queries = [] if record_queries else None
        self.saturation_strength = float(saturation_strength)
        self.saturation_coefficients = (
            np.zeros(geometry.dimension, dtype=float)
            if saturation_coefficients is None
            else np.asarray(saturation_coefficients, dtype=float)
        )
        if self.saturation_coefficients.shape != (geometry.dimension,):
            raise ValueError("Saturation coefficients must match the channel dimension")
        if self.saturation_strength < 0 or np.any(self.saturation_coefficients < 0):
            raise ValueError("Saturation parameters must be nonnegative")

    @property
    def primitive_cost(self) -> int:
        return max(1, self.smoothing_samples)

    def outcomes(self, x: ArrayLike) -> NDArray[np.float64]:
        point = np.asarray(x, dtype=float)
        if not self.geometry.contains(point):
            raise ValueError("The oracle received an infeasible allocation")
        weights = point * (1 - self.saturation_strength * self.saturation_coefficients * point)
        return self.returns @ weights

    def components(self, x: ArrayLike) -> ScoreComponents:
        return score_components(self.outcomes(x), self.alpha, self.kappa, self.rho)

    def _primitive(self, x: ArrayLike) -> float:
        if self.budget is not None and self.count >= self.budget:
            raise OracleBudgetExceeded("The primitive query budget has been exhausted")
        value = -self.components(x).score
        self.count += 1
        if self.queries is not None:
            self.queries.append(np.asarray(x, dtype=float).copy())
        return float(value)

    def evaluate(self, x: ArrayLike) -> float:
        if self.budget is not None and self.count + self.primitive_cost > self.budget:
            raise OracleBudgetExceeded("Not enough primitive queries remain")
        if self.smoothing_samples == 0:
            return self._primitive(x)
        if not self.geometry.contains(x):
            raise ValueError("The smoothing center must be feasible")
        radius = self.smoothing_radius
        dimension = self.basis.shape[1]
        total = 0.0
        for _ in range(self.smoothing_samples):
            draw = self.rng.normal(size=dimension)
            length = np.linalg.norm(draw)
            while length == 0:
                draw = self.rng.normal(size=dimension)
                length = np.linalg.norm(draw)
            draw *= self.rng.random() ** (1 / dimension) / length
            sample = np.asarray(x, dtype=float) + radius * (self.basis @ draw)
            total += self._primitive(sample)
        return total / self.smoothing_samples
