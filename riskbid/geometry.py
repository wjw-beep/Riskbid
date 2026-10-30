from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import null_space


@dataclass(frozen=True)
class BoxSimplex:
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    tolerance: float = 1e-9

    @classmethod
    def from_settings(cls, settings: dict, dimension: int) -> "BoxSimplex":
        lower = np.broadcast_to(np.asarray(settings["lower"], dtype=float), (dimension,)).copy()
        upper = np.broadcast_to(np.asarray(settings["upper"], dtype=float), (dimension,)).copy()
        bounds = cls(lower=lower, upper=upper, tolerance=float(settings["tolerance"]))
        bounds.validate()
        return bounds

    @property
    def dimension(self) -> int:
        return self.lower.size

    def validate(self) -> None:
        if not (np.all(np.isfinite(self.lower)) and np.all(np.isfinite(self.upper))):
            raise ValueError("All constraint bounds must be finite")
        if np.any(self.lower < 0) or np.any(self.upper > 1) or np.any(self.lower >= self.upper):
            raise ValueError("Bounds must satisfy 0 <= lower < upper <= 1")
        if not (self.lower.sum() < 1 < self.upper.sum()):
            raise ValueError("The box-simplex must have a nonempty relative interior")

    def contains(self, x: ArrayLike, lower: ArrayLike | None = None, upper: ArrayLike | None = None) -> bool:
        point = np.asarray(x, dtype=float)
        low = self.lower if lower is None else np.asarray(lower, dtype=float)
        high = self.upper if upper is None else np.asarray(upper, dtype=float)
        return bool(
            point.shape == (self.dimension,)
            and np.all(np.isfinite(point))
            and abs(point.sum() - 1) <= self.tolerance
            and np.all(point >= low - self.tolerance)
            and np.all(point <= high + self.tolerance)
        )

    def tube(self, margin: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if margin < 0:
            raise ValueError("The query margin cannot be negative")
        low = self.lower + margin
        high = self.upper - margin
        if np.any(low > high) or low.sum() > 1 + self.tolerance or high.sum() < 1 - self.tolerance:
            raise ValueError("The requested query tube is empty")
        return low, high

    def project(self, point: ArrayLike, lower: ArrayLike | None = None, upper: ArrayLike | None = None) -> NDArray[np.float64]:
        value = np.asarray(point, dtype=float)
        low = self.lower if lower is None else np.asarray(lower, dtype=float)
        high = self.upper if upper is None else np.asarray(upper, dtype=float)
        if value.shape != (self.dimension,) or np.any(~np.isfinite(value)):
            raise ValueError("The projected point has an invalid shape or contains nonfinite values")
        if np.any(low > high) or low.sum() > 1 + self.tolerance or high.sum() < 1 - self.tolerance:
            raise ValueError("The requested projected region is empty")
        left = float(np.min(value - high))
        right = float(np.max(value - low))
        for _ in range(100):
            shift = (left + right) / 2
            total = np.clip(value - shift, low, high).sum()
            if total > 1:
                left = shift
            else:
                right = shift
        result = np.clip(value - (left + right) / 2, low, high)
        if not self.contains(result, low, high):
            raise ArithmeticError("Projection could not satisfy the budget and box constraints")
        return result

    def tangent_basis(self) -> NDArray[np.float64]:
        return null_space(np.ones((1, self.dimension)))

    def inverse_metric(self, x: ArrayLike) -> NDArray[np.float64]:
        point = np.asarray(x, dtype=float)
        if np.any(point <= self.lower) or np.any(point >= self.upper):
            raise ValueError("The barrier metric requires a strictly interior allocation")
        diagonal = 1 / (point - self.lower)
        bounded = self.upper < 1
        diagonal[bounded] += 1 / np.square(self.upper[bounded] - point[bounded])
        return 1 / diagonal

    def riemannian_direction(self, x: ArrayLike, gradient: ArrayLike) -> NDArray[np.float64]:
        weights = self.inverse_metric(x)
        gradient = np.asarray(gradient, dtype=float)
        lagrange = np.dot(weights, gradient) / weights.sum()
        return -weights * (gradient - lagrange)

    def max_safe_step(self, x: ArrayLike, direction: ArrayLike, low: ArrayLike, high: ArrayLike) -> float:
        point = np.asarray(x, dtype=float)
        delta = np.asarray(direction, dtype=float)
        minimum = np.asarray(low, dtype=float)
        maximum = np.asarray(high, dtype=float)
        positive = delta > 0
        negative = delta < 0
        bounds = []
        if np.any(positive):
            bounds.append(np.min((maximum[positive] - point[positive]) / delta[positive]))
        if np.any(negative):
            bounds.append(np.min((point[negative] - minimum[negative]) / (-delta[negative])))
        return max(0.0, float(min(bounds))) if bounds else float("inf")

    def linear_minimizer(self, gradient: ArrayLike) -> NDArray[np.float64]:
        coefficient = np.asarray(gradient, dtype=float)
        allocation = self.lower.copy()
        remaining = 1 - allocation.sum()
        for index in np.argsort(coefficient, kind="stable"):
            increase = min(remaining, self.upper[index] - allocation[index])
            allocation[index] += increase
            remaining -= increase
            if remaining <= self.tolerance:
                break
        if not self.contains(allocation):
            raise ArithmeticError("The linear minimization oracle returned an infeasible point")
        return allocation
