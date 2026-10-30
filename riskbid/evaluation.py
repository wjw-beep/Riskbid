from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import spearmanr

from .baselines import run_baseline
from .config import resolve_path
from .data import ReturnPanel
from .geometry import BoxSimplex
from .objective import EmpiricalOracle, ScoreComponents, score_components
from .optimizer import RiskBid


def _rho(config: dict, returns: NDArray[np.float64], geometry: BoxSimplex, strength: float = 0.0) -> float:
    params = config["risk"]["rho"]
    if params["strategy"] == "fixed":
        value = float(params["value"])
    elif params["strategy"] == "equal_budget_training_mean":
        allocation = geometry.project(np.full(geometry.dimension, 1 / geometry.dimension))
        coefficients = np.asarray(config.get("response", {}).get("coefficients", np.zeros(geometry.dimension)), dtype=float)
        if coefficients.size != geometry.dimension:
            raise ValueError("The configured saturation vector has the wrong dimension")
        weights = allocation * (1 - strength * coefficients * allocation)
        value = float(np.mean(returns @ weights))
    else:
        raise ValueError("Unknown reference-floor strategy")
    value *= float(params["multiplier"])
    if not np.isfinite(value) or value <= 0:
        raise ValueError("The calibrated risk floor must be strictly positive")
    return value


def _oracle(
    data: NDArray[np.float64], geometry: BoxSimplex, config: dict, rho: float,
    budget: int | None, rng: np.random.Generator, record: bool = False,
    strength: float = 0.0,
) -> EmpiricalOracle:
    optimizer = config["optimizer"]
    coefficients = config.get("response", {}).get("coefficients")
    return EmpiricalOracle(
        returns=data,
        geometry=geometry,
        alpha=float(config["risk"]["alpha"]),
        kappa=float(config["risk"]["kappa"]),
        rho=rho,
        budget=budget,
        smoothing_radius=float(optimizer["smoothing_radius"]) if optimizer["mode"] == "smoothed" else 0,
        smoothing_samples=int(optimizer["smoothing_samples"]),
        rng=rng,
        record_queries=record,
        saturation_strength=strength,
        saturation_coefficients=coefficients,
    )


def _moving_block_indices(n: int, length: int, rng: np.random.Generator) -> NDArray[np.int64]:
    if not 1 <= length <= n:
        raise ValueError("The bootstrap block length must lie within the held-out horizon")
    blocks = int(np.ceil(n / length))
    starts = rng.integers(0, n, size=blocks)
    offsets = np.arange(length)
    return ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]


def _bootstrap(
    values: dict[str, list[NDArray[np.float64]]], alpha: float, kappa: float, rho: float,
    repetitions: int, block_length: int, seed: int, confidence_level: float,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    rng = np.random.default_rng(seed)
    if repetitions < 2:
        raise ValueError("At least two bootstrap replicates are required")
    if not 0 < confidence_level < 1:
        raise ValueError("The confidence level must be in (0, 1)")
    matrices = {name: np.stack(scenarios) for name, scenarios in values.items()}
    length = next(iter(matrices.values())).shape[1]
    if any(matrix.shape[1] != length for matrix in matrices.values()):
        raise ValueError("All methods must share the same held-out dates")
    draws = {name: np.empty(repetitions) for name in matrices}
    for replicate in range(repetitions):
        indices = _moving_block_indices(length, block_length, rng)
        for name, matrix in matrices.items():
            draws[name][replicate] = np.mean([
                score_components(outcomes[indices], alpha, kappa, rho).score
                for outcomes in matrix
            ])
    deviations = {name: float(np.std(samples, ddof=1)) for name, samples in draws.items()}
    paired = {}
    if "riskbid" in draws:
        for name, samples in draws.items():
            if name == "riskbid":
                continue
            differences = draws["riskbid"] - samples
            tail = (1 - confidence_level) / 2
            low, high = np.quantile(differences, [tail, 1 - tail])
            paired[name] = {
                "comparator": name,
                "difference_bootstrap_mean": float(differences.mean()),
                "difference_ci_low": float(low),
                "difference_ci_high": float(high),
            }
    return deviations, paired


def run_experiment(config: dict, panel: ReturnPanel) -> dict:
    geometry = BoxSimplex.from_settings(config["constraints"], len(panel.channels))
    if not geometry.contains(geometry.project(np.full(geometry.dimension, 1 / geometry.dimension))):
        raise ArithmeticError("Could not initialize the feasible allocation")
    rho = _rho(config, panel.train, geometry)
    methods = list(config["evaluation"]["methods"])
    seeds = list(config["evaluation"]["seeds"])
    if len(methods) != len(set(methods)):
        raise ValueError("Methods cannot be duplicated")
    output = resolve_path(config, config["evaluation"]["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "traces").mkdir(exist_ok=True)
    rows = []
    allocations = []
    test_returns: dict[str, list[NDArray[np.float64]]] = {name: [] for name in methods}
    for method in methods:
        for seed in seeds:
            rng = np.random.default_rng(int(seed))
            is_search = method == "riskbid"
            budget = int(config["optimizer"]["max_evaluations"] if is_search else config["baselines"]["max_evaluations"])
            oracle = _oracle(panel.train, geometry, config, rho, budget, rng)
            if is_search:
                optimizer = RiskBid(geometry, oracle, config["optimizer"])
                initialization = config["optimizer"]["initialization"]
                if initialization["mode"] == "equal":
                    initial = None
                elif initialization["mode"] == "random_tangent":
                    center = geometry.project(
                        np.full(geometry.dimension, 1 / geometry.dimension),
                        optimizer.tube_lower,
                        optimizer.tube_upper,
                    )
                    perturbation = optimizer.basis @ rng.normal(size=geometry.dimension - 1)
                    initial = geometry.project(
                        center + float(initialization["scale"]) * perturbation,
                        optimizer.tube_lower,
                        optimizer.tube_upper,
                    )
                else:
                    raise ValueError("Unknown initialization mode")
                result = optimizer.optimize(initial=initial)
            else:
                result = run_baseline(method, geometry, oracle, config["baselines"], rng)
            if not geometry.contains(result.allocation):
                raise ArithmeticError(f"Method {method} returned an infeasible allocation")
            train_components = oracle.components(result.allocation)
            test_oracle = _oracle(panel.test, geometry, config, rho, None, rng)
            test_components = test_oracle.components(result.allocation)
            test_returns[method].append(test_oracle.outcomes(result.allocation))
            row = {
                "dataset": panel.name,
                "method": method,
                "seed": int(seed),
                "train_score": train_components.score,
                "test_score": test_components.score,
                "test_mean_return": test_components.mean_return,
                "test_quantile": test_components.quantile,
                "test_violation_rate": test_components.violation_rate,
                "primitive_queries": result.evaluations,
                "iterations": result.iterations,
                "status": result.status,
            }
            rows.append(row)
            allocations.append({"method": method, "seed": int(seed), **dict(zip(panel.channels, result.allocation))})
            if result.trace:
                pd.DataFrame(result.trace).to_csv(output / "traces" / f"{method}_seed_{seed}.csv", index=False)
    detailed = pd.DataFrame(rows)
    detailed.to_csv(output / "seed_results.csv", index=False)
    pd.DataFrame(allocations).to_csv(output / "allocations.csv", index=False)
    evaluator = config["evaluation"]["bootstrap"]
    deviations, paired = _bootstrap(
        test_returns,
        alpha=float(config["risk"]["alpha"]),
        kappa=float(config["risk"]["kappa"]),
        rho=rho,
        repetitions=int(evaluator["replicates"]),
        block_length=int(evaluator["block_length"]),
        seed=int(evaluator["seed"]),
        confidence_level=float(evaluator["confidence_level"]),
    )
    summary = detailed.groupby("method", sort=False).agg(
        score_mean=("test_score", "mean"),
        mean_return=("test_mean_return", "mean"),
        quantile_mean=("test_quantile", "mean"),
        violation_rate=("test_violation_rate", "mean"),
        primitive_queries=("primitive_queries", "mean"),
    ).reset_index()
    summary["bootstrap_std"] = summary["method"].map(deviations)
    summary = summary.sort_values("score_mean", ascending=False)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(list(paired.values()), columns=[
        "comparator", "difference_bootstrap_mean", "difference_ci_low", "difference_ci_high"
    ]).to_csv(output / "paired_comparisons.csv", index=False)
    metadata = {
        "dataset": panel.name,
        "channels": list(panel.channels),
        "train_periods": len(panel.train),
        "test_periods": len(panel.test),
        "first_test_period": panel.test_times[0],
        "reference_floor": rho,
        "alpha": float(config["risk"]["alpha"]),
        "kappa": float(config["risk"]["kappa"]),
        "methods": methods,
        "seeds": seeds,
        "optimizer_mode": config["optimizer"]["mode"],
        "source_config": config["_config_path"],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"summary": summary, "paired": paired, "metadata": metadata, "output": str(output)}


def response_study(config: dict, panel: ReturnPanel) -> pd.DataFrame:
    if "response" not in config:
        raise ValueError("A response configuration is required")
    geometry = BoxSimplex.from_settings(config["constraints"], len(panel.channels))
    settings = config["response"]
    rng = np.random.default_rng(int(settings["seed"]))
    candidates = [geometry.project(np.full(geometry.dimension, 1 / geometry.dimension))]
    concentration = float(settings["candidate_concentration"])
    while len(candidates) < int(settings["candidate_count"]):
        candidates.append(geometry.project(rng.dirichlet(np.full(geometry.dimension, concentration))))
    matrix = np.stack(candidates)
    static_rho = _rho(config, panel.train, geometry)
    static_scores = np.array([
        score_components(panel.train @ x, config["risk"]["alpha"], config["risk"]["kappa"], static_rho).score
        for x in matrix
    ])
    static_index = int(np.argmax(static_scores))
    coefficients = np.asarray(settings["coefficients"], dtype=float)
    if coefficients.shape != (geometry.dimension,):
        raise ValueError("Response coefficients do not match the channel dimension")
    rows = []
    for strength in settings["strengths"]:
        strength = float(strength)
        response_rho = _rho(config, panel.train, geometry, strength=strength)
        response_scores = []
        for x in matrix:
            weights = x * (1 - strength * coefficients * x)
            returns = panel.train @ weights
            response_scores.append(score_components(
                returns, config["risk"]["alpha"], config["risk"]["kappa"], response_rho
            ).score)
        response_scores = np.asarray(response_scores)
        response_index = int(np.argmax(response_scores))
        def heldout(index: int) -> ScoreComponents:
            x = matrix[index]
            weights = x * (1 - strength * coefficients * x)
            return score_components(
                panel.test @ weights,
                config["risk"]["alpha"], config["risk"]["kappa"], response_rho
            )
        static_test = heldout(static_index)
        response_test = heldout(response_index)
        top_k = int(settings["top_k"])
        correlation = spearmanr(static_scores, response_scores).statistic
        rows.append({
            "strength": strength,
            "static_selected_score": static_test.score,
            "response_selected_score": response_test.score,
            "heldout_difference": response_test.score - static_test.score,
            "rank_correlation": float(correlation),
            "top_k_overlap": len(set(np.argsort(static_scores)[-top_k:]) & set(np.argsort(response_scores)[-top_k:])) / top_k,
        })
    output = resolve_path(config, config["evaluation"]["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output / "response_study.csv", index=False)
    return result
