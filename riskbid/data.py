from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .config import resolve_path


@dataclass(frozen=True)
class ReturnPanel:
    name: str
    channels: tuple[str, ...]
    times: tuple[str, ...]
    train: NDArray[np.float64]
    test: NDArray[np.float64]
    train_times: tuple[str, ...]
    test_times: tuple[str, ...]


def _read_table(path: Path, file_format: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if file_format == "csv":
        return pd.read_csv(path)
    if file_format == "parquet":
        return pd.read_parquet(path)
    if file_format == "feather":
        return pd.read_feather(path)
    raise ValueError(f"Unsupported data format: {file_format}")


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output file extension: {suffix}")


def prepare_events(config: dict) -> Path:
    data = config["data"]
    settings = data.get("preparation")
    if not settings:
        raise ValueError("No data.preparation configuration was provided")
    source = resolve_path(config, settings["raw_path"])
    frame = _read_table(source, settings["raw_format"])
    periods = pd.to_datetime(frame[settings["timestamp_column"]], errors="raise", utc=True)
    frame = frame.assign(_period=periods.dt.tz_convert(None).dt.to_period(settings["frequency"]).dt.to_timestamp())
    channel_cols = settings["channel_columns"]
    if len(channel_cols) == 1:
        frame["_channel"] = frame[channel_cols[0]].astype(str)
    else:
        frame["_channel"] = frame[channel_cols].astype(str).agg(settings["channel_separator"].join, axis=1)
    reward_spec = settings["reward"]
    cost_spec = settings["cost"]
    if reward_spec["mode"] == "column":
        rewards = pd.to_numeric(frame[reward_spec["column"]], errors="raise").to_numpy(dtype=float)
    elif reward_spec["mode"] == "constant":
        rewards = np.full(len(frame), float(reward_spec["value"]))
    else:
        raise ValueError("Unknown reward mode")
    if cost_spec["mode"] == "column":
        costs = pd.to_numeric(frame[cost_spec["column"]], errors="raise").to_numpy(dtype=float)
    elif cost_spec["mode"] == "constant":
        costs = np.full(len(frame), float(cost_spec["value"]))
    else:
        raise ValueError("Unknown cost mode")
    frame["_reward"] = rewards * float(reward_spec["multiplier"])
    frame["_cost"] = costs * float(cost_spec["multiplier"])
    if not np.isfinite(frame[["_reward", "_cost"]].to_numpy()).all():
        raise ValueError("Raw reward and cost values must be finite")
    if (frame["_cost"] < 0).any():
        raise ValueError("Raw cost values must be nonnegative")
    aggregates = frame.groupby(["_period", "_channel"], sort=True)[["_reward", "_cost"]].sum()
    min_cost = float(settings["minimum_aggregate_cost"])
    if (aggregates["_cost"] <= min_cost).any():
        raise ValueError("Some period-channel groups have insufficient cost")
    aggregates["_roi"] = aggregates["_reward"] / aggregates["_cost"]
    wide = aggregates["_roi"].unstack("_channel").reset_index().rename(columns={"_period": data["timestamp_column"]})
    output = resolve_path(config, data["path"])
    _write_table(wide, output)
    return output


def load_panel(config: dict) -> ReturnPanel:
    settings = config["data"]
    frame = _read_table(resolve_path(config, settings["path"]), settings["format"])
    time_column = settings["timestamp_column"]
    if time_column not in frame:
        raise ValueError(f"Missing timestamp column: {time_column}")
    dates = pd.to_datetime(frame[time_column], errors="raise", utc=True)
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Panel timestamps must be nonnull and unique")
    frame = frame.assign(_timestamp=dates).sort_values("_timestamp", kind="stable")
    if settings["channels"] == "auto":
        channels = [column for column in frame if column not in (time_column, "_timestamp")]
    else:
        channels = list(settings["channels"])
    if len(channels) != int(settings["expected_channels"]):
        raise ValueError("The panel has the wrong number of channels")
    if len(channels) != len(set(channels)):
        raise ValueError("Channel names must be unique")
    if any(column not in frame.columns for column in channels):
        raise ValueError("One or more configured channel columns do not exist")
    values = frame[channels].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if values.shape[0] < int(settings["minimum_periods"]):
        raise ValueError("The panel does not contain enough periods")
    fraction = float(settings["train_fraction"])
    if not 0 < fraction < 1:
        raise ValueError("The training fraction must lie strictly between zero and one")
    cutoff = int(np.floor(len(values) * fraction))
    if cutoff <= 0 or cutoff >= len(values):
        raise ValueError("The chronological split is empty")
    missing = settings["missing_policy"]
    if missing == "error":
        if not np.isfinite(values).all():
            raise ValueError("The return panel has missing or infinite values")
    elif missing == "train_median":
        values[np.isinf(values)] = np.nan
        medians = np.nanmedian(values[:cutoff], axis=0)
        if not np.isfinite(medians).all():
            raise ValueError("Some channels have no valid training observations")
        indices = np.where(np.isnan(values))
        values[indices] = medians[indices[1]]
    elif missing == "forward_fill":
        values = pd.DataFrame(values).ffill().to_numpy(dtype=float)
    else:
        raise ValueError("Unknown missing value policy")
    if not np.isfinite(values).all():
        raise ValueError("The cleaned return panel still contains nonfinite values")
    transform = settings["normalization"]
    if transform["mode"] == "train_mean_uniform":
        divisor = float(np.mean(values[:cutoff].mean(axis=1)))
        if divisor <= 0:
            raise ValueError("The normalization divisor must be positive")
        values /= divisor
    elif transform["mode"] != "none":
        raise ValueError("Unknown return normalization")
    clock = tuple(frame["_timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return ReturnPanel(
        name=config["dataset"],
        channels=tuple(channels),
        times=clock,
        train=values[:cutoff].copy(),
        test=values[cutoff:].copy(),
        train_times=clock[:cutoff],
        test_times=clock[cutoff:],
    )
