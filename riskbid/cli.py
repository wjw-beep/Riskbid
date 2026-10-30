import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, resolve_path
from .data import load_panel, prepare_events
from .evaluation import response_study, run_experiment


def generate_demo(config: dict) -> Path:
    settings = config.get("synthetic")
    if settings is None:
        raise ValueError("The dataset YAML does not define synthetic generation")
    rng = np.random.default_rng(int(settings["seed"]))
    dimension = int(config["data"]["expected_channels"])
    length = int(settings["periods"])
    means = np.asarray(settings["channel_means"], dtype=float)
    scales = np.asarray(settings["channel_scales"], dtype=float)
    if means.shape != (dimension,) or scales.shape != (dimension,):
        raise ValueError("Synthetic channel parameters do not match the dimension")
    common = rng.normal(size=(length, 1)) * float(settings["common_volatility"])
    independent = rng.normal(size=(length, dimension)) * scales
    shock = rng.random((length, 1)) < float(settings["shock_probability"])
    values = means + common + independent - shock * float(settings["shock_scale"])
    dates = pd.date_range(settings["start_date"], periods=length, freq=settings["frequency"])
    names = [settings["channel_prefix"] + str(index) for index in range(dimension)]
    frame = pd.DataFrame(values, columns=names)
    frame.insert(0, config["data"]["timestamp_column"], dates)
    output = resolve_path(config, config["data"]["path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="riskbid")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "response-study", "generate-demo", "validate"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "generate-demo":
        print(generate_demo(config))
    elif args.command == "prepare":
        print(prepare_events(config))
    elif args.command == "validate":
        panel = load_panel(config)
        print(f"{panel.name}: train={len(panel.train)} test={len(panel.test)} channels={len(panel.channels)}")
    elif args.command == "run":
        result = run_experiment(config, load_panel(config))
        print(result["summary"].to_string(index=False))
        print(f"Outputs: {result['output']}")
    elif args.command == "response-study":
        result = response_study(config, load_panel(config))
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
