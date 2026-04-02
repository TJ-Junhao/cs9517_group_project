import argparse
import json

from project.utils.constant import CONFIG_PATH, REGULAR_CHANNEL_IN
from project.utils.registry import MODELS


def eval_arg_parse(prog: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=prog)

    parser.add_argument("-C", "--config", type=str, default=None)
    parser.add_argument("-M", "--model", choices=MODELS.keys(), default="unet")
    parser.add_argument("-R", "--run", type=str, default="undefined")
    parser.add_argument("-c", "--criteria", type=float, default=0.6)
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="test",
        choices=["test", "train", "validation"],
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=[],
        choices=["exg", "hsv"],
    )

    # first pass: only inspect model/config and whether extra args exist
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("-R", "--run", type=str, default=None)
    config_parser.add_argument("-C", "--config", type=str, default=None)
    config_parser.add_argument("-m", "--mode", type=str, default="test")
    pre_args, extras = config_parser.parse_known_args()

    if pre_args.config is not None:
        if extras:
            parser.error(
                f"When --config is used, only --resume is allowed, got: {extras}"
            )
        apply_config_defaults(parser, pre_args.run, pre_args.config)

    return parser.parse_args()


def train_arg_parse(prog: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("-M", "--model", choices=MODELS.keys(), default="unet")
    parser.add_argument("-R", "--run", type=str, default="undefined")
    parser.add_argument("-C", "--config", type=str, default=None)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("-ep", "--epoch", type=int, default=40)
    parser.add_argument("-p", "--patience", type=int, default=5)
    parser.add_argument("-md", "--min_delta", type=float, default=1e-4)
    parser.add_argument("-c", "--criteria", type=float, default=0.6)
    parser.add_argument("-b", "--batch_size", type=int, default=3)
    parser.add_argument("-r", "--resume", type=str, default=None)
    parser.add_argument(
        "--features",
        nargs="*",
        default=[],
        choices=["exg", "hsv"],
    )

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("-R", "--run", type=str, default=None)
    config_parser.add_argument("-C", "--config", type=str, default=None)
    config_parser.add_argument("-r", "--resume", type=str, default=None)

    pre_args, extras = config_parser.parse_known_args()

    if pre_args.config is not None:
        if extras:
            parser.error(
                f"When --config is used, only --resume is allowed as extra arg, got: {extras}"
            )
        apply_config_defaults(parser, pre_args.run, pre_args.config)

    return parser.parse_args()


def apply_config_defaults(
    parser: argparse.ArgumentParser,
    run_name: str,
    config_name: str,
) -> None:
    with open(CONFIG_PATH / config_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = data.get(run_name)
    if config is None:
        raise NameError(f"run name {run_name} not found in {config_name}")

    valid_keys = {
        "learning_rate",
        "epoch",
        "patience",
        "min_delta",
        "criteria",
        "features",
        "title",
        "batch_size",
        "model",
    }

    unknown_keys = set(config) - valid_keys
    if "resume" in unknown_keys:
        raise ValueError(
            "'resume' is runtime-only and must be provided via command line, not config"
        )
    if unknown_keys:
        raise ValueError(f"Unknown config keys: {sorted(unknown_keys)}")

    parser.set_defaults(**config)


def count_channels(feature_configs):
    base = REGULAR_CHANNEL_IN
    for f in feature_configs:
        if f["name"] in ["exg", "hsv"]:
            base += 1
    return base
