"""Small inspection CLI; training and cluster control remain explicit scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def data_explain(args: argparse.Namespace) -> int:
    from .config import load_recipe
    from .data.adapters import AdapterContext, get_adapter
    from .data.render import render_qwen3_5
    from .data.tokenize import LossPolicy, TokenizerSnapshot, explain_tokens, tokenize_document

    row = json.loads(Path(args.row).read_text())
    if not isinstance(row, dict):
        raise ValueError("row JSON must contain an object")
    context = AdapterContext(
        repo_id=args.repo_id,
        revision=args.revision,
        config=args.dataset_config,
        split=args.split,
        row_index=args.row_index,
    )
    sample = get_adapter(args.adapter)(row, context)
    document = render_qwen3_5(sample)
    snapshot, encoder = TokenizerSnapshot.load(args.tokenizer)
    policy = LossPolicy()
    if args.recipe:
        policy = LossPolicy.from_config(load_recipe(args.recipe).objective)
    tokenized = tokenize_document(
        document,
        encoder,
        tokenizer_hash=snapshot.identity_hash,
        policy=policy,
        max_length=args.max_length,
        truncation=args.truncation,
    )
    print(explain_tokens(document, tokenized, encoder))
    return 0


def model_inspect(args: argparse.Namespace) -> int:
    from .models.qwen3_5 import Qwen35Config, parameter_count

    config = Qwen35Config.from_json(args.config)
    result = {
        "architecture": "qwen3_5",
        "parameter_count": parameter_count(config),
        "parameter_gib_bfloat16": parameter_count(config) * 2 / 1024**3,
        "layers": list(config.layer_types),
        "config": config.__dict__,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jaxsft")
    groups = parser.add_subparsers(dest="group", required=True)
    data = groups.add_parser("data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    explain = data_commands.add_parser("explain")
    explain.add_argument("--row", required=True)
    explain.add_argument("--adapter", required=True)
    explain.add_argument("--tokenizer", required=True)
    explain.add_argument("--recipe")
    explain.add_argument("--repo-id", default="local")
    explain.add_argument("--revision", default="local")
    explain.add_argument("--dataset-config", default="default")
    explain.add_argument("--split", default="local")
    explain.add_argument("--row-index", type=int, default=0)
    explain.add_argument("--max-length", type=int)
    explain.add_argument("--truncation", choices=("reject", "right", "left"), default="reject")
    explain.set_defaults(function=data_explain)

    model = groups.add_parser("model")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    inspect = model_commands.add_parser("inspect")
    inspect.add_argument("--config", required=True)
    inspect.set_defaults(function=model_inspect)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
