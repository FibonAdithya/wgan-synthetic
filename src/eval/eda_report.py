"""CLI entrypoint for the descriptor EDA report.

The implementation lives in `src/eval/eda/`; this module exists so that
`python -m src.eval.eda_report`, the command in `README.md`,
`docs/datasets/*.md` and `check_gate.py`'s error messages, keeps working.

It deliberately re-exports nothing. Import from `src.eval.eda.<module>`.
"""

from src.eval.eda.cli import parse_args
from src.eval.eda.pipeline import run


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
