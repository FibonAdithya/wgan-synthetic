"""CLI entrypoint for the descriptor EDA report.

The implementation lives in `src/eval/eda/`; this module exists so that
`python -m src.eval.eda_report`, the command in `README.md`,
`docs/datasets/*.md` and `check_gate.py`'s error messages, keeps working.

Nothing in this repo imports names from here; import from
`src.eval.eda.<module>` instead. (`parse_args` and `run` are importable off
this module too, as a side effect of the imports below -- that is harmless,
but this module is not where they are meant to be imported from.)
"""

from src.eval.eda.cli import parse_args
from src.eval.eda.pipeline import run


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
