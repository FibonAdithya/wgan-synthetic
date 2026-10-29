"""Write a v2 config with `seed` and `output_dir` replaced and nothing else changed.

Usage: seed_config.py CONFIG SEED OUTPUT_DIR OUT
run.sh passes the run dir as OUTPUT_DIR, so the trainer writes where the later steps look.
"""

import sys
from pathlib import Path

import yaml


def main(argv):
    if len(argv) != 4:
        raise SystemExit("usage: seed_config.py CONFIG SEED OUTPUT_DIR OUT")
    config, seed, output_dir, out = argv
    cfg = yaml.safe_load(Path(config).read_text())
    cfg["seed"] = int(seed)
    cfg["output_dir"] = output_dir
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(yaml.safe_dump(cfg, sort_keys=False))


if __name__ == "__main__":
    main(sys.argv[1:])
