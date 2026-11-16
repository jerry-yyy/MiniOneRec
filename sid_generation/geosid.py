"""Thin entry point for optional GeoSID-style RQ-VAE training and diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'export', 'diagnose'])
    argv = list(sys.argv[1:] if argv is None else argv)
    # Dispatch the first positional argument only. Let the selected command
    # handle its own --help rather than consuming it in the outer parser.
    args = parser.parse_args(argv[:1])
    rest = argv[1:]
    if args.action == 'train':
        from rq.rqvae import main as train
        train(rest)
    else:
        from rq.geosid_tools import main as tools_main
        tools_main([args.action] + rest)


if __name__ == '__main__':
    main()
