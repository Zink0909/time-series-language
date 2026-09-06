#!/usr/bin/env python3
"""Compatibility entry point for the per-domain MiGAS report.

The implementation lives in migas_finetune.py so training and normalization fixes cannot diverge
between two copied runners.
"""
import sys

from migas_finetune import main


if __name__ == "__main__":
    if "--per-domain" not in sys.argv:
        sys.argv.append("--per-domain")
    main()
