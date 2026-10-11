#!/usr/bin/env python3
from sys import argv

from uhpo.pipeline import main


raise SystemExit(main(argv[1:], fixed_stage=4))
