#!/usr/bin/env python3
"""InternData historical 30Hz data-gate entrypoint."""

import sys

from check_interndata_a1_gate import main


if __name__ == "__main__":
    if "--target-control-hz" not in sys.argv:
        sys.argv.extend(["--target-control-hz", "30"])
    main()
