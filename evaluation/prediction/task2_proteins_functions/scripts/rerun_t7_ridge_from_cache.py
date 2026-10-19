#!/usr/bin/env python3


import sys
from evaluate_t7 import main

if __name__ == "__main__":
    main_arguments = sys.argv[1:]
    if "--run-mode" in main_arguments:
        raise ValueError("This entry point always uses ridge_only.")
    sys.argv.extend(["--run-mode", "ridge_only"])
    main()
