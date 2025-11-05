import sys, pathlib, os
# Ensure project root is on PYTHONPATH when running as script
project_root = pathlib.Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from train.train_stealthgan import cli

if __name__ == "__main__":
    cli() 