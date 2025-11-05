import sys, pathlib
project_root = pathlib.Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from eval.evaluate_checkpoint import cli

if __name__ == "__main__":
    cli() 