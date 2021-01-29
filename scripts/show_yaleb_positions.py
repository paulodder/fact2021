import dotenv
import sys
from pathlib import Path

PROJECT_DIR = Path(dotenv.dotenv_values()["PROJECT_DIR"])
sys.path.insert(0, str(PROJECT_DIR / "src"))
from plots import plot_yaleb_lighting_positions

if __name__ == "__main__":
    plot_yaleb_lighting_positions("yaleb_lighting_positions")
