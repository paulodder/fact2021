import re
import matplotlib.pyplot as plt
import dataloaders
from dotenv import dotenv_values, find_dotenv
from pathlib import Path

DOTENV = dotenv_values(find_dotenv())
PLOTS_DIR = Path(DOTENV["PROJECT_DIR"]) / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def plot_yaleb_lighting_positions(fname="yaleb_lighting_positions"):
    def get_yaleb_poses():
        root = dataloaders.DATA_DIR / "yaleb" / "CroppedYale"
        filepaths = root.glob("**/*")
        reg = "A(.\d+)E(.\d+)"
        poses = []
        for filepath in filepaths:
            s = re.search(reg, str(filepath))
            if s:
                poses.append(s.groups())
        return poses

    for group in set(get_yaleb_poses()):
        plt.scatter(int(group[0]), int(group[1]))
    plt.xlabel("azimuth")
    plt.ylabel("elevation")

    plt.savefig(PLOTS_DIR / (fname + ".png"), bbox_inches="tight")
