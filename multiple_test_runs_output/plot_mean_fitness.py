from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "mean_fitness_comparison_ruleBased.png"


def discover_runs() -> list[Path]:
    runs = []
    for path in BASE_DIR.iterdir():
        if not path.is_dir():
            continue
        if not re.fullmatch(r"1run_\d+", path.name):
            continue
        if (path / "fitness.csv").exists():
            runs.append(path)
    return sorted(runs, key=lambda path: int(path.name.split("_")[1]))


def load_mean_fitness(csv_path: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    return data.groupby("year", as_index=False)["fitness"].mean()


def main() -> None:
    runs = discover_runs()
    if not runs:
        raise FileNotFoundError("No run_<n> folders with fitness.csv were found.")

    plt.figure(figsize=(9, 5))

    i = 0
    for run_path in runs:
        i += 1
        mean_by_year = load_mean_fitness(run_path / "fitness.csv")
        plt.plot(mean_by_year["year"], mean_by_year["fitness"], marker="o", label=f"Run {i}")

    plt.title("Mean Fitness per Year Across Runs")
    plt.xlabel("Year")
    plt.ylabel("Mean Fitness")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.close()

    print(f"Saved chart to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()