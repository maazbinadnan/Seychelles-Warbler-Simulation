from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RUN_PATTERNS = {
    "genome": {
        "pattern": r"run_\d+",
        "output": BASE_DIR / "territory_count_by_year_genome.png",
    },
    "rulebased": {
        "pattern": r"1run_\d+",
        "output": BASE_DIR / "territory_count_by_year_rulebased.png",
    },
}


def discover_runs(pattern: str) -> list[Path]:
    runs = []
    for path in BASE_DIR.iterdir():
        if not path.is_dir():
            continue
        if not re.fullmatch(pattern, path.name):
            continue
        if (path / "territory.csv").exists():
            runs.append(path)
    return sorted(runs)


def load_territory_counts(csv_path: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    return data.groupby("year", as_index=False)["territory"].nunique()


def plot_group(group_name: str, pattern: str, output_path: Path) -> None:
    runs = discover_runs(pattern)
    if not runs:
        print(f"No runs found for {group_name}.")
        return

    plt.figure(figsize=(9, 5))

    for run_path in runs:
        yearly_counts = load_territory_counts(run_path / "territory.csv")
        plt.plot(
            yearly_counts["year"],
            yearly_counts["territory"],
            marker="o",
            label=run_path.name,
        )

    plt.title(f"Number of Territories per Year Across {group_name.title()} Runs")
    plt.xlabel("Year")
    plt.ylabel("Territory count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved chart to {output_path}")


def main() -> None:
    for group_name, config in RUN_PATTERNS.items():
        plot_group(group_name, config["pattern"], config["output"])


if __name__ == "__main__":
    main()