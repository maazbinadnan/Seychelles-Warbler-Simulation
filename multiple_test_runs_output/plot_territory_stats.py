from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILES = {
    "num_subordinates": BASE_DIR / "territory_subordinates_by_year_rulebased.png",
    "size": BASE_DIR / "territory_size_by_year_rulebased.png",
    "num_fledglings": BASE_DIR / "territory_fledglings_by_year_rulebased.png",
}


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


def load_territory_means(csv_path: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    return data.groupby("year", as_index=False)[["num_subordinates", "size", "num_fledglings"]].mean()


def main() -> None:
    runs = discover_runs()
    if not runs:
        raise FileNotFoundError("No run_<n> folders with territory.csv were found.")

    plot_specs = [
        ("num_subordinates", "Mean number of subordinates", OUTPUT_FILES["num_subordinates"]),
        ("size", "Mean territory size", OUTPUT_FILES["size"]),
        ("num_fledglings", "Mean number of fledglings", OUTPUT_FILES["num_fledglings"]),
    ]

    for metric, title, output_path in plot_specs:
        plt.figure(figsize=(9, 5))
        i=0
        for run_path in runs:
            i+=1
            yearly_means = load_territory_means(run_path / "territory.csv")
            plt.plot(
                yearly_means["year"],
                yearly_means[metric],
                marker="o",
                label=f"Run {i}",
            )

        plt.title(f"{title} per Year Across Runs")
        plt.xlabel("Year")
        plt.ylabel("Mean value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

        print(f"Saved chart to {output_path}")


if __name__ == "__main__":
    main()