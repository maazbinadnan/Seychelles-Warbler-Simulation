from pathlib import Path

from main import run_simulation


OUTPUT_ROOT = Path("multiple_test_runs_output")
RUN_COUNT = 3


def main() -> None:
    OUTPUT_ROOT.mkdir(exist_ok=True)

    for run_number in range(1, RUN_COUNT + 1):
        run_folder = OUTPUT_ROOT / f"run__ruleBasedAI_{run_number}"
        run_folder.mkdir(parents=True, exist_ok=True)
        print(f"Starting run {run_number} -> {run_folder}")
        run_simulation(output_path=str(run_folder))
        print(f"Finished run {run_number}")


if __name__ == "__main__":
    main()
