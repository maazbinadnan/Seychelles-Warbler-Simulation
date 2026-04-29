from main import run_simulation
from ax.api.client import Client
import pandas as pd

terr = pd.read_csv('territory.csv')
pop = pd.read_csv('population.py')
fit = pd.read_csv('fitness.py')

client = Client()

client.create_experiment(
    name="multi_objective",
    parameters=[
        {"name": "x1", "type": "range", "bounds": [0.0, 10.0]},
        {"name": "x2", "type": "range", "bounds": [0.0, 5.0]},
    ],
    objectives={
        "metric1": {"minimize": False},
        "metric2": {"minimize": True},
    },
)
n_trials = 30
for _ in range(n_trials):
    parameters, trial_index = client.get_next_trial()

    x1 = parameters["x1"]
    x2 = parameters["x2"]
    try:
        run_simulation(x1, x2)
        result = get_result()

        client.complete_trial(
            trial_index=trial_index,
            raw_data={
                "metric1": result[0],
                "metric2": result[1],
            },
        )
    except Exception as e:
        client.mark_trial_failed(trial_index=trial_index)
        raise


pareto = client.get_pareto_optimal_parameters()