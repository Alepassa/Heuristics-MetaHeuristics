import optuna
import numpy as np
import os
import datetime
from algorithm import clonalg
from algorithm import config
from utils import input_reader

N_RUNS_PER_TRIAL = 10
DATA_PATH = "../data/"

def load_instance(instance_name: str) -> tuple[np.ndarray, int]:
    """
    Carica un'istanza del problema.
    """
    filename_csv = f"{instance_name}.csv"
    filepath = os.path.join(DATA_PATH, filename_csv)
    instance_loader = input_reader.WBSInstance(filename_csv)
    instance_loader.load_from_file(filepath)
    weights = instance_loader.get_weights_array()
    optimal_fitness = instance_loader.max_fitness
    return weights, optimal_fitness


def objective_multiobj(trial: optuna.Trial) -> tuple[float, float]:
    """
    Esegue N_RUNS_PER_TRIAL per ogni configurazione di parametri
    e restituisce metriche aggregate per una valutazione più robusta.
    """
    print(f"Inizio Trial {trial.number}/{N_TRIALS_TOTAL} (esecuzione di {N_RUNS_PER_TRIAL} run)...")

    try:
        weights, optimal_fitness = load_instance("instance_8")
    except FileNotFoundError as e:
        print(e)
        raise optuna.exceptions.TrialPruned()

    cfg = config.Config()
    
    # parametri da ottimizzare
    # inizialmente la prima ricerca è stata condotta con parametri più ampi
    # la seconda viene condotta con il range vicino tra le soluzioni ottime trovate dalla prima ricerca
    cfg.POPULATION_SIZE = trial.suggest_int("POPULATION_SIZE", 35, 50, step=5)
    cfg.SELECTION_RATIO = trial.suggest_float("SELECTION_RATIO", 0.08, 0.2)
    cfg.CLONE_FACTOR = trial.suggest_float("CLONE_FACTOR", 0.4, 0.8)
    cfg.MUTATION_DECAY_RATE = trial.suggest_float("MUTATION_DECAY_RATE", 7.5, 8.5)
    cfg.RANDOM_REPLACEMENT_RATIO = trial.suggest_float("RANDOM_REPLACEMENT_RATIO", 0.1, 0.2)
    cfg.MAX_EVALUATIONS = 150000

    all_gaps = []
    all_evals_on_success = []
    successful_runs_count = 0

    for i in range(N_RUNS_PER_TRIAL):
        algo = clonalg.CLONALG(weights, cfg)
        seed = np.random.randint(0, 10000) + i 
        _, best_fit, _, _, best_evals = algo.run(seed=seed, optimal_fitness=optimal_fitness)
        
        gap = max(0.0, optimal_fitness - best_fit)
        all_gaps.append(gap)
        
        if np.isclose(gap, 0.0):
            successful_runs_count += 1
            all_evals_on_success.append(best_evals)

    mean_gap = np.mean(all_gaps)
    success_rate = successful_runs_count / N_RUNS_PER_TRIAL


    if all_evals_on_success:
        mean_evals = np.mean(all_evals_on_success)
    else:
        mean_evals = cfg.MAX_EVALUATIONS

    trial.set_user_attr("success_rate", success_rate)
    trial.set_user_attr("mean_gap", mean_gap)
    return mean_gap, mean_evals


if __name__ == "__main__":
    N_TRIALS_TOTAL = 85 # Numero totale di configurazioni da testare
    TIMEOUT_SECONDS = 7200 # Limite di tempo

    RESULTS_PATH = "results/"
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(objective_multiobj, n_trials=N_TRIALS_TOTAL, timeout=TIMEOUT_SECONDS)
    
    if optuna.visualization.is_available():
        figure_filename = os.path.join(RESULTS_PATH, "robust_tuning_pareto_front.html")
        fig = optuna.visualization.plot_pareto_front(
            study, 
            target_names=["Gap Medio dalla Soluzione Ottima", "Valutazioni Medie (su successo)"]
        )
        fig.write_html(figure_filename)
    
