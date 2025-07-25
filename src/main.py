import numpy as np
import time
import argparse
import multiprocessing
from typing import List, Dict, Any
from algorithm.config import Config
from utils.input_reader import WBSInstance, load_all_instances
from algorithm.clonalg import CLONALG
from utils.visualization import ResultsVisualizer


def run_single_clinalg_task(args):
    """Esegue una singola run."""
    weights_np, config, seed, optimal_fitness, run_id = args

    algorithm = CLONALG(weights_np, config)
    _, best_fitness, convergence, iterations_to_best, evaluations_to_best = algorithm.run(
        seed=seed, optimal_fitness=optimal_fitness
    )

    return best_fitness, iterations_to_best, convergence, evaluations_to_best

def run_experiments_for_instance(instance: WBSInstance, config: Config) -> Dict[str, Any]:
    """Esegue 100 run per una singola istanza in parallelo."""

    print(f"Elaborando 100 run dell'istanza : {instance.filename}")
    
    start_time = time.time()
    weights_np = np.array(instance.weights)
    
    run_args = [
        (weights_np, config, config.RANDOM_SEED_BASE + i, instance.max_fitness, i + 1)
        for i in range(config.NUM_RUNS)
    ]
    
    with multiprocessing.Pool() as pool:
        all_results = pool.map(run_single_clinalg_task, run_args)
    
    all_best_fitnesses = [res[0] for res in all_results]
    all_iterations_to_best = [res[1] for res in all_results]
    all_convergence_histories = [res[2] for res in all_results]
    all_evaluations_to_best = [res[3] for res in all_results]
    successful_runs = sum(1 for bf in all_best_fitnesses if np.isclose(bf, instance.max_fitness))
    results = {
        'name': instance.filename.replace('.csv', ''),
        'best': np.max(all_best_fitnesses),
        'mean': np.mean(all_best_fitnesses),
        'std': np.std(all_best_fitnesses),
        'mean_iterations_to_best': np.mean(all_iterations_to_best),
        'mean_evaluations_to_best': np.mean(all_evaluations_to_best),
        'convergence_histories': all_convergence_histories,
        'execution_time_seconds': time.time() - start_time,
        'optimal': instance.max_fitness,
        'success_rate': (successful_runs / config.NUM_RUNS) * 100,
        'string_length': instance.string_length,
        'all_evaluations_to_best': all_evaluations_to_best,
        'all_iterations_to_best': all_iterations_to_best,
    }
    
    print(f"  Best: {results['best']:.2f}, Mean: {results['mean']:.2f}, "
          f"Success: {results['success_rate']:.1f}%, Time: {results['execution_time_seconds']:.1f}s")
    
    return results


def generate_outputs(all_results: List[Dict[str, Any]], config: Config):
    """Genera grafici e tabelle dai risultati."""
    visualizer = ResultsVisualizer(config)
    results_summary = {res['name']: res for res in all_results}
    
    for res in all_results:
        visualizer.plot_convergence(res['convergence_histories'], res['name'], res['optimal'])
        visualizer.plot_detailed_run_convergence(res['convergence_histories'], res['name'], res['optimal'])
    
    visualizer.plot_results_summary(results_summary)
    
    print(f"\n{'Istanza':<12} {'Success%':<8} {'Media':<8} {'Std':<6} {'Optimal':<8} {'Iter':<6} {'Eval':<8} {'Tempo(s)':<9}")
    print("-" * 75)
    for res in all_results:
        print(f"{res['name']:<12} {res['success_rate']:<8.1f} "
              f"{res['mean']:<8.2f} {res['std']:<6.2f} "
              f"{res['optimal']:<8.0f} "
              f"{res['mean_iterations_to_best']:<6.1f} {res['mean_evaluations_to_best']:<8.0f} "
              f"{res['execution_time_seconds']:<9.2f}")

def run_all_experiments(config: Config):
    """Esegue esperimenti su tutte le istanze."""
    instances = load_all_instances(config.DATA_PATH)
    if not instances:
        print(f"Nessuna istanza trovata in: {config.DATA_PATH}")
        return
    
    all_results = []
    for instance in instances:
        result = run_experiments_for_instance(instance, config)
        if result:
            all_results.append(result)
    
    if all_results:
        generate_outputs(all_results, config)


def run_single_instance(identifier: str, config: Config):
    """Esegue esperimenti su una singola istanza."""
    instances = load_all_instances(config.DATA_PATH)
    target_instance = next((inst for inst in instances if identifier in inst.filename), None)
    
    if not target_instance:
        print(f"Istanza '{identifier}' non trovata")
        return
    
    result = run_experiments_for_instance(target_instance, config)
    if result:
        generate_outputs([result], config)


def main():
    parser = argparse.ArgumentParser(description="Esperimenti")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--all', action='store_true', help='Tutte le istanze')
    group.add_argument('--instance', type=str, help='Singola istanza')
    
    args = parser.parse_args()
    config = Config()

    if args.instance:
        run_single_instance(args.instance, config)
    else:  # -all 
        run_all_experiments(config)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()