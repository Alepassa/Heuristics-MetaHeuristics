import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import os
from algorithm.config import Config

class ResultsVisualizer:

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.results_path = self.config.RESULTS_PATH
        self.detailed_plots_path = self.config.DETAILED_PLOTS_PATH
        
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.detailed_plots_path, exist_ok=True)
    
    def plot_convergence(self, convergence_histories: List[List[float]], instance_name: str, optimal_fitness: float = None) -> None:
        """
        Disegna il grafico di convergenza media per una determinata istanza.
        """
        
        plt.figure(figsize=self.config.FIGURE_SIZE, dpi=self.config.DPI)
        
        max_length = max(len(h) for h in convergence_histories if h)
        padded_histories = [h + [h[-1]] * (max_length - len(h)) for h in convergence_histories if h]
        avg_convergence = np.mean(padded_histories, axis=0)

        plt.plot(avg_convergence, color='yellow', linewidth=2, label='Average Convergence')
        plt.axhline(y=optimal_fitness, color='red', linestyle='--', linewidth=2, label=f'Optimal: {int(optimal_fitness)}')
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.title(f'Average Convergence Analysis - {instance_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        filename = os.path.join(self.results_path, f'convergence_{instance_name}.png')
        plt.savefig(filename, dpi=self.config.DPI, bbox_inches='tight')
        plt.close()
        
    
    def plot_detailed_run_convergence(self, convergence_histories: List[List[float]], instance_name: str, optimal_fitness: float = None) -> None:
        """
        Disegna un grafico dettagliato con le curve di convergenza di 5 run individuali
        """
        plt.figure(figsize=self.config.FIGURE_SIZE, dpi=self.config.DPI)
        
        num_runs_to_plot = min(len(convergence_histories), 5)
        
        for i, history in enumerate(convergence_histories[:num_runs_to_plot]):
            plt.plot(history, alpha=0.8, linewidth=1.0, label=f'Run {i+1}' if num_runs_to_plot <= 5 else None)

        plt.axhline(y=optimal_fitness, color='red', linestyle='--',  linewidth=2, label=f'Optimal: {int(optimal_fitness)}') 
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.title(f'Detailed Convergence of {num_runs_to_plot} Runs - {instance_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        filename = os.path.join(self.detailed_plots_path, f'detailed_convergence_{instance_name}.png')
        plt.savefig(filename, dpi=self.config.DPI, bbox_inches='tight')
        plt.close()
        

    def plot_results_summary(self, results: Dict[str, Dict]) -> None:
        """Riepilogo risultati dell'esperimento con 3 grafici."""
        if not results:
            return
        
        instances = list(results.keys())
        best_values = [results[inst]['best'] for inst in instances]
        mean_values = [results[inst]['mean'] for inst in instances]
        optimal_values = [results[inst]['optimal'] for inst in instances]
        string_lengths = [results[inst]['string_length'] for inst in instances]
        execution_times = [results[inst]['execution_time_seconds'] for inst in instances]
        mean_evaluations = [results[inst]['mean_evaluations_to_best'] for inst in instances]

        # Crea figura con 3 subplot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8), dpi=self.config.DPI)
        
        # Grafico 1: Performance
        x = np.arange(len(instances))
        ax1.bar(x - 0.2, best_values, 0.4, label='Best', alpha=0.8)
        ax1.bar(x + 0.2, mean_values, 0.4, label='Mean', alpha=0.8)
        ax1.plot(x, optimal_values, 'ro-', label='Optimal')
        ax1.set_xlabel('Instance')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(instances, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Grafico 2: Scalabilità con Regressione Lineare
        ax2.scatter(string_lengths, execution_times, alpha=0.8, color='#2ca02c', s=60, label='Execution Time')
        m, b = np.polyfit(string_lengths, execution_times, 1)
        x_trend = np.linspace(min(string_lengths), max(string_lengths), 100)
        y_trend = m * x_trend + b
        ax2.plot(x_trend, y_trend, 'r--', linewidth=2, label=f'Trend: y = {m:.2e}x + {b:.2f}')
        ax2.set_xlabel('String Length (Problem Complexity)')
        ax2.set_ylabel('Execution Time (s)')
        ax2.set_title('Scalability Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Grafico 3:  Valutazioni medie al variare della complessità del problema
        sorted_data = sorted(zip(string_lengths, mean_evaluations, instances))
        sorted_lengths, sorted_evals, sorted_instances = zip(*sorted_data)
        ax3.plot(sorted_lengths, sorted_evals, 'o-', color='#9C27B0', linewidth=2.5,  markersize=8, alpha=0.8, label='Avg Evaluations to Best')
        ax3.fill_between(sorted_lengths, sorted_evals, alpha=0.3, color='#E1BEE7')
        max_evals = 150000  # MAX_EVAL
        ax3.axhline(y=max_evals, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'Budget Limit ({max_evals:,})')
        
        # Anotazioni per ogni punto
        for length, evals, instance in zip(sorted_lengths, sorted_evals, sorted_instances):
            ax3.annotate(instance, (length, evals), xytext=(0, 10), 
                        textcoords='offset points', ha='center', fontsize=9, 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax3.set_xlabel('String Length (Problem Complexity)')
        ax3.set_ylabel('Average Evaluations to Best')
        ax3.set_title('Efficiency Analysis')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, 'results_summary.png'),  dpi=self.config.DPI, bbox_inches='tight')
        plt.close()
        