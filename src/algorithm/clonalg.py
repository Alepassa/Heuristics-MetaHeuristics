import numpy as np
from typing import Tuple, List
from . import config

class CLONALG:
    """    
    L'algoritmo è ispirato al sistema immunitario:
    - Anticorpi = Soluzioni candidate
    - Affinità = Fitness della soluzione
    - Clonazione = Riproduzione delle soluzioni migliori
    - Mutazione = Esplorazione dello spazio delle soluzioni
    """

    def __init__(self, weights: np.ndarray, config: config):
        """
        Inizializzazione dell'algoritmo.
        """
        self.weights = weights
        self.n_bits = len(weights)
        self.config = config
        self.dtype = np.int8

    def _evaluate(self, solutions: np.ndarray) -> np.ndarray:
        """
        Calcola la fitness delle soluzioni.        
        Esempio:
            weights = [2, 3, 1]
            solution = [1, 0, 1] 
            fitness = 2x1 + 3x0 + 1x1 = 3
        """
        return np.dot(solutions.reshape(-1, self.n_bits), self.weights)

    def _select_parents(self, pop: np.ndarray, fit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Seleziona i migliori individui da clonare.

        Logica:
            - Calcola n_select = 15% della popolazione 
            - Ordina per fitness crescente
            - Prende gli ultimi n_select (saranno i migliori)
        """
        n_select = max(1, int(self.config.POPULATION_SIZE * self.config.SELECTION_RATIO))
        parent_indices = np.argsort(fit)[-n_select:]
        return parent_indices, fit[parent_indices]

    def _clone(self, pop: np.ndarray, parent_indices: np.ndarray) -> np.ndarray:
        """
        Clonazione intensificata che concentra più cloni sul migliore.

        Logica:
            1. Per ogni genitore calcola numero base cloni
            2. Se è il migliore: cloni = base × 2
            3. Altrimenti: cloni = base
            4. Replica ogni genitore n_clones volte
        """
        clones = []
        n_parents = len(parent_indices)
        
        for rank, idx in enumerate(parent_indices):
            base_clones = max(1, int(self.config.CLONE_FACTOR * self.config.POPULATION_SIZE / (n_parents - rank)))
            
            if rank == n_parents - 1:  
                n_clones = base_clones * 2
            else:
                n_clones = base_clones
            
            clones.extend([pop[idx]] * n_clones)
        
        if not clones:
            return np.array([], dtype=self.dtype)
        return np.array(clones, dtype=self.dtype)

    def _hypermutate(self, clones: np.ndarray, parent_fits: np.ndarray, parent_indices: np.ndarray) -> np.ndarray:
        """
        Mutazione inversamente proporzionale all'affinità.
        - Anticorpi migliori → mutazione bassa (ricerca locale)
        - Anticorpi peggiori → mutazione alta (esplorazione globale)
        
        Formula: mutation_rate = e^(-DECAY_RATE x fitness_normalizzata)
        """
        if clones.size == 0:
            return clones

        min_fit, max_fit = np.min(parent_fits), np.max(parent_fits)
        if max_fit == min_fit:
            norm_fits = np.ones_like(parent_fits)
        else:
            norm_fits = (parent_fits - min_fit) / (max_fit - min_fit)

        mutation_probs_per_parent = np.exp(-self.config.MUTATION_DECAY_RATE * norm_fits)
        
        prob_matrix_list = []
        n_parents = len(parent_indices)
        
        for rank, _ in enumerate(parent_indices):
            base_clones = max(1, int(self.config.CLONE_FACTOR * self.config.POPULATION_SIZE / (n_parents - rank)))
            
            if rank == n_parents - 1:  
                n_clones = base_clones * 2  
            else:
                n_clones = base_clones
            
            parent_mutation_prob = mutation_probs_per_parent[rank]
            prob_matrix_list.append(np.full((n_clones, self.n_bits), parent_mutation_prob))
        
        if not prob_matrix_list:
            return clones

        prob_matrix = np.vstack(prob_matrix_list)

        random_matrix = np.random.random(clones.shape)
        flip_mask = random_matrix < prob_matrix
        
        mutated_clones = clones.copy()
        mutated_clones[flip_mask] ^= 1
        
        return mutated_clones

    def _introduce_diversity(self, pop: np.ndarray, fit: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Sostituisce una piccola frazione dei peggiori individui con nuovi casuali per mantenere diversità e prevenire convergenza prematura.

        Logica:
            1. Calcola 14% della popolazione da sostituire
            2. Identifica i peggiori individui
            3. Sostituisce con soluzioni casuali
            4. Ricalcola fitness dei nuovi individui
        """
        n_random = int(self.config.POPULATION_SIZE * self.config.RANDOM_REPLACEMENT_RATIO)
        if n_random == 0:
            return pop, fit, 0
        
        worst_indices = np.argsort(fit)[:n_random]
        
        new_individuals = np.random.randint(0, 2, size=(n_random, self.n_bits), dtype=self.dtype)
        new_fits = self._evaluate(new_individuals)
        
        pop[worst_indices] = new_individuals
        fit[worst_indices] = new_fits
        
        return pop, fit, n_random

    def run(self, seed: int, optimal_fitness: float) -> Tuple[np.ndarray, float, List[float], int, int]:
        """
        1. Inizializzazione popolazione casuale
        2. LOOP fino a MAX_EVALUATIONS / OPTIMAL FITNESS:
           a. Selezione genitori
           b. Clonazione
           c. Mutazione
           d. Sopravvivenza
           e. Diversificazione
        3. Ritorna migliore soluzione trovata
        """
        np.random.seed(seed)
        
        pop = np.random.randint(0, 2, (self.config.POPULATION_SIZE, self.n_bits), dtype=self.dtype)
        fit = self._evaluate(pop)
        evals = self.config.POPULATION_SIZE

        best_fit_so_far = np.max(fit)
        best_sol_so_far = pop[np.argmax(fit)].copy()
        hist, gen, best_gen, best_evals = [best_fit_so_far], 0, 0, evals

        while evals < self.config.MAX_EVALUATIONS:
            gen += 1
            
            # 1. Seleziona il top 15% della popolazione per riproduzione
            parent_indices, parent_fits = self._select_parents(pop, fit)

            # 2. Genera cloni degli individui selezionati, il migliore ottiene doppi cloni per accellerare la convergenza
            clones = self._clone(pop, parent_indices)

            # 3.  Muta i cloni con probabilità inversamente proporzionale alla fitness
            # Principio: soluzioni migliori → mutazione minore (ricerca locale)
            #           soluzioni peggiori → mutazione maggiore (esplorazione globale)
            mutated_clones = self._hypermutate(clones, parent_fits, parent_indices)
            if mutated_clones.size == 0:
                continue  
            mutated_fits = self._evaluate(mutated_clones)
            evals += len(mutated_clones)

            # 4. Competizione globale tra popolazione attuale e cloni
            combined_pop = np.vstack((pop, mutated_clones))
            combined_fit = np.concatenate((fit, mutated_fits))
            # Seleziona i migliori N individui per formare la nuova generazione.
            top_indices = np.argsort(combined_fit)[-self.config.POPULATION_SIZE:]
            pop = combined_pop[top_indices]
            fit = combined_fit[top_indices]

            # 5. Sostituisce ~14% dei peggiori individui con nuovi casuali per diversità
            pop, fit, d_evals = self._introduce_diversity(pop, fit)
            evals += d_evals
            
            # Dato che pop è ordinata per fitness, il migliore è l'ultimo elemento
            current_best_fit = fit[-1]
            if current_best_fit > best_fit_so_far:
                best_fit_so_far = current_best_fit
                best_sol_so_far = pop[-1].copy()
                best_gen = gen
                best_evals = evals
            hist.append(best_fit_so_far)

            # Condizione di uscita
            if optimal_fitness is not None and np.isclose(best_fit_so_far, optimal_fitness):
                break

        return best_sol_so_far, best_fit_so_far, hist, best_gen, best_evals
