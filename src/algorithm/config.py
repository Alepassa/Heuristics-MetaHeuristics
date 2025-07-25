class Config:

    NUM_RUNS = 100
    RANDOM_SEED_BASE = 43
    MAX_EVALUATIONS = 150000
    

    # parametri trovati tramite tuning
    POPULATION_SIZE = 35 # dimensione della popolazione di anticorpi
    SELECTION_RATIO = 0.15  # % di popolazione selezionata per la clonazione
    CLONE_FACTOR = 0.45   # fattore moltiplicativo per il numero di cloni generati
    MUTATION_DECAY_RATE = 7.84 # tasso di decadimento per la mutazione adattiva
    RANDOM_REPLACEMENT_RATIO = 0.14 # % di popolazione sostituita casualmente

    FIGURE_SIZE = (12, 8)
    DPI = 300
    DATA_PATH = "data/" 
    RESULTS_PATH = "results/"
    DETAILED_PLOTS_PATH = "results/detailed_runs/"