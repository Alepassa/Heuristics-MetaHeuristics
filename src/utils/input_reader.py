import numpy as np
from typing import List
import os

class WBSInstance:
    """
    Classe che rappresenta una singola istanza del problema.

    Formato file CSV atteso:
    Max Fitness,<valore>
    String Length,<valore>
    Min Weight,<valore>
    Max Weight,<valore>
    Genes,<bit1>,<bit2>,...,<bitN>
    Weights,<peso1>,<peso2>,...,<pesoN>
    """
    

    def __init__(self, filename: str):
        self.filename = filename
        self.max_fitness = 0
        self.string_length = 0
        self.min_weight = 0
        self.max_weight = 0
        self.optimal_genes = []
        self.weights = []
        
    def load_from_file(self, filepath: str) -> bool:
        """Carica i dati da ogni istanza CSV """
        try:
            with open(filepath, 'r') as file:
                lines = file.readlines()
                
            for line in lines:
                line = line.strip()
                if line.startswith('Max Fitness,'):
                    self.max_fitness = int(line.split(',')[1])
                elif line.startswith('String Length,'):
                    self.string_length = int(line.split(',')[1])
                elif line.startswith('Min Weight,'):
                    self.min_weight = float(line.split(',')[1])
                elif line.startswith('Max Weight,'):
                    self.max_weight = float(line.split(',')[1])
                elif line.startswith('Genes,'):
                    genes_str = line.split(',')[1:]
                    self.optimal_genes = [int(g) for g in genes_str if g.strip()]
                elif line.startswith('Weights,'):
                    weights_str = line.split(',')[1:]
                    self.weights = [float(w) for w in weights_str if w.strip()]
            
            return True
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return False
    
    def validate(self) -> bool:
        "Valida la consistenza dei dati caricati."
        calculated_fitness = sum(g * w for g, w in zip(self.optimal_genes, self.weights))
        if abs(calculated_fitness - self.max_fitness) > 1e-6:
            print(f"Warning: calculated fitness ({calculated_fitness}) != reported fitness ({self.max_fitness})")
        return True
    
    def get_weights_array(self) -> np.ndarray:
        """Restituisce i pesi come array NumPy per efficienza computazionale."""
        return np.array(self.weights)
    
    def get_optimal_solution(self) -> np.ndarray:
        """Restituisce la soluzione ottimale come array NumPy."""
        return np.array(self.optimal_genes)

def load_all_instances(data_path: str) -> List[WBSInstance]:
    """Carica tutte le istanze WBS da una directory."""
    instances = []

    if not os.path.exists(data_path):
        print(f"Data path {data_path} not found")
        return instances
    
    for filename in sorted(os.listdir(data_path)):
        if filename.endswith('.csv') and filename.startswith('instance_'):
            filepath = os.path.join(data_path, filename)
            instance = WBSInstance(filename)
            
            if instance.load_from_file(filepath):
                if instance.validate():
                    instances.append(instance)
                    print(f"Loaded {filename}: length={instance.string_length}, optimal={instance.max_fitness}")
            else:
                print(f"Failed to load {filename}")
                
    print(f"\n")
    return instances
