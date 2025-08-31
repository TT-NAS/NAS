import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, List
from codec import Chromosome
from .surrogate import SurrogateModel 
import random
#from .surrogate import SurrogateModel
 
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, List

class Evaluator():
  def __init__(self, codification: Literal["binary", "real"]) -> None:
    """
    Clase para evaluar la población de individuos.
    """
    self.surrogate_model = SurrogateModel(model_path = r"./sustituto/xgboost_model.json")
    self.codification = codification

  def get_params(self, population: np.ndarray) -> np.ndarray:
    """
    Obtener los parámetros de la población de individuos.
    Args:
        population (np.ndarray): Población de individuos a evaluar.
    Returns:
        np.ndarray: Parámetros de los individuos.
    """
    params = np.zeros((len(population), 1))
    for i in range(len(population)):
        model = Chromosome(chromosome=list(population[i]))
        unet = model.get_unet()
        trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        params[i] = trainable_params
    return params

  def evaluate_population(self, population):
    """
    Evaluar la población de individuos.
    Args:
        population (np.ndarray): Población de individuos a evaluar.
    Returns:
        np.ndarray: Aptitudes de los individuos.
    """
    # Primer objetivo (Predicción de IoU)
    fitness_1 = self.surrogate_model.predict(population)
    fitness_1[fitness_1 < 0] = - np.inf
    fitness_1[fitness_1 > 1] = - np.inf
    fitness_1 = 1 - fitness_1  # Invertir la función objetivo para minimizar
    
    # Segundo objetivo (Número de parámetros)
    fitness_2 = self.get_params(population)
    
    # Matriz de fitness
    fitness = np.column_stack((fitness_1, fitness_2))
    return fitness


class TwoObjectiveOptimizer:
    def __init__(self) -> None:
        """
        Clase que generaliza la optimización multiobjetivo y sus operadores.
        """
        self.evaluator = Evaluator(codification="real")
        self.population = []
        self.fitness = []
        self.fitness_history = []
        self.pareto_fronts = []
        self.pareto_history = []

    def initialize_population(self, n_pop: int) -> np.ndarray:
        """
        Inicializar la población de individuos.
        Args:
            n_pop (int): Número de individuos en la población.
        """
        self.population = [Chromosome() for _ in range(n_pop)]
        self.population = np.array([chromosome.get_real() for chromosome in self.population])
        
    def get_pareto_front(self, fitness: np.ndarray) -> np.ndarray:
        """
        Obtiene el frente de Pareto de la población dada para dos objetivos.
        Devuelve índices de la población que forman el frente de Pareto.
        """
        sorted_idxs = np.argsort(fitness[:, 0])
        sorted_fitness = fitness[sorted_idxs]

        pareto_front_sorted = []
        obj2_min = np.inf
        for i in range(len(sorted_fitness)):
            if sorted_fitness[i, 1] < obj2_min:
                pareto_front_sorted.append(sorted_idxs[i])
                obj2_min = sorted_fitness[i, 1]

        return np.array(pareto_front_sorted)

    def ranked_pareto_fronts(self, fitness: np.ndarray) -> List[np.ndarray]:
        """
        Obtiene los frentes de Pareto ordenados por dominancia (índices globales).
        Devuelve una lista de arrays, cada uno representa los índices de un frente de pareto
        """
        fronts = []
        remaining = np.arange(len(fitness))
        while len(remaining) > 0:
            # Calcular frente en los individuos restantes
            front_rel = self.get_pareto_front(fitness[remaining])
            # Convertir a índices globales
            front_global = remaining[front_rel]
            # Guardar frente y actualizar los individuos restantes
            fronts.append(front_global)
            remaining = np.setdiff1d(remaining, front_global)
        return fronts
    
    def get_crowding_distance(self, front: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """
        Calcula la distancia de hacinamiento para un frente de Pareto dado.
        Recibe los índices del frente y la matriz de fitness
        
        Devuelve un array de crowding distances para los individuos en el frente.
        """
        if len(front) == 0:
            return np.array([])

        distances = np.zeros(len(front))
        num_objectives = fitness.shape[1]

        for i in range(num_objectives):
            # Ordenar el frente por el objetivo i
            sorted_indices = np.argsort(fitness[front, i])
            sorted_fitness = fitness[front][sorted_indices, i]

            f_min, f_max = sorted_fitness[0], sorted_fitness[-1]
            if f_max == f_min:
                continue  # Evitar división por cero si todos son iguales en este objetivo

            # Extremos reciben infinito
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            # Calcular crowding para los puntos intermedios
            for j in range(1, len(front) - 1):
                if not np.isinf(distances[sorted_indices[j]]):  # No sobrescribir extremos
                    distances[sorted_indices[j]] += (
                        (sorted_fitness[j + 1] - sorted_fitness[j - 1]) / (f_max - f_min)
                    )
        return distances

    def plot_pareto_fronts(self, fitness: np.ndarray, fronts) -> None:
        """
        Plotea los frentes de Pareto de la población dada.
        """
        plt.figure(figsize=(10, 6))
        for i, front in enumerate(fronts):
            plt.scatter(fitness[front, 0], fitness[front, 1], label=f'Front {i+1}')

        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title('Pareto Fronts')
        plt.legend()
        plt.grid()
        plt.show()

class DiferentialEvolution(TwoObjectiveOptimizer):
    def __init__(self,  base: Literal["random", "current"] = "random", 
                n_differences: int = 1, crossover: Literal["bin", "exp"] = "bin") -> None:
        """
        Clase para el algoritmo de evolución diferencial.

        Args:
            base (Literal["random", "current"]): Estrategia base para la evolución diferencial.
            n_differences (int): Número de vectores diferencia a considerar, puede ser 1 o 2.
            crossover (Literal["bin", "exp"]): Tipo de cruce a utilizar, puede ser binario o exponencial.
        """
        super().__init__()
        assert base in ["random", "current"], "base must be 'random' or 'current'"
        assert n_differences == 1 or n_differences == 2, "n_differences must be 1 or 2"
        assert crossover in ["bin", "exp"], "crossover must be 'bin' or 'exp'"

        self.variant_dict = {
            "base": base,
            "n_differences": n_differences,
            "crossover": crossover
        }

    def start(self, n_pop: int = 100, max_gen: int = 1000, mutation_rate: float = 0.5, crossover_rate: float = 0.7) -> None:
        """
        Iniciar el algoritmo de evolución diferencial.

        Args:
            n_pop (int): Número de individuos en la población.
            max_gen (int): Número máximo de generaciones.
            mutation_rate (float): Tasa de mutación.
            crossover_rate (float): Tasa de cruce.
        """
        self.initialize_population(n_pop)
        for gen in range(max_gen):
            print(f"Generación {gen+1}/{max_gen}")
            offspring = []
            for target in range(len(self.population)):
                # Selección del vector base
                if self.variant_dict["base"] == "random":
                    r1 = np.random.randint(0, len(self.population))
                else: # current
                    r1 = target
                # Selección de los vectores diferencia
                r2, r3 = random.sample([j for j in range(len(self.population)) if j != target and j != r1], 2)
                if self.variant_dict["n_differences"] == 1:
                    trial = self.population[r1] + mutation_rate * (self.population[r2] - self.population[r3])
                elif self.variant_dict["n_differences"] == 2:
                    r4, r5 = random.sample([j for j in range(len(self.population)) if j != target and j != r1 and j != r2 and j != r3], 2)
                    trial = self.population[r1] + mutation_rate * (
                            (self.population[r2] - self.population[r3]) + 
                            (self.population[r4] - self.population[r5]))

                # Crossover
                fixed_idx = np.random.randint(0, len(self.population[0]))
                fixed_value = self.population[target][fixed_idx]
                if self.variant_dict["crossover"] == "bin":
                    crossover_mask = np.random.rand(len(self.population[0])) < crossover_rate
                    trial[crossover_mask] = self.population[target][crossover_mask]
                elif self.variant_dict["crossover"] == "exp":
                    start = np.random.randint(0, len(self.population[0]))
                    j = start
                    while True:
                        trial[j] = self.population[target][j]
                        j = (j + 1) % len(self.population[0])
                        if j == start or np.random.rand() > crossover_rate:
                            break
                trial[fixed_idx] = fixed_value # Valor fijo de target
                # Reparar trial
                trial = np.clip(trial, 0, 1)
                offspring.append(trial)
                
            # Mu + Lambda
            print(f"Evaluando población")
            self.offspring = np.array(offspring)
            self.offspring = np.vstack((self.population, self.offspring))
            
            self.fitness = self.evaluator.evaluate_population(self.offspring)
            self.fitness_history.append(self.fitness)
            fronts = self.ranked_pareto_fronts(self.fitness)
            self.pareto_history.append(fronts)
            self.pareto_fronts = fronts
            crowding_distances = [self.get_crowding_distance(front, self.fitness) for front in fronts]

            # Selección de la nueva población
            new_population = []
            for i, front in enumerate(fronts):
                # Selección de individuos de la misma frente
                front_individuals = self.offspring[front]
                front_crowding_distances = crowding_distances[i]
                # Selecciónar todo el frente, si sobrepasa el tamaño de la población, seleccionar por crowding distance
                if len(new_population) + len(front) <= n_pop:
                    new_population.extend(front_individuals)
                else:
                    # Ordenar por crowding distance y seleccionar los mejores
                    sorted_indices = np.argsort(front_crowding_distances)[::-1]
                    selected_indices = sorted_indices[:n_pop - len(new_population)]
                    new_population.extend(front_individuals[selected_indices])
            self.population = np.array(new_population)

def main():
    # Prueba evolución diferencial
    differential_evolution = DiferentialEvolution()
    differential_evolution.start(100, 100)
    for fitness, pareto_fronts in zip(differential_evolution.fitness_history, differential_evolution.pareto_history):
        print(f"Fitness: {fitness}")
        print(f"Pareto fronts: {pareto_fronts}")
        differential_evolution.plot_pareto_fronts(fitness, pareto_fronts)
        
    # Imprimir los individuos del frente 1
    for individual in differential_evolution.pareto_fronts[0]:
        print(f"Individual: {differential_evolution.population[individual]}")
    
    
if __name__ == "__main__":
    main()
