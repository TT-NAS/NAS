# Implementación de algoritmos de búsqueda multiobjetivo
# Implementa NSGA II, con los objetivos: Predcción del modelo sustituto (1-IoU) y número de parámetros del modelo
# To Do:
# - Generalizar la selección Mu + Lambda
# - Implementar otros algoritmos poblacionalesb (Genético primero)
# - Comparación con PYMOO

import random
import sys
from typing import Literal, List

import matplotlib.pyplot as plt
import numpy as np

from codec import Chromosome
from .surrogate import SurrogateModel 

class Evaluator():
  def __init__(self, codification: Literal["binary", "real"]) -> None:
    """
    Clase del evaluador multiobjetivo.
    
    Parametros:
    ===========
        - codifiication (str): Tipo de codificación, puede ser "binary" o "real".
    """
    self.surrogate_model = SurrogateModel(model_path = r"./sustituto/xgboost_model.json")
    self.codification = codification

  def get_params(self, population: np.ndarray) -> np.ndarray:
    """
    Obtener los parámetros de la población de individuos.
    
    Parametros:
    ===========
        - population (np.ndarray): Población de individuos a evaluar (Arreglo de cromosomas).
    Retorna:
    ===========
        - np.ndarray: Arreglo con los parámetros de los individuos.
    """
    # Inicializar el array
    params = np.zeros((len(population), 1))
    
    for i in range(len(population)):
        model = Chromosome(chromosome=list(population[i]))
        unet = model.get_unet()
        # Contar parámetros entrenables
        trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
        params[i] = trainable_params
    return params

  def evaluate_population(self, population):
    """
    Obtiene el fitness de la población de individuos.
    
    Parametros:
    ===========
        - population (np.ndarray): Población de individuos a evaluar (Arreglo de cromosomas).
    Retorna:
    ===========
        - np.ndarray: Aptitudes de los individuos.
    """
    # Evaluación del primer objetivo (Predicción de IoU)
    fitness_1 = self.surrogate_model.predict(population)
    fitness_1[fitness_1 < 0] = - np.inf
    fitness_1[fitness_1 > 1] = - np.inf
    fitness_1 = 1 - fitness_1  # Se invierte la función objetivo (Para minimizar)

    # Evaluación del segundo objetivo (Número de parámetros)
    fitness_2 = self.get_params(population)
    
    # Matriz de fitness
    fitness = np.column_stack((fitness_1, fitness_2))
    return fitness


class TwoObjectiveOptimizer:
    def __init__(self) -> None:
        """
        Clase base para optimizadores multiobjetivo con dos objetivos.
        Implementa métodos comunes para los algorítmos poblacionales.
        """
        self.evaluator = Evaluator(codification="real")
        self.population = []
        self.fitness = []
        self.fitness_history = []
        self.pareto_fronts = []
        self.pareto_history = []

    def initialize_population(self, n_pop: int) -> None:
        """
        Inicializar la población de individuos.
        
        Parametros:
        ===========
            - n_pop (int): Número de individuos en la población.
        """
        self.population = [Chromosome() for _ in range(n_pop)]
        self.population = np.array([chromosome.get_real() for chromosome in self.population])
        
    def get_pareto_front(self, fitness: np.ndarray) -> np.ndarray:
        """
        Obtiene el frente de Pareto para dos objetivos dada una matriz de fitness.
        Parametros:
        ===========
            - fitness (np.ndarray): Matriz de fitness de la población (cada fila es un individuo).
        Retorna:
        ===========
            - np.ndarray: Índices de la población que forman el frente de Pareto.
        """
        ### Método de selección para dos objetivos (minimización)
        ### Se ordenan según el primer objetivo, así se asegura que el primer objetivo siempre empeora
        ### Para que un punto sea no dominado, el segundo objetivo debe mejorar respecto al mínimo actual
        ### Si no mejora, será dominado por un punto anterior (mejor o igual en el primer objetivo y mejor en el segundo)

        # Ordenar por el primer objetivo
        sorted_idxs = np.argsort(fitness[:, 0])
        sorted_fitness = fitness[sorted_idxs]

        # Seleccionar puntos no dominados
        pareto_front_sorted = []
        obj2_min = np.inf # Valor inicial alto para el segundo objetivo
        for i in range(len(sorted_fitness)):
            # Condición de no dominancia (Si el segundo objetivo es menor que el mínimo actual)
            if sorted_fitness[i, 1] < obj2_min:
                pareto_front_sorted.append(sorted_idxs[i])
                obj2_min = sorted_fitness[i, 1]
        return np.array(pareto_front_sorted)

    def ranked_pareto_fronts(self, fitness: np.ndarray) -> List[np.ndarray]:
        """
        Obtiene los frentes de Pareto para una matriz de fitness.
        
        Parametros:
        ===========
            - fitness (np.ndarray): Matriz de fitness de la población (cada fila es un individuo).
        Retorna:
        ===========
            - List[np.ndarray]: Lista de frentes de Pareto, cada uno es un array de índicess de la población.
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
        Calcula la distancia de ahincamiento para un frente de Pareto dado.
        
        Parametros:
        ===========
            - front (np.ndarray): Índices de los individuos en el frente de Pareto.
            - fitness (np.ndarray): Matriz de fitness de la población (cada fila es un individuo).
            
        Retorna:
        ===========
            - np.ndarray: Distancias de ahincamiento para los individuos en el frente.
        """
        if len(front) == 0:
            return np.array([])

        distances = np.zeros(len(front))

        for i in range(2): # Para cada objetivo
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
                    # Sumar la distancia normalizada en el objetivo i
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
        Clase para el algoritmo de evolución diferencial (Adaptación para dos objetivos).

        Parametros:
            - base (str): Estrategia base para la evolución diferencial. Puede ser "random" o "current".
            - n_differences (int): Número de vectores diferencia a considerar, puede ser 1 o 2.
            - crossover (str): Tipo de cruce a utilizar, puede ser "bin" o "exp".
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

        Parametros:
        ===========
            - n_pop (int): Número de individuos en la población.
            - max_gen (int): Número máximo de generaciones.
            - mutation_rate (float): Tasa de mutación.
            - crossover_rate (float): Tasa de cruce.
        """
        self.initialize_population(n_pop)
        for gen in range(max_gen):
            print(f"Generación {gen+1}/{max_gen}")
            offspring = []
            for target in range(len(self.population)):
                # Selección del vector base
                if self.variant_dict["base"] == "random": # random to best
                    r1 = np.random.randint(0, len(self.population))
                else: # current to best
                    r1 = target

                # Selección de los vectores diferencia
                r2, r3 = random.sample([j for j in range(len(self.population)) if j != target and j != r1], 2)
                if self.variant_dict["n_differences"] == 1: # Caso de un vector diferencia
                    trial = self.population[r1] + mutation_rate * (self.population[r2] - self.population[r3])
                else: # Caso de dos vectores diferencia
                    r4, r5 = random.sample([j for j in range(len(self.population)) if j != target and j != r1 and j != r2 and j != r3], 2)
                    trial = self.population[r1] + mutation_rate * (
                            (self.population[r2] - self.population[r3]) + 
                            (self.population[r4] - self.population[r5]))

                # Crossover
                fixed_idx = np.random.randint(0, len(self.population[0])) # Asegurar que al menos un gen viene del trial
                fixed_value = self.population[trial][fixed_idx]
                if self.variant_dict["crossover"] == "bin":
                    crossover_mask = np.random.rand(len(self.population[0])) < crossover_rate
                    trial[crossover_mask] = self.population[target][crossover_mask]
                elif self.variant_dict["crossover"] == "exp":
                    # Punto de inicio
                    start = np.random.randint(0, len(self.population[0]))
                    j = start
                    while True:
                        trial[j] = self.population[target][j]
                        # Vuelta al inicio o probabilidad de corte
                        j = (j + 1) % len(self.population[0])
                        if j == start or np.random.rand() > crossover_rate:
                            break
                trial[fixed_idx] = fixed_value
                # Reparar trial
                trial = np.clip(trial, 0, 1)
                offspring.append(trial)
                
            # Mu + Lambda
            self.offspring = np.array(offspring)
            self.offspring = np.vstack((self.population, self.offspring))
            
            # Evaluar la población combinada            
            self.fitness = self.evaluator.evaluate_population(self.offspring)
            self.fitness_history.append(self.fitness)
            fronts = self.ranked_pareto_fronts(self.fitness)
            self.pareto_history.append(fronts)
            self.pareto_fronts = fronts

            # Selección de la nueva población
            new_population = []
            for front in fronts:
                # Selección de individuos de la misma frente
                front_individuals = self.offspring[front]
                # Selecciónar todo el frente, si sobrepasa el tamaño de la población, seleccionar por crowding distance
                if len(new_population) + len(front) <= n_pop:
                    new_population.extend(front_individuals)
                else:
                    # Ordenar por crowding distance y seleccionar los mejores
                    front_crowding_distances = self.get_crowding_distance(front, self.fitness)
                    sorted_indices = np.argsort(front_crowding_distances)[::-1]
                    selected_indices = sorted_indices[:n_pop - len(new_population)]
                    new_population.extend(front_individuals[selected_indices])
            self.population = np.array(new_population)
