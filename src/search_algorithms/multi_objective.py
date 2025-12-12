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
from .evaluator import TwoObjectiveEvaluator
        
class TwoObjectiveOptimizer:
  def __init__(self, codification) -> None:
    """
    Clase base para optimizadores multiobjetivo con dos objetivos.
    Implementa métodos comunes para los algorítmos poblacionales.
    """
    self.codification = codification
    self.evaluator = TwoObjectiveEvaluator(codification)
    self.fitness = None
    self.pareto_front = None
    self.population = None
    self.fitness_history = []
    self.pareto_history = []
    self.diversity = []
    self.nds_history = []

  def initialize_population(self, n_pop: int) -> None:
    """
    Inicializar la población de individuos.
    
    Parametros:
    ===========
        - n_pop (int): Número de individuos en la población.
    """
    self.population = [Chromosome() for _ in range(n_pop)]
    self.population = np.array([chromosome.get_real() for chromosome in self.population])
  
  def dominates(self, fitness_a: np.ndarray, fitness_b: np.ndarray) -> bool:
    """
    Verifica si un individuo A domina a otro individuo B.
    """
    return all(a <= b for a, b in zip(fitness_a, fitness_b)) and any(a < b for a, b in zip(fitness_a, fitness_b))
  
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
        continue  # Evita división por cero si todos son iguales en este objetivo

      # Extremos reciben infinito
      distances[sorted_indices[0]] = np.inf
      distances[sorted_indices[-1]] = np.inf

      # Calcular crowding para los puntos intermedios
      for j in range(1, len(front) - 1):
        if not np.isinf(distances[sorted_indices[j]]):  # No sobrescribir extremos
          # Sumar la distancia normalizada en el objetivo i
          distances[sorted_indices[j]] += ((sorted_fitness[j + 1] - sorted_fitness[j - 1]) / (f_max - f_min))
    return distances
      
  def get_diversity(self) -> float:
    """
    Calcula la diversidad fenotípica de la población.
    Returns.
      float: Diversidad de la población.
    """
    if self.codification == "real": # Distancia euclidiana para codificación real
      distance_metric = lambda x, y: np.linalg.norm(x-y)
    else: # Distancia hamming para codificación binaria
      distance_metric = lambda x, y: np.sum(x != y)
      
    mean_distance = 0
    n = len(self.population)
    for i in range(n-1):
      for j in range(i+1, n):
        mean_distance+=distance_metric(self.population[i], self.population[j])
    # Normalizaciones
    if self.codification == "real":
      return mean_distance/(n*(n-1))
    else:
      d = len(self.population[0])
      return mean_distance/(n*(n-1)*d)

  def plot_pareto_fronts(self, fitness: np.ndarray, fronts) -> None:
      """
      Plotea los frentes de Pareto de la población dada.
      """
      plt.figure(figsize=(10, 6))
      for i, front in enumerate(fronts):
          plt.scatter(fitness[front, 0], fitness[front, 1], label=f'Front {i+1}')
      plt.xlabel('IoU (Predicted)')
      plt.ylabel('No. Parámetros')
      plt.title('Frentes de pareto')
      plt.legend()
      plt.grid()
      plt.show()
    
  def plot_pareto_front(self, save_path = None) -> None:
      """
      Plotea el último frente de pareto
      """
      plt.figure(figsize=(10, 6))
      plt.scatter(1 - self.fitness[self.pareto_front, 0], self.fitness[self.pareto_front, 1], color='red')
      plt.xlabel('IoU (Predicted)')
      plt.ylabel('No. Parámetros')
      plt.title('Frente de pareto')
      plt.legend()
      plt.grid()
      if save_path:
          plt.savefig(save_path)
      else:
        plt.show()
  
  def plot_diversity(self) -> None:
    plt.plot(self.diversity)
    plt.xlabel("Generación")
    plt.ylabel("Diversidad")
    plt.title("Historial de diversidad")
    plt.legend()
    plt.show()

  def plot_nds_history(self) -> None:
    plt.plot(self.nds_history)
    plt.xlabel("Generacion")
    plt.ylabel("NDS")
    plt.title("Historial de NDS")
    plt.legend()
    plt.show()

class DifferentialEvolution(TwoObjectiveOptimizer):
  def __init__(self,  base: Literal["best", "random", "current"] = "best", 
                n_differences: int = 1, crossover: Literal["bin", "exp"] = "bin") -> None:
    """
    Clase para el algoritmo de evolución diferencial.
    
    Args:
        base (Literal["best", "random", "current"]): Estrategia base para la evolución diferencial.
        n_differences (int): Número de vectores diferencia a considerar, puede ser 1 o 2.
        crossover (Literal["bin", "exp"]): Tipo de cruce a utilizar, puede ser binario o exponencial.
    """
    super().__init__("real")
    assert base in ["best", "random", "current"], "base must be 'best', 'random', 'current' or 'current_to_best'"
    assert n_differences == 1 or n_differences == 2, "n_differences must be 1 or 2"
    assert crossover in ["bin", "exp"], "crossover must be 'bin' or 'exp'"
    
    self.variant_dict = {
      "base": base,
      "n_differences": n_differences,
      "crossover": crossover
    }
    
  def _crossover(self, target: np.ndarray, vy: np.ndarray, crossover_rate: float) -> np.ndarray:
    """
    Operador de cruza.
    """
    D = len(target)
    fixed_idx = np.random.randint(0, D)
    
    if self.variant_dict["crossover"] == "bin":
      crossover_mask = np.random.rand(D) <= crossover_rate
      
    elif self.variant_dict["crossover"] == "exp":
      crossover_mask = np.zeros(D, dtype=bool)
      start = np.random.randint(0, D)
      j = start
      while True:
        crossover_mask[j] = True
        j = (j + 1) % D
        if j == start or np.random.rand() > (1 - crossover_rate): # No estoy seguro si utilizar CR o 1-CR
            break
    
    crossover_mask[fixed_idx] = True
    trial = target.copy()
    trial[crossover_mask] = vy[crossover_mask]
    
    return trial
  
  def _difference_vector(self, v1: int, F: float) -> np.ndarray:
    """
    Obtener el vector diferencia.
    """
    if self.variant_dict["n_differences"] == 1:
      v2, v3 = random.sample([j for j in range(len(self.population)) if j != v1], 2)
      diff_vector = self.population[v2] - self.population[v3]
    elif self.variant_dict["n_differences"] == 2:
      v2, v3, v4, v5 = random.sample([j for j in range(len(self.population)) if j != v1], 4)
      diff_vector = (self.population[v2] - self.population[v3]) + (self.population[v4] - self.population[v5])
      
    return F * diff_vector
  
  def start(self, n_pop: int = 100, max_gen: int = 1000, F: float = 0.5, crossover_rate: float = 0.9) -> None:
    """
    Algoritmo de evolución diferencial.
    
    Args:
        n_pop (int): Número de individuos en la población.
        max_gen (int): Número máximo de generaciones.
        F (float): Escala del vector diferencia.
        crossover_rate (float): Tasa de cruza.
    """
    self.initialize_population(n_pop)
    fitness = self.evaluator.evaluate_population(self.population)
    for gen in range(max_gen):
      offspring = []
      offspring_fitness = []
      # Selección del vector base
      if self.variant_dict["base"] == "best":
        # Obtener el primer frente
        pareto_front = self.get_pareto_front(fitness)
        # Mejor individuo por crowding distance
        crowding_distances = self.get_crowding_distance(pareto_front, fitness)
        v1 = pareto_front[np.argmax(crowding_distances)]
        
      for target in range(len(self.population)):
        if self.variant_dict["base"] == "random":
          v1 = np.random.randint(0, len(self.population))
        else:
          v1 = target        
        # Selección de los vectores diferencia
        vy = self.population[v1] + self._difference_vector(v1, F)
        # Cruce
        trial = self._crossover(self.population[target], vy, crossover_rate)        
        # Fix
        trial = np.clip(trial, 0, 1)
        
        # Reemplazo por dominancia
        fitness_trial = self.evaluator.evaluate_individual(trial)
        fitness_target = fitness[target]
        
        if self.dominates(fitness_trial, fitness_target):
          offspring.append(trial)
          offspring_fitness.append(fitness_trial)
        elif self.dominates(fitness_target, fitness_trial):
          offspring.append(self.population[target])
          offspring_fitness.append(fitness_target)
        else:
          offspring.extend([trial, self.population[target]])
          offspring_fitness.extend([fitness_trial, fitness_target])

      # Selección por frentes de Pareto y crowding distance   
      offspring = np.array(offspring)
      offspring_fitness = np.array(offspring_fitness)
         
      fronts = self.ranked_pareto_fronts(offspring_fitness)
      self.pareto_history.append(fronts)
      self.fitness_history.append(offspring_fitness)
      
      new_population = []
      new_fitness = []
      for front in fronts:
        # Selección de individuos de la misma frente
        front_individuals = offspring[front]
        # Selecciónar todo el frente, si sobrepasa el tamaño de la población, seleccionar por crowding distance
        if len(new_population) + len(front) <= n_pop:
          new_population.extend(front_individuals)
          new_fitness.extend([offspring_fitness[i] for i in front])
        else:
          # Ordenar por crowding distance y seleccionar los mejores
          front_crowding_distances = self.get_crowding_distance(front, offspring_fitness)
          sorted_indices = np.argsort(front_crowding_distances)[::-1]
          selected_indices = sorted_indices[:n_pop - len(new_population)]
          new_population.extend(front_individuals[selected_indices])
          new_fitness.extend([offspring_fitness[front[i]] for i in selected_indices])
          break
    
      self.population = np.array(new_population)
      fitness = np.array(new_fitness)
      
      #self.diversity.append(self.get_diversity())
      self.diversity.append(0) # Por tiempo de ejecución
      self.nds_history.append(len(fronts[0]))
      
      sys.stdout.write(f"\r[Generation {gen}/{max_gen}] - NDS: {len(fronts[0])} - Diversity: {self.diversity[-1]:.4f}")
      sys.stdout.flush()

    # Guardar el ultimo frente
    self.pareto_front = self.get_pareto_front(fitness)
    self.fitness = fitness