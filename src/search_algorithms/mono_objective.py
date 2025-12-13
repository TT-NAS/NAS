# Algorítmos de búsqueda de un solo objetivo
# Fitness = IoU (maximizar)
from typing import List, Literal, Optional

import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import json
from time import time

from .evaluator import CombinedMetricEvaluator
from codec import Chromosome
from surrogate.surrogate_model import SurrogateModel

class SearchAlgorithm():
  def __init__(self, codification: Literal["binary", "real"]) -> None:
    """
    Clase base para los algoritmos de búsqueda.

    Args:
        evaluator (Evaluator): Evaluador de la población de individuos.
        codification (Literal["binary", "real"]): Tipo de codificación de los individuos.
    """
    assert codification in ["binary", "real"], "Codification must be 'binary' or 'real'"
    self.codification = codification
    self.evaluator = None
    self.population = None
    self.lower = []
    self.upper = []
    self.mean = []
    self.diversity = []
    self.results = {}

    self.diversity_loss = False
    self.reached_target = False
    self.reached_gens = False
  def initialize_population(self, n_pop: int) -> np.ndarray:
    """
    Inicializar la población de individuos.
    Args:
        n_pop (int): Número de individuos en la población.
    """
    self.population = [Chromosome() for _ in range(n_pop)]
    if self.codification == "binary":
      population = np.array([chromosome.get_binary() for chromosome in self.population])
      self.population = np.array([list(i) for i in population], dtype=np.uint8)
    else:
      self.population = np.array([chromosome.get_real() for chromosome in self.population])

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

  def start(self, *args, **kwargs) -> None:
    raise NotImplementedError("Implemented in subclass")

  def plot_convergence(self, save_path = None) -> None:
    """
    Graficar la convergencia algoritmo.
    """
    plt.plot(self.lower, label="Mejor")
    plt.plot(self.upper, label="Peor")
    plt.plot(self.mean, label="Promedio")
    plt.xlabel("Generación", fontsize=18)
    plt.ylabel("Aptitud", fontsize=18)
    plt.title("Grafico de convergencia", fontsize=18)
    plt.legend(fontsize=18)
    if save_path is not None:
      plt.savefig(save_path)
      plt.close()
    else:
      plt.show()

  def plot_diversity(self, save_path = None) -> None:
    plt.plot(self.diversity)
    plt.xlabel("Generación", fontsize=18)
    plt.ylabel("Diversidad", fontsize=18)
    plt.title("Gráfico de diversidad", fontsize=18)
    if save_path is not None:
      plt.savefig(save_path)
      plt.close()
    else:
      plt.show()

class DifferentialEvolution(SearchAlgorithm):
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
    if self.variant_dict["crossover"] == "bin":
      fixed_idx = np.random.randint(0, D)
      crossover_mask = np.random.rand(D) <= crossover_rate
      crossover_mask[fixed_idx] = True

    elif self.variant_dict["crossover"] == "exp":
      crossover_mask = np.zeros(D, dtype=bool)
      j = np.random.randint(0, D)
      L = 0
      while True:
        crossover_mask[j] = True
        L += 1
        if L == D or np.random.rand() > crossover_rate:
            break
        j = (j + 1) % D

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

  async def start(self, n_pop: int = 100, max_gen: int = 1000, F: float = 0.5, crossover_rate: float = 0.9,
            diversity_min: float = None, target_fitness: float = None):
    """
    Algoritmo de evolución diferencial.

    Args:
        n_pop (int): Número de individuos en la población.
        max_gen (int): Número máximo de generaciones.
        F (float): Escala del vector diferencia.
        crossover_rate (float): Tasa de cruza.
    """
    if self.evaluator is None:
      raise ValueError("Evaluator not set. Please set the evaluator before starting the algorithm.")

    start_time = time()
    self.initialize_population(n_pop)
    stop_conditions = []
    gen = 0
    while True:
      offspring = []
      fitness = self.evaluator.evaluate_population(self.population)
      # Selección del vector base
      if self.variant_dict["base"] == "best":
        v1 = np.argmax(fitness)
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
        offspring.append(trial)

      # Selección
      new_population = []
      offspring = np.array(offspring)
      offspring_fitness = self.evaluator.evaluate_population(offspring)
      mask = offspring_fitness >= fitness
      new_population = np.where(mask[:, None], offspring, self.population)
      self.population = np.array(new_population)

      # Estadísticas
      self.lower.append(np.min(fitness))
      self.upper.append(np.max(fitness))
      self.mean.append(np.mean(fitness))
      self.diversity.append(self.get_diversity())

      sys.stdout.write(f"\r[Generation: {gen+1}/{max_gen}] Upper: {self.upper[-1]} - Mean: {self.mean[-1]} - Lower: {self.lower[-1]} - Diversity: {self.diversity[-1]}")
      sys.stdout.flush()

      progress_payload = {
        "type": "progress",
        "generation": gen + 1,
        "best_fitness": float(self.upper[-1]),
        "best_binary": Chromosome(chromosome=self.population[np.argmax(fitness)].tolist()).get_binary(zip=True),
        "best_real": self.population[np.argmax(fitness)].tolist()
      }
      yield json.dumps(progress_payload) + "\n"

      # Stop conditions
      if diversity_min and self.diversity[-1] <= diversity_min:
        stop_conditions.append("Diversity minimum reached")
        self.diversity_loss = True
      if target_fitness and self.upper[-1] >= target_fitness:
        stop_conditions.append("Target fitness reached")
        self.reached_target = True
      if gen == max_gen-1:
        stop_conditions.append("Max generations reached")
        self.reached_gens = True
      if len(stop_conditions) > 0:
        print("\nStopping criteria met: " + ", ".join(stop_conditions))
        break
      gen += 1

    self.gen = gen
    self.fitness = self.evaluator.evaluate_population(self.population)

    self.search_time = time() - start_time
    final_chromosome = Chromosome(chromosome=self.population[np.argmax(self.fitness)].tolist())
    final_json = final_chromosome.get_json()

    def _ensure_basic(obj, fallback=None):
      if obj is None:
        return fallback
      if hasattr(obj, "tolist"):
        return obj.tolist()
      if isinstance(obj, (np.ndarray, np.generic)):
        return obj.item() if np.isscalar(obj) else obj.tolist()
      return obj

    real_codification = _ensure_basic(final_chromosome.get_real(), [])

    result_payload = {
        "type": "result",
        "message": "Búsqueda completada exitosamente",
        "results": {
            "predicted_iou": float(self.upper[-1]),
            "search_time": self.search_time,
            "stop_gen": self.gen,
            "stop_reason": ", ".join(stop_conditions),
            "vector": [float(fitness) for fitness in self.upper],
            "real_codification": real_codification,
            "binary": final_chromosome.get_binary(zip=True),
            "architecture": final_json.get("unet") if isinstance(final_json, dict) else final_json,
            "trained": False
        }
    }

    yield json.dumps(result_payload) + "\n"


class GeneticAlgorithm(SearchAlgorithm):
  def __init__(self,
                selection: Literal["roulette", "tournament"] = "tournament",
                crossover: Literal["one_point", "two_point", "uniform"] = "two_point",
                mutation_type: Literal["bit_flip"] = "bit_flip") -> None:
    """
    Clase para el algoritmo genético.
    Args:
      selection (Literal["roulette", "tournament"]): Método de selección.
      crossover (Literal["one_point", "two_point", "uniform"]): Tipo de cruce.
      mutation_type (Literal["bit_flip"]): Tipo de mutación (actualmente solo bit_flip).
    """
    super().__init__("binary")
    assert selection in ["roulette", "tournament"], "selection must be 'roulette' or 'tournament'"
    assert crossover in ["one_point", "two_point", "uniform"], "crossover must be 'one_point', 'two_point' or 'uniform'"
    assert mutation_type in ["bit_flip"], "mutation_type must be 'bit_flip'"

    self.variant_dict = {
        "selection": selection,
        "crossover": crossover,
        "mutation_type": mutation_type
    }

  async def start(self, n_pop: int = 100, max_gen: int = 100,
            crossover_rate: float = 0.9, mutation_rate: float = 0.01,
            tournament_size: int = 3, diversity_min: float = None, target_fitness: float = None):
    """
    Iniciar el algoritmo genético.

    Args:
        n_pop (int): Número de individuos en la población.
        max_gen (int): Número máximo de generaciones.
        crossover_rate (float): Probabilidad de cruce.
        mutation_rate (float): Probabilidad de mutación por gen.
        tournament_size (int): Tamaño del torneo para selección por torneo.
    """
    self.initialize_population(n_pop)
    if self.evaluator is None:
      raise ValueError("Evaluator not set. Please set the evaluator before starting the algorithm.")

    start_time = time()
    fitness = self.evaluator.evaluate_population(self.population)
    gen = 0
    while True:
      offspring = []
      for _ in range(n_pop // 2):
        # Selección de padres
        if self.variant_dict["selection"] == "roulette":
          parent1 = self._roulette_selection(fitness)
          parent2 = self._roulette_selection(fitness)
        else:  # tournament
          parent1 = self._tournament_selection(fitness, tournament_size)
          parent2 = self._tournament_selection(fitness, tournament_size)
        # Cruce
        if np.random.rand() < crossover_rate:
          child1, child2 = self._crossover(parent1, parent2)
        else:
          child1, child2 = parent1.copy(), parent2.copy()

        # Mutación
        child1 = self._mutate(child1, mutation_rate)
        child2 = self._mutate(child2, mutation_rate)

        offspring.extend([child1, child2])

      # Elitismo (Mantener al mejor)
      best_idx = np.argmax(fitness)
      offspring[0] = self.population[best_idx].copy()

      self.population = offspring[:n_pop]  # Asegurar tamaño correcto

      # Estadísticas
      current_fitness = self.evaluator.evaluate_population(self.population)
      self.lower.append(np.min(current_fitness))
      self.upper.append(np.max(current_fitness))
      self.mean.append(np.mean(current_fitness))
      self.diversity.append(self.get_diversity())

      fitness = current_fitness
      sys.stdout.write(f"\r[Generation: {gen+1}/{max_gen}] Upper: {self.upper[-1]} - Mean: {self.mean[-1]} - Lower: {self.lower[-1]} - Diversity: {self.diversity[-1]}")
      sys.stdout.flush()

      progress_payload = {
          "type": "progress",
          "generation": gen + 1,
          "best_fitness": float(self.upper[-1]),
          "best_binary": Chromosome(chromosome="".join(self.population[np.argmax(fitness)].astype(str).tolist())).get_binary(zip=True),
          "best_real": self.population[np.argmax(fitness)].tolist()
      }
      yield json.dumps(progress_payload) + "\n"

      # Stop conditions
      stop_conditions = []
      if diversity_min and self.diversity[-1] <= diversity_min:
        stop_conditions.append("Diversity minimum reached")
        self.diversity_loss = True
      if target_fitness and self.upper[-1] >= target_fitness:
        stop_conditions.append("Target fitness reached")
        self.reached_target = True
      if gen == max_gen-1:
        stop_conditions.append("Max generations reached")
        self.reached_gens = True
      if len(stop_conditions) > 0:
        print("\nStopping criteria met: " + ", ".join(stop_conditions))
        break
      gen += 1
    self.gen = gen
    self.fitness = self.evaluator.evaluate_population(self.population)

    self.search_time = time() - start_time
    final_chromosome = Chromosome(chromosome="".join(self.population[np.argmax(self.fitness)].astype(str).tolist()))
    final_json = final_chromosome.get_json()

    def _ensure_basic(obj, fallback=None):
      if obj is None:
        return fallback
      if hasattr(obj, "tolist"):
        return obj.tolist()
      if isinstance(obj, (np.ndarray, np.generic)):
        return obj.item() if np.isscalar(obj) else obj.tolist()
      return obj

    real_codification = _ensure_basic(final_chromosome.get_real(), [])

    result_payload = {
        "type": "result",
        "message": "Búsqueda completada exitosamente",
        "results": {
            "predicted_iou": float(self.upper[-1]),
            "search_time": self.search_time,
            "stop_gen": self.gen,
            "stop_reason": ", ".join(stop_conditions),
            "vector": [float(fitness) for fitness in self.upper],
            "real_codification": real_codification,
            "binary": final_chromosome.get_binary(zip=True),
            "architecture": final_json.get("unet") if isinstance(final_json, dict) else final_json,
            "trained": False
        }
    }
    yield json.dumps(result_payload) + "\n"

  def _roulette_selection(self, fitness: list) -> np.ndarray:
    """Selección por ruleta."""
    wights = fitness / np.sum(fitness)
    return self.population[np.random.choice(len(self.population), p=wights)]

  def _tournament_selection(self, fitness: list, size: int) -> np.ndarray:
    """Selección por torneo."""
    contestants = np.random.choice(len(self.population), size=size, replace=False)
    best = contestants[np.argmax(fitness[contestants])]
    return self.population[best]

  def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple:
    """Operadores de cruce."""
    D = len(parent1)
    if self.variant_dict["crossover"] == "one_point":
      point = np.random.randint(1, D)
      child1 = np.concatenate([parent1[:point], parent2[point:]])
      child2 = np.concatenate([parent2[:point], parent1[point:]])

    elif self.variant_dict["crossover"] == "two_point":
      points = sorted(np.random.choice(D, size=2, replace=False))
      child1 = np.concatenate([
        parent1[:points[0]],
        parent2[points[0]:points[1]],
        parent1[points[1]:]
      ])
      child2 = np.concatenate([
        parent2[:points[0]],
        parent1[points[0]:points[1]],
        parent2[points[1]:]
      ])

    else:  # uniform
      mask = np.random.rand(D) < 0.5
      child1 = np.where(mask, parent1, parent2)
      child2 = np.where(mask, parent2, parent1)
    return child1, child2

  def _mutate(self, individual: np.ndarray, mutation_rate: float) -> np.ndarray:
    """Mutación"""
    mutation_mask = np.random.rand(len(individual)) < mutation_rate
    individual = individual.copy()
    individual[mutation_mask] = individual[mutation_mask]
    return individual
