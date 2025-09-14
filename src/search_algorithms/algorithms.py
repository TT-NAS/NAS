from codec import Chromosome
from .surrogate import SurrogateModel
import numpy as np
import random
from typing import List, Literal, Optional

class Evaluator():
  def __init__(self, codification: Literal["binary", "real"]) -> None:
    """
    Clase para evaluar la población de individuos.
    """
    self.surrogate_model = SurrogateModel(model_path = r"./sustituto/xgboost_model.json")
    self.codification = codification

    
  def bin_to_real(self, population: np.ndarray) -> np.ndarray:
    """
    Convertir una población de individuos codificados en binario a codificación real.
    Args:
        population (np.ndarray): Población de individuos codificados en binario.
    Returns:
        np.ndarray: Población de individuos codificados en real.
    """
    return np.array([Chromosome(chromosome=ind).get_real() for ind in population])
  
  def evaluate_population(self, population):
    """
    Evaluar la población de individuos.
    Args:
        population (np.ndarray): Población de individuos a evaluar.
    Returns:
        np.ndarray: Aptitudes de los individuos.
    """
    if self.codification == "binary":
      population = self.bin_to_real(population)
    fitness = self.surrogate_model.predict(population)
    fitness[fitness < 0] = - np.inf
    fitness[fitness > 1] = - np.inf
    return fitness
  
  def evaluate_individual(self, individual: np.ndarray) -> float:
    """
    Evaluar un individuo.
    Args:
        individual (np.ndarray): Individuo a evaluar.
    Returns:
        float: Aptitud del individuo.
    """
    individual = individual.reshape(1, -1)
    fitness = self.surrogate_model.predict(np.array(individual))
    if fitness < 0 or fitness > 1:
      return -np.inf
    return fitness

# Modelo base para las estrategias de búsqueda (Hereda los métodos de inicialización de la población y evaluación de la población, graficas de convergencia)
class SearchAlgorithm():
  def __init__(self, codification: Literal["binary", "real"]) -> None:
    """
    Clase base para los algoritmos de búsqueda.
    
    Args:
        evaluator (Evaluator): Evaluador de la población de individuos.
        codification (Literal["binary", "real"]): Tipo de codificación de los individuos, puede ser "binary" o "real".
    """
    assert codification in ["binary", "real"], "Codification must be 'binary' or 'real'"
    self.evaluator = Evaluator(codification=codification)
    self.population = None
    self.codification = codification
  
    self.lower = []
    self.upper = []
    self.mean = []
    
  def initialize_population(self, n_pop: int) -> np.ndarray:
    """
    Inicializar la población de individuos.
    Args:
        n_pop (int): Número de individuos en la población.
    """
    self.population = [Chromosome() for _ in range(n_pop)]
    if self.codification == "binary":
      self.population = np.array([chromosome.get_binary() for chromosome in self.population])
    else:
      self.population = np.array([chromosome.get_real() for chromosome in self.population])
  
  def stop_conditions(self, criteria: List[Literal["diversity_loss", "fitness_stagnation", "target_reached"]],
                     epsilon: Optional[float] = 0.01, target: Optional[float] = None) -> bool:
    """
    Verificar si se cumple alguna condición de parada.
    
    Args:
        criteria (List[Literal["diversity_loss", "fitness_stagnation", "target_reached"]]: Lista de criterios de parada.
        epsilon (Optional[float]): Umbral para la diversidad o estancamiento de la aptitud.
        target (Optional[float]): Valor objetivo para la aptitud.
    Returns:
        bool: Verdadero si se cumple alguna condición de parada, falso en caso contrario.
    """
    if self.upper and self.lower:
      if "diversity_loss" in criteria:
        diversity = np.std(self.upper[-1] - self.lower[-1])
        if diversity < epsilon:
          return True
      if "fitness_stagnation" in criteria and len(self.mean) >= 10:
        stagnation = np.std(self.mean[-10:])
        if stagnation < epsilon:
          return True
      if "target_reached" in criteria and target is not None:
        if np.max(self.upper) >= target:
          return True
    return False

  def plot_convergence(self):
    """
    Graficar la convergencia del algoritmo.
    """
    import matplotlib.pyplot as plt
    plt.plot(self.lower, label="Lower")
    plt.plot(self.upper, label="Upper")
    plt.plot(self.mean, label="Mean")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Convergence Plot")
    plt.legend()
    plt.show()

class DiferentialEvolution(SearchAlgorithm):
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
    assert base in ["best", "random", "current", "current_to_best"], "base must be 'best', 'random', 'current' or 'current_to_best'"
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
      new_population = []
      fitness = self.evaluator.evaluate_population(self.population)
      for target in range(len(self.population)):
        # Selección del vector base
        match self.variant_dict["base"]:
          case "best":
            r1 = np.argmax(fitness)
          case "random":
            r1 = np.random.randint(0, len(self.population))
          case "current":
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
        trial[fixed_idx] = fixed_value
        
        # selection
        if self.evaluator.evaluate_individual(trial) > fitness[target]:
          new_population.append(trial)
        else:
          new_population.append(self.population[target])
        
      self.population = new_population
      self.lower.append(np.min(fitness))
      self.upper.append(np.max(fitness))
      self.mean.append(np.mean(fitness))
      
    #  print(f"Generation {gen}: Lower={self.lower[-1]}, Upper={self.upper[-1]}, Mean={self.mean[-1]}")
    
    self.plot_convergence() 