from typing import Literal

import numpy as np
import torch.nn as nn

from codec import Chromosome
from surrogate.surrogate_model import SurrogateModel

class Evaluator():
  def __init__(self, codification: Literal["binary", "real"], dataset: Literal["carvana", "road"] = "carvana") -> None:
    """
    Clase para evaluar la población de individuos.
    """
    self.codification = codification
    self.surrogate_model = SurrogateModel(model_path = f"./subrogate_models/{dataset}_sub.json")
    
  def bin_to_real(self, population: np.ndarray) -> np.ndarray:
    """
    Convertir una población de individuos codificados en binario a codificación real.
    Args:
        population (np.ndarray): Población de individuos en codificación binaria.
    Returns:
        np.ndarray: Población de individuos codificados en real.
    """
    return np.array([Chromosome(chromosome=x).get_real() for x in population])
  
  def evaluate_population(self, population: np.ndarray) -> np.ndarray:
    """
    Evaluar la población de individuos.
    Args:
        population (np.ndarray): Población de individuos a evaluar.
    Returns:
        np.ndarray: Aptitudes de los individuos.
    """
    pop = population.copy()
    if self.codification == "binary":
      pop = np.array(["".join(map(str, s)) for s in population])
      population = self.bin_to_real(pop)
    fitness = self.surrogate_model.predict(pop)
    fitness[(fitness < 0) | (fitness > 1)] = - np.inf
    return fitness
  
  def evaluate_individual(self, individual: np.ndarray) -> float:
    """
    Evaluar un individuo.
    Args:
        individual (np.ndarray): Individuo a evaluar.
    Returns:
        float: Aptitud del individuo.
    """
    fitness = self.surrogate_model.predict(np.array(individual.reshape(1, -1)))
    if fitness < 0 or fitness > 1:
      fitness = -np.inf
    return fitness

class TwoObjectiveEvaluator():
  def __init__(self, codification: Literal["binary", "real"],  dataset: Literal["carvana", "road"] = "carvana") -> None:
    """
    Clase del evaluador multiobjetivo.
    
    Parametros:
    ===========
        - codifiication (str): Tipo de codificación, puede ser "binary" o "real".
    """
    self.surrogate_model = SurrogateModel(model_path = f"./subrogate_models/{dataset}_sub.json")
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
    # Evaluación del segundo objetivo (Número de parámetros)
    fitness_2 = self.get_params(population)
    
    # Evaluación del primer objetivo (Predicción de IoU invertida)
    if self.codification == "binary":
      population = self.bin_to_real(population)
    fitness_1 = 1 - self.surrogate_model.predict(population)
    fitness_1[(fitness_1 < 0) | (fitness_1 > 1)] = np.inf

    # Matriz de fitness
    fitness = np.column_stack((fitness_1, fitness_2))
    return fitness

  def evaluate_individual(self, individual: np.ndarray) -> np.ndarray:
    """
    Evaluar un individuo.
    Parametros:
    ===========
      - individual (np.ndarray): Individuo a evaluar (Cromosoma).
    Retorna:
    ===========
      - np.ndarray: Aptitudes del individuo.
    """ 
    fitness_1 = 1 - self.surrogate_model.predict(np.array(individual.reshape(1, -1)))
    if fitness_1 < 0 or fitness_1 > 1:
      fitness_1 = np.inf
    
    model = Chromosome(chromosome=list(individual))
    unet = model.get_unet()
    params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    fitness_2 = params
    
    return np.array([fitness_1.item(), fitness_2])
  
class CombinedMetricEvaluator():
  def __init__(self, codification: Literal["binary", "real"],  dataset: Literal["carvana", "road"] = "carvana", beta: float = 0.5) -> None:
    """
    Clase del evaluador multiobjetivo.
    
    Parametros:
    ===========
        - codifiication (str): Tipo de codificación, puede ser "binary" o "real".
    """
    self.surrogate_model = SurrogateModel(model_path = f"./subrogate_models/{dataset}_sub.json")
    self.codification = codification
    self.beta = beta
  
  def addition_penalty(self, iou, n_params, min_params=4, max_params=506_597_377):
    norm = ((np.log(n_params) - np.log(min_params)) /
            (np.log(max_params) - np.log(min_params)))
    return self.beta * iou + (1 - self.beta) * (1 - norm.squeeze())

  def bin_to_real(self, population: np.ndarray) -> np.ndarray:
    """
    Convertir una población de individuos codificados en binario a codificación real.
    Args:
        population (np.ndarray): Población de individuos en codificación binaria.
    Returns:
        np.ndarray: Población de individuos codificados en real.
    """
    return np.array([Chromosome(chromosome=x).get_real() for x in population])
  
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
    pop = population.copy()
    if self.codification == "binary":
      pop = np.array(["".join(map(str, s)) for s in population])
      pop = self.bin_to_real(pop)
    # Evaluación del segundo objetivo (Número de parámetros)
    fitness_2 = self.get_params(pop)
    
    # Evaluación del primer objetivo (Predicción de IoU invertida)
    fitness_1 = self.surrogate_model.predict(pop)
    fitness_1[(fitness_1 < 0) | (fitness_1 > 1)] = np.inf
  
    # Matriz de fitness
    fitness = self.addition_penalty(fitness_1, fitness_2)

    return fitness

  def evaluate_individual(self, individual: np.ndarray) -> np.ndarray:
    """
    Evaluar un individuo.
    Parametros:
    ===========
      - individual (np.ndarray): Individuo a evaluar (Cromosoma).
    Retorna:
    ===========
      - np.ndarray: Aptitudes del individuo.
    """ 
    ind = individual.copy()
    if self.codification == "binary":
      ind = "".join(map(str, individual))
      ind = Chromosome(chromosome=ind).get_real()
  
    fitness_1 = self.surrogate_model.predict(np.array(ind.reshape(1, -1)))
    if fitness_1 < 0 or fitness_1 > 1:
      fitness_1 = np.inf
    
    model = Chromosome(chromosome=list(individual))
    unet = model.get_unet()
    params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    fitness_2 = params
    
    return self.addition_penalty(fitness_1, np.array([fitness_2, ]))
  
  def get_params__(self, population: np.ndarray) -> np.ndarray:
    """
    Contar solo los parámetros pertenecientes a capas Conv2d o ConvTranspose2d.
    """
    params = np.zeros((len(population), 1))

    for i in range(len(population)):
      model = Chromosome(chromosome=list(population[i]))
      unet = model.get_unet()

      conv_params = 0

      # Recorremos los módulos y filtramos solo convs
      for module in unet.modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
          for p in module.parameters(recurse=False):  # parámetros directos del módulo
            conv_params += p.numel()

      params[i] = conv_params

      return params