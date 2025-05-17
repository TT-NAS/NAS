# To Do:
# - Test random selection vs best selection
# - Test uniform crossover vs exponential crossover
import sys
from time import time

import random
import numpy as np

from codec import Chromosome
from .surrogate import SurrogateModel

class DiferentialEvolution():
  def __init__(self, surrogate_model: SurrogateModel,  pop_size: int = 10, f: float = 0.9, 
               crossover_rate: float = 0.9, mutation_rate: float = 0.2, max_gen: int = 100) -> None:
    """
    Differential Evolution Algorithm
    Args:
        surrogate_model (SurrogateModel): Surrogate model to be used.
        pop_size (int, optional): Population size. Defaults to 100.
        f (float, optional): Scale factor. Defaults to 0.9.
        crossover_rate (float, optional): Crossover rate. Defaults to 0.9.
        max_gen (int, optional): Maximum number of generations. Defaults to 100.
    
    """
    assert surrogate_model is not None, "Surrogate model cannot be None"
    
    self.surrogate_model = surrogate_model
    self.pop_size = pop_size
    self.max_gen = max_gen
    self.crossover_rate = crossover_rate
    self.mutation_rate = mutation_rate
    self.f = f
    
    self.g = 0
    self.best = None
    self.best_fitness = None
    self.fitness = None
    self.search_time = 0
    self.stop_reason = None
    
    self.upper = []
    self.lower = []
    self.mean =[]
  
  def initialize_population(self):
    """
    Initialize the population
    """
    self.population = [Chromosome() for _ in range(self.pop_size)]
    self.population = np.array([chromosome.get_real() for chromosome in self.population])

  def evaluate_population(self):
    """
    Evaluate the population
    """
    fitness = self.surrogate_model.predict(self.population)
    fitness[fitness < 0] = - np.inf
    fitness[fitness > 1] = - np.inf
    self.fitness = fitness
    
  def evaluate_individual(self, individual):
    """
    Evaluate an individual
    """
      
    fitness = self.surrogate_model.predict(np.array(individual))
    
    if fitness < 0 or fitness > 1:
      return -np.inf
    return fitness
  
  def start(self):
    # Initialize population
    self.initialize_population()
    self.search_time= 0
    start_time = time()
    for g in range(self.max_gen):
      sys.stdout.write(f"\r[{g+1}/{self.max_gen}] - Mejor aptitúd (Predicción del IoU): {self.best_fitness}")
      sys.stdout.flush()
      # Iterate for target vector
      self.evaluate_population()
      for i, target in enumerate(self.population):
        # Trial vector
        r1 = np.argmax(self.fitness)
        r2, r3 = random.sample([num for num in range(self.pop_size) if num != i and num!=r1], k=2)
        v = self.population[r1]#
        d = self.f * (self.population[r2]- self.population[r3])
        v = v + self.f * d
        #Crossover target and trial
        u=[]
        #Select a random index
        idx = random.randint(0, self.surrogate_model.n_features_in_)

        #Uniform crossover
        for f in range(target.shape[0]):
          if random.random() <= self.crossover_rate or f==idx:
            u.append(v[f])
          else:
            u.append(target[f])
        u = np.array(u)

        # Mutation
        for f in range(u.shape[0]):
          if random.random() <= self.mutation_rate:
            u[f] += np.random.uniform(-0.1, 0.1)
            u[f]= max(u[f], 0)
            u[f]= min(u[f], 1)
          else:
            u[f] = target[f]
        
        # Repair
        u = np.clip(u, 0, 1)
        
        #Selection
        u_score = self.evaluate_individual(u.reshape(1, -1))
        target_score = self.evaluate_individual(target.reshape(1, -1))
        if (u_score > target_score):
          self.population[i] = u
          self.fitness[i] = self.evaluate_individual(u.reshape(1, -1))
      
      self.best = self.population[np.argmax(self.fitness)]
      self.best_fitness = self.fitness[np.argmax(self.fitness)]
      
      self.g+=1
      epsilon = 1e-15
      
      self.upper.append(np.max(self.fitness).item())
      self.lower.append(np.min(self.fitness).item())
      self.mean.append(np.mean(self.fitness).item())
      

      #Stop conditions
      if(max(self.fitness) > 0.985):
        print(f'\nDetenido. Arquitectura encontrada - IoU predicho: {max(self.fitness)}')
        self.stop_reason = 'Arquitectura encontrada'
        break

      elif(abs(max(self.fitness) - min(self.fitness)) < epsilon):
        print('\nDetenido. Pedida de diversidad')
        self.stop_reason = 'Perdida de diversidad'
        break

      elif(self.g >= self.max_gen):
        print('\nDetenido. Maximo de generaciones alcanzado')
        self.stop_reason = 'Maximo de generaciones alcanzado'
        break
    self.search_time = time() - start_time