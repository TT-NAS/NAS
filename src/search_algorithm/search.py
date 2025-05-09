# To Do:
# - Test random selection vs best selection
# - Test uniform crossover vs exponential crossover
import os
import sys

import random
import torch

from codec import Chromosome
from .surrogate import SurrogateModel, dim_reduction
from sklearn.cross_decomposition import PLSRegression

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class DiferentialEvolution():
  def __init__(self, surrogate_model: SurrogateModel, dim_reduction_model: PLSRegression, pop_size: int = 10, f: float = 0.9, crossover_rate: float = 0.9, max_gen: int = 100) -> None:
    """
    Differential Evolution Algorithm
    Args:
        surrogate_model (SurrogateModel): Surrogate model to be used.
        pop_size (int, optional): Population size. Defaults to 100.
        f (float, optional): Scale factor. Defaults to 0.6.
        crossover_rate (float, optional): Crossover rate. Defaults to 0.9.
        max_gen (int, optional): Maximum number of generations. Defaults to 100.
    
    """
    assert surrogate_model is not None, "Surrogate model cannot be None"
    
    self.surrogate_model = surrogate_model
    self.dim_reduction_model = dim_reduction_model
    self.pop_size = pop_size
    self.max_gen = max_gen
    self.crossover_rate = crossover_rate
    self.f = f
    
    self.g = 0
    self.best = None
    self.fitness = None
    
    self.upper = []
    self.lower = []
    self.mean =[]
  
  def initialize_population(self):
    """
    Initialize the population
    """
    self.population = [Chromosome() for _ in range(self.pop_size)]
    self.population = torch.tensor([chromosome.get_real() for chromosome in self.population])

  def evaluate_population(self):
    """
    Evaluate the population
    """
    X_scaled = self.dim_reduction_model.transform(self.population)

    with torch.no_grad():
      self.surrogate_model.eval()
      
      fitness = self.surrogate_model(torch.tensor(X_scaled, dtype=torch.float32).detach().clone())    
      fitness = fitness.detach().clone()
      fitness[fitness < 0] = torch.inf
      self.fitness = fitness
    
  def evaluate_individual(self, individual):
    """
    Evaluate the population
    """
    X_scaled = self.dim_reduction_model.transform(individual)
    with torch.no_grad():
      
      fitness = self.surrogate_model(torch.tensor(X_scaled, dtype=torch.float32).detach().clone())
      fitness = fitness.detach().clone()
      
      if fitness < 0:
        return torch.inf
      return fitness
    
  def start(self):
    print("Starting Differential Evolution")
    # Initialize population
    self.initialize_population()
    
    for g in range(self.max_gen):
      sys.stdout.write(f"\r[Generation {g+1}/{self.max_gen}] - Best fitnesss (loss metric): {max(self.fitness) if self.fitness is not None else 'N/A'}")
      sys.stdout.flush()
      # Iterate for target vector
      self.evaluate_population()
      for i, target in enumerate(self.population):
        # Trial vector
        r1 = torch.argmax(self.fitness)
        r1_score = self.fitness[r1] # For debugg
        r2, r3 = random.sample([num for num in range(self.pop_size) if num != i and num!=r1], k=2)
        v = self.population[r1]#
        d = self.f * (self.population[r2]- self.population[r3])
        v = v + self.f * d
        #Crossover target and trial
        u=[]
        #Select a random index
        idx = random.randint(0, self.surrogate_model.input_dim-1)

        #Uniform crossover
        for f in range(target.shape[0]):
          if random.random() <= self.crossover_rate or f==idx:
            u.append(v[f])
          else:
            u.append(target[f])
        u = torch.tensor(u)

        # Mutation
        for f in range(u.shape[0]):
          if random.random() <= 0.2:
            u[f] += torch.rand(1).item()
            u[f]= max(u[f], 0)
            u[f]= min(u[f], 1)
          else:
            u[f] = target[f]
        
        #Selection
        # for debugg
        u_score = self.evaluate_individual(u.reshape(1, -1))
        target_score = self.evaluate_individual(target.reshape(1, -1))
        if (u_score > target_score):
          self.population[i] = u
          self.fitness[i] = self.evaluate_individual(u.reshape(1, -1))
      
      self.best = self.population[torch.argmax(self.fitness)]
      
      self.g+=1
      epsilon = 1e-15
      
      self.upper.append(torch.max(self.fitness).item())
      self.lower.append(torch.min(self.fitness).item())
      self.mean.append(torch.mean(self.fitness).item())
      

      #Stop conditions
      if(max(self.fitness) > 0.985):
        print(f'\nStopped. Reason: Global Min Found - {max(self.fitness)}')
        break

      elif(abs(max(self.fitness) - min(self.fitness)) < epsilon):
        print('\nStopped. Reason: Diversity loss')
        break

      elif(self.g >= self.max_gen):
        print('\nStopped. Reason: Max Gen Reached')
        break
