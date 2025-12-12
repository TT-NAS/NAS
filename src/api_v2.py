from search_algorithms.evaluator import CombinedMetricEvaluator
from search_algorithms.mono_objective import DifferentialEvolution, GeneticAlgorithm
from codec import Chromosome
import numpy as np

# -- [Configuración del evaluador] --
# [Dataset carvana] - beta: 0.837 - target_fitness = 0.846
# [Dataset road] - beta: 0.79 - target_fitness = 0.79
# [Para evolución diferencial] - codification: "real" 
# [Para algoritmo genético] - codification: "binary" 
evaluator = CombinedMetricEvaluator(
    codification = "real", # codification = "binary"
    dataset = "carvana", # "road"
    beta = 0.837, # 0.79
)

# -- [Configuración del Evolución Diferencial] --
# Variante ganadora en las pruebas
de = DifferentialEvolution(
    base = "random",
    n_differences = 1,
    crossover = "bin"
)

# -- [Configuración del Algoritmo genético] --
# Variante ganadora en las pruebas
ga = GeneticAlgorithm(
    selection = "tournament",
    crossover = "uniform"
)

# Ejecución de los algoritmos (valores utiliizados en las pruebas)
de.start(
    n_pop = 25,
    max_gen = 50,
    F = 0.5,
    crossover_rate = 0.9,
    diversity_min = 0.01,
    target_fitness=0.79 # Depende del dataset, 0.846 para carvana y 0.79 para road
)

ga.start(
    n_pop = 25,
    mutation_rate = 0.2,
    crossover_rate = 0.8,
    crossover_rate = 0.9,
    diversity_min = 0.01,
    target_fitness=0.79 # Depende del dataset, 0.846 para carvana y 0.79 para road
)

# Métricas de los algoritmos

# DE
chromosome = Chromosome(chromosome=de.population[np.argmax(de.fitness)].tolist()).get_binary(zip=True)
row = {
    "generations": de.gen,
    "diversity": de.diversity[-1],
    "best_fitness": np.max(de.fitness),
    "reached_target": de.reached_target,
    "reached_diversity_loss": de.diversity_loss,
    "reached_max_gens": de.reached_gens,
    "Chromosome": chromosome,
}

# GA
chromosome = Chromosome(chromosome=chromosome).get_binary(zip=True)
row = {
        "generations": ga.gen,
        "diversity": ga.diversity[-1],
        "best_fitness": np.max(ga.fitness),
        "reached_target": ga.reached_target,
        "reached_diversity_loss": ga.diversity_loss,
        "reached_max_gens": ga.reached_gens,
        "Chromosome": chromosome,
    }