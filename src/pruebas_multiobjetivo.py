from search_algorithms.multi_objective import DiferentialEvolution
from codec import Chromosome
from multiprocessing import freeze_support
import numpy as np
def main():
    differential_evolution = DiferentialEvolution(base="random", n_differences=1, crossover="bin")
    differential_evolution.start(n_pop=50, max_gen=10)

    for fitness, pareto_fronts in zip(differential_evolution.fitness_history, differential_evolution.pareto_history):
        print(f"Fitness: {fitness}")
        print(f"Pareto fronts: {pareto_fronts}")
        differential_evolution.plot_pareto_fronts(fitness, pareto_fronts)
        
    for individual in differential_evolution.pareto_fronts[0]:
            print(f"Individual: {differential_evolution.offspring[individual]}")
            print(f"Fitness: {fitness[individual]}")

    # Obtener el que tiene menor número de parámetros (objetivo 2)
    idx_min_obj2 = np.argmax(fitness[:, 1])

    chromosome = list(differential_evolution.offspring[idx_min_obj2])
    c = Chromosome(chromosome=chromosome)
    # Número de parámetros
    unet = c.get_unet()
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {trainable_params}")
    c.train_unet(data_loader="carvana", dataset_len = 1000, batch_size = 16, epochs=50)
    c.show_results(data_loader="carvana", path="./MO_Results")
    c.show_results(data_loader="car", path="./MO_Results_car")

if __name__ == "__main__":
    freeze_support()
    main()