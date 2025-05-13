from search_algorithms import SurrogateModel, DiferentialEvolution
import matplotlib.pyplot as plt

surrogate_model = SurrogateModel(model_path = r"./sustituto/xgboost_model.json")
de = DiferentialEvolution(surrogate_model=surrogate_model, pop_size=100)
de.start()

plt.plot(de.lower, label="Lower")
plt.plot(de.upper, label="Upper")
plt.plot(de.mean, label="Mean")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Differential Evolution Fitness")
plt.legend()
plt.show()