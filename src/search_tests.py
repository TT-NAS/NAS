from search_algorithm import dim_reduction, SurrogateModel
from search_algorithm import DiferentialEvolution
from codec import Chromosome

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr, kendalltau
from Score_models import score_model

df, pls = dim_reduction(10)
X = np.stack(df["X_reduced"].values)
Y = np.array(df["iou"].values)
Y = Y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = SurrogateModel(input_dim=X.shape[1], output_dim=1)
model.fit(X_train, y_train,X_test, y_test, epochs=500, learning_rate=0.01)

model.eval()
X_test = torch.tensor(X_test, dtype=torch.float32).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
y_pred = model(X_test)
y_pred = y_pred.cpu().detach().numpy()


# Calculate metrics
spearmanr_corr, _ = spearmanr(y_test, y_pred)
kendalltau_corr, _ = kendalltau(y_test, y_pred)
pearson_corr, _ = pearsonr(y_test.flatten(), y_pred.flatten())

print(f"\n[Spearman] - {spearmanr_corr}")
print(f"[Kendalltau] - {kendalltau_corr}")
print(f"[Pearson] - {pearson_corr}")

# Plot
plt.plot(y_test, label="Real")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.savefig("predictions_plot.png")
plt.clf()

# Scatter
plt.scatter(y_test, y_pred)
plt.xlabel("Real")
plt.ylabel("Predicted")
plt.title("Real vs Predicted")
plt.savefig("predictions_scatter.png")
plt.clf()

# Search algorithm
de = DiferentialEvolution(surrogate_model=model, dim_reduction_model=pls, pop_size=100, f=0.2, crossover_rate=0.9, max_gen=100)
de.start()

plt.plot(de.upper, label='Upper')
plt.plot(de.lower, label='lower')
plt.plot(de.mean, label='Mean')

plt.xlabel('Generation')
plt.ylabel('Evaluation')
plt.title('Convergence')
plt.legend()
plt.savefig("convergence_plot.png")
plt.clf()
      
print(de.best.tolist())
trained = score_model(dataset="carvana",
            chromosome=de.best.tolist(),
            save_model=True,
            alternative_datasets=["car",],
            dataset_len=100)

print(f"Trained model: {trained}")