"""Mini-surrogado parra pruebas del algoritmo de búsqueda"""
import os
import sys

import numpy as np
import pandas as pd

from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import codec
from codec import constants

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def pearson_corr(y_true, y_pred):
    vx = y_pred - torch.mean(y_pred)
    vy = y_true - torch.mean(y_true)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
    return corr.item()


def dim_reduction(df, n_components):
    # 1. Cargar el conjunto de datos
    results = os.listdir("results")
    df = pd.DataFrame()

    for file in results:
        if file.endswith(".csv"):
            df_temp = pd.read_csv(f"results/{file}")
            df = pd.concat([df, df_temp], ignore_index=True)
    # 2. Preprocesar los datos (Agregar la codificación real al dataframe)
    df["real_codification"] = None
    for index, row in df.iterrows():
        chromosome = row["binary codification"]
        c = codec.Chromosome(max_layers = constants.MAX_LAYERS, max_convs_per_layer = constants.MAX_CONVS_PER_LAYER, chromosome = chromosome)
        c_real = c.get_real()
        df.at[index, "real_codification"] = c_real

    # 2.1. Aplicar PLS Regression (Reducción supervisada de dimensionalidad)
    X = np.stack(df["real_codification"].values)
    Y = df["iou"]
    Y = np.array(Y)
    Y = Y.reshape(-1, 1)  

    pls = PLSRegression(n_components=n_components)
    X_reduced = pls.fit_transform(X, Y)[0]  # [0] regresa X transformado

    # 2.2. Guardar X_reduced en el dataframe
    df["X_reduced"] = None
    for index, row in df.iterrows():
        df.at[index, "X_reduced"] = X_reduced[index]
    
    return df, pls

# Modelo surrogado (MLP)
class SurrogateModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SurrogateModel, self).__init__()
        self.dropout = nn.Dropout(0.1)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(self.input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 32)
        self.fc4 = nn.Linear(32, output_dim)
        
        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, x):
        x = x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = F.relu(self.fc3(x))
        #x = self.dropout(x)
        x = self.fc4(x)
        return x
    
    def fit(self, X, y, X_test, y_test, epochs=1000, learning_rate=0.001):
        # Convertir los datos a tensores de PyTorch
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        X_tensor = X_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        y_tensor = y_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)
        X_test = X_test.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        y_test = y_test.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Definir el optimizador y la función de pérdida
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion=nn.MSELoss()

        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        # Entrenar el modelo
        train_losses = []
        test_losses = []
        pearson = 0
        trigger_times = 0
        loss = None    
        best_score = -float('inf')
        patience = 50
        trigger_times = 0
        
        for epoch in range(epochs):
            sys.stdout.write(f"\rEpoch {epoch+1}/{epochs} - Loss: {loss.detach().item() if loss is not None else 'N/A'} - Pearson: {pearson} - Best: {best_score} - Trigger times: {trigger_times}")
            sys.stdout.flush()
            self.train()

            optimizer.zero_grad()        
            outputs = self(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Evaluar en el conjunto de prueba
            self.eval()
            with torch.no_grad():
                test_outputs = self(X_test)
                test_loss = criterion(test_outputs, y_test)
                test_losses.append(test_loss.item())
                pearson = pearson_corr(y_test, test_outputs)
            #scheduler.step(pearson)   
            
            # Early Stopping
            
            if pearson > 0.5:
                break
            
            # Plot
            plt.plot(train_losses, label="Train Loss")
            plt.plot(test_losses, label="Test Loss")
            
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Loss vs Epochs")
            
            plt.legend()
            plt.savefig("loss_plot.png")
            plt.clf()
            
            
            
            
            
