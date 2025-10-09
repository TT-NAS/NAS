# Experimento MeCo para las redes del dataset
import math
from multiprocessing import freeze_support

import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm
tqdm.pandas()

from utils import UNet, TorchDataLoader
from codec import Chromosome
from utils.globals import CUDA

class MeCo():
    def __init__(self):
        self.encoder_hooks = []
        self.decoder_hooks = []
        self.meco_scores = []
        
    def _save_activation_encoder(self, activation):
        self.encoder_hooks.append(activation.detach().cpu())
        
    def _save_activation_decoder(self, activation):
        self.decoder_hooks.append(activation.detach().cpu())

    def _forward_hook_encoder(self, module, inputs, output):
        self._save_activation_encoder(output)
                    
    def _forward_hook_decoder(self, module, inputs, output):
        self._save_activation_decoder(output)
                    
    def _set_hooks(self, model):
        """
        Hook para capturar las salidas de las activaciones
        """
        for i in range(0, len(model.encoder), 2):
            modules = model.encoder[i]
            for j in range(0, len(modules), 3):
                activation = modules[j + 2]
                activation.register_forward_hook(self._forward_hook_encoder)
        
        for i in range(1, len(model.decoder), 2):
            modules = model.decoder[i]
            for j in range(0, len(modules), 3):
                activation = modules[j + 2]
                activation.register_forward_hook(self._forward_hook_decoder)
    
    def forward(self, model: UNet, data_loader: TorchDataLoader):
        model = model.to(CUDA)
        self._set_hooks(model)
        optimizer = torch.optim.SGD(
            model.parameters(),
            momentum=0.95,
            weight_decay=1e-4
        )
        
        for i, (images, masks) in enumerate(data_loader.train):
            images = images.to(CUDA)
            masks = masks.to(CUDA)
            optimizer.zero_grad()
            
            output = model(images)
            break

        self.encoder_hooks = [hook.mean(dim=0).cpu() for hook in self.encoder_hooks[::-1]]
        self.decoder_hooks = [hook.mean(dim=0).cpu() for hook in self.decoder_hooks[::-1]]

    def get_pearson_matrix(self):
        # Aplanar los mapas de características (sobre la dimensión 1 y 2)
        flat_fm_encoder = [activation.view(activation.size(0), -1) for activation in self.encoder_hooks]
        flat_fm_decoder = [activation.view(activation.size(0), -1) for activation in self.decoder_hooks]
        
        # Para cada feature map, obtener una matriz de correlación de Pearson
        for activation in flat_fm_encoder:
            corr_matrix = torch.corrcoef(activation)
            if corr_matrix.ndim < 2:
                continue
            n = corr_matrix.shape[0]
            try:
                U, S, V = torch.svd(corr_matrix)
            except:
                continue

            # Los valores singulares al cuadrado son los valores propios
            eigenvalues = S**2
            min_eigenvalue = torch.min(eigenvalues).item()
            max_eigenvalue = torch.max(eigenvalues).item()
            second_max = torch.topk(eigenvalues, 2).values[1].item()

            diff = max_eigenvalue - second_max
            self.meco_scores.append(diff)

        for activation in flat_fm_decoder:
            corr_matrix = torch.corrcoef(activation)
            if corr_matrix.ndim < 2:
                continue
            n = corr_matrix.shape[0]
            try:
                U, S, V = torch.mean(corr_matrix)
            except:
                continue
    
            # Los valores singulares al cuadrado son los valores propios
            eigenvalues = S**2
            min_eigenvalue = torch.min(eigenvalues).item()
            max_eigenvalue = torch.max(eigenvalues).item()
            second_max = torch.topk(eigenvalues, 2).values[1].item()

            diff = max_eigenvalue - second_max
            self.meco_scores.append(diff)


def main():
    df = pd.read_csv("./results/results_filtered.csv") 
    # MeCo funciona con un solo dato
    data_loader = TorchDataLoader("carvana", dataset_len = 4, batch_size=1)
    real_scores = df['val_iou'].values
    predictor_scores = []
    for index, row in tqdm(df.iterrows(), desc="Evaluando redes", total=df.shape[0]):
        c = Chromosome(chromosome=row['binary codification'])
        model = c.get_unet()
        
        meco = MeCo()
        meco.forward(model, data_loader)
        
        # Scores MeCo
        meco.get_pearson_matrix()
        
        # Graficar los puntajes
        plt.figure(figsize=(10, 5))
        plt.plot(meco.meco_scores, label='MeCo Scores')
        plt.xlabel('Layer')
        plt.ylabel('Score')
        plt.title(f'MeCo Scores for Model {row["val_iou"]}')
        plt.legend()
        plt.savefig(f"./meco_scores_layer/meco_scores_model_{row['val_iou']}.png")
        plt.close()
        # Obtener un putaje a partir de los Scores MeCo
        final_score = 0
        for score in meco.meco_scores:
            if math.isnan(score):
                continue
            final_score += score
        #final_score /= len(meco.meco_scores)
        predictor_scores.append(1 - final_score)

    # Correlaciones
    pearson_corr, _ = pearsonr(real_scores, predictor_scores)
    spearman_corr, _ = spearmanr(real_scores, predictor_scores)
    print(f"Pearson Correlation: {pearson_corr}")
    print(f"Spearman Correlation: {spearman_corr}")

    # Graficar resultados
    plt.figure(figsize=(10, 5))
    plt.plot(real_scores, label='Real Scores')
    plt.plot(predictor_scores, label='Predicted Scores')
    plt.xlabel('Sample')
    plt.ylabel('IoU Score')
    plt.title('Real vs Predicted IoU Scores')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    freeze_support()
    main()