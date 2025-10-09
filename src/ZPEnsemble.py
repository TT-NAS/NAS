from multiprocessing import freeze_support
from ZeroShotProxy.gradnorm import compute_nas_score as compute_gradnorm_score
from ZeroShotProxy.synflow import compute_nas_score as compute_synflow_score
from ZeroShotProxy.tenas import compute_RN_score, compute_NTK_score

from utils import TorchDataLoader
from codec import Chromosome
import pandas as pd
from tqdm import tqdm

def main():
    dataframe = pd.read_csv("results/selected_networks.csv")
    dataframe["gradnorm"] = 0.0
    dataframe["synflow"] = 0.0
    dataframe["RN"] = 0.0
    dataframe["NTK"] = 0.0

    for idx, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0], desc="Calculando scores"):
        c = Chromosome(chromosome=row["binary codification"])
        model = c.get_unet()

        data_loader = TorchDataLoader("carvana")
        gradnorm_score = compute_gradnorm_score(model=model, data_loader=data_loader)
        print(f"GradNorm score: {gradnorm_score}")
        synflow_score = compute_synflow_score(model=model, data_loader=data_loader)
        print(f"SynFlow score: {synflow_score}")
        #RN_score = compute_RN_score(model=model, batch_size=4, image_size=512, num_batch=1, gpu=0)
        NTK_score = compute_NTK_score(model=model, data_loader=data_loader)
        print(f"NTK score: {NTK_score}")
        
        dataframe.at[idx, "gradnorm"] = gradnorm_score
        dataframe.at[idx, "synflow"] = synflow_score
        dataframe.at[idx, "NTK"] = NTK_score

        dataframe.to_csv("results/selected_networks_scored.csv", index=False)
        if idx == 100:
            break
        
        # drop zeros
    dataframe.dropna(inplace=True)
    dataframe.to_csv("results/selected_networks_scored.csv", index=False)
if __name__ == "__main__":
    freeze_support()
    main()