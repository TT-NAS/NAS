## Experimentos para las activaciones zero shot (MeCo)
import pandas as pd
from codec import Chromosome
from tqdm import tqdm
tqdm.pandas()

df = pd.read_csv("./results/results_completos.csv")
df.dropna(inplace=True, axis=1)

# Para cada red, obtener su número de parámetros
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

df['num_parameters'] = 0
for index, row in tqdm(df.iterrows(), desc="Counting params", total=df.shape[0]):
    c = Chromosome(chromosome=row['binary codification'])
    model = c.get_unet()
    num_params = count_parameters(model)
    df.at[index, 'num_parameters'] = num_params

df.to_csv("./results/results_with_params.csv")