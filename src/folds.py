import json
import os 
import shutil

from codec import Chromosome
from utils import TorchDataLoader
from Score_models import score_model

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

def print_fold_info(fold, train_subset, val_subset):
    train_indices = train_subset.indices if isinstance(train_subset, Subset) else list(range(len(train_subset)))
    val_indices = val_subset.indices if isinstance(val_subset, Subset) else list(range(len(val_subset)))

    print(f"\nFold {fold + 1}")
    print(f"Train size: {len(train_indices)}")
    print(f"Validation size: {len(val_indices)}")

    
# Obtener los modelos
models = []
for path in os.listdir('./output'):
    path = os.path.join('./output', path)
    path = os.path.join(path, 'model.json')
    # Read model.json
    with open(path, 'r') as f:
        data = json.load(f)
    real_codification = data["real_codification"]
    models.append(real_codification)
    
    
# Crear los folds
data_loader_args, kwargs = TorchDataLoader.get_args({"dataset_len": 1000})
data_loader = TorchDataLoader("carvana", **data_loader_args)
dataset = data_loader.full_dataset

kf = KFold(n_splits=5, shuffle=False)


for i, real_codification in enumerate(models):
    results_data = {"real_codification": real_codification, "folds" : []}
    k_folds = kf.split(dataset)
    for fold, (train_idx, val_idx) in enumerate(k_folds):
        model = Chromosome(chromosome = real_codification)
        if os.path.exists('__checkpoints__'):
            shutil.rmtree('__checkpoints__')
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=4, shuffle=False, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=4, shuffle=False, num_workers=2)
        print(f"[Trainig for] fold: - {fold} train_size: {train_loader.__len__()} val_size: {val_loader.__len__()}")
        results = model.train_unet(data_loader="carvana", k_folds_subsets=(train_subset, val_subset), epochs=15)
        
        results_data["folds"].append(
            {
                "fold": fold,
                "training_iou": results[2]["train_iou"][-1],
                "validation_iou" : results[2]["val_iou"][-1]
            }
        )
        fold_results_path = f'k_fold_results_{i}.json'
        with open(fold_results_path, 'w') as f:
            json.dump(results_data, f, indent=4)
        
        
        
    
    