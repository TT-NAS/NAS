# Preprocesamiento de los datasets para los experimentos (Vehículos y caminos obtenidos de CARLA)
from src.utils import ROAD_DATA_PATH, CAR_DATA_PATH
from src.utils import TorchDataLoader

def preprocess_dataset(name, data_path):
    print(f"\nPreprocesando dataset: {name}")
    data_loader = TorchDataLoader(name, data_path=data_path)
    
    train = data_loader.train
    data_train = next(iter(train))
    val = data_loader.validation
    data_val = next(iter(val))
    test = data_loader.test
    data_test = next(iter(test))
    
    print(f"Datos cargados para {name}:")
    print(
        f"Train: {len(train)},\t length: {len(data_train)},\t shape: {data_train[0].shape}"
    )
    print(
        f"Val:   {len(val)},\t length: {len(data_val)},\t shape: {data_val[0].shape}"
    )
    print(
        f"Test:  {len(test)},\t length: 1,\t shape: {data_test.shape}"
    )

def main():
    preprocess_dataset("road", ROAD_DATA_PATH)
    preprocess_dataset("car", CAR_DATA_PATH)

if __name__ == "__main__":
    main()