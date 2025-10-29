import pickle

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Annotated
from fastapi import Body

from codec import Chromosome
from search_algorithms.de_search import DiferentialEvolution
from search_algorithms.surrogate import SurrogateModel
import random

from utils import save_pickle

app = FastAPI(title = "NAS API", version = "1.0.0")
surrogate_model = SurrogateModel(model_path = r"./sustituto/xgboost_model.json")

class SearchParams(BaseModel):
  pop_size: int = Field(100, ge=10, le=100, description="Tamaño de la población")
  f: float = Field(0.9, ge=0.1, le=1.0, description="Factor de escala del diferencial")
  crossover_rate: float = Field(0.9, ge=0, le=1.0, description="Probabilidad de cruce")
  mutation_rate: float = Field(0.2, ge=0, le=1.0, description="Tasa de mutación")
  max_gen: int = Field(100, ge=2, le=1000, description="Número máximo de generaciones")

class TrainingArg(BaseModel):
  chromosome: list = Field(description="Codificación real del cromosoma")
  data_loader: str = Field("car", description="Dataset que va a utilizar para entrenar")
  dataset_len: int = Field(500, ge = 100, le = 1000, description="Tamaño del dataset de entrenamiento")
  epochs: int = Field(15, ge = 1, le = 30, description="Cantidad de epocas de entrenamiento")

@app.get("/")
def root():
  return {"Hello": "world"}


@app.get("/download-model")
def download_model(chromosome: Annotated[list[float], Body(..., embed=True)]):
  model = Chromosome(chromosome=chromosome)

  file_path = "models"

  save_pickle(model.get_unet(), file_path, model.get_binary(zip=True))

  return FileResponse(
    path=f"{file_path}/{model.get_binary(zip=True)}.pkl",
    media_type="application/octet-stream",
    filename=model.get_binary(zip=True) + ".pkl"
  )

@app.post("/search")
def run_search(params: SearchParams):
  # Realiza la búsqueda
  de = DiferentialEvolution(surrogate_model, **params.model_dump())
  fitness_register = de.start()
  vector = [random.randint(int(i/2), i) for i in range(100)]
  vector.reverse()
  json_data = {
        "search_time": de.search_time,
        "stop_reason": de.stop_reason,
        "stop_gen": de.g,
        "real_codification": de.best.tolist(),
        "predicted_iou": float(de.best_fitness),
        "trained": False,
        "vector": fitness_register
    }
  # Retorna los resultados
  result = {"params": params, "results": json_data}
  return result

@app.post("/train")
def train_network(args: TrainingArg):
  model = Chromosome(chromosome=args.chromosome)
  # Se entrena
  results  = model.train_unet(data_loader = args.data_loader, dataset_len = args.dataset_len, epochs = args.epochs)

  register = {
    "training_time": results[0],
    "last_epoch": results[1] + 1,
    "training_iou": results[2]["train_iou"][-1],
    "validation_iou": results[2]["val_iou"][-1]
  }

  file_path = "models"

  save_pickle(model.get_unet(), file_path, model.get_binary(zip=True))

  return {
    "register": register,
    "pickle_url": f"/download/{model.get_binary(zip=True)}"
  }

  # return register


@app.get("/download/{name}")
def download_file(name: str):
  return FileResponse(
    path=f"models/{name}.pkl",
    media_type="application/octet-stream",
    filename=f"{name}.pkl"
  )

@app.post("/json")
def get_json_chromosome(chromosome: Annotated[list[float], Body(..., embed=True)]):
    model = Chromosome(chromosome=chromosome)
    return model.get_json()


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="", port=8000)
