from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Annotated
from fastapi import Body

from codec import Chromosome
from search_algorithms.de_search import DiferentialEvolution
from search_algorithms.surrogate import SurrogateModel

from utils import save_pickle

app = FastAPI(title = "NAS API", version = "1.0.0")
surrogate_model = SurrogateModel(model_path = r"./sustituto/xgboost_model.json")
file_path = "models"

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



@app.get("/", summary="Health check", response_description="Greeting payload confirming the API is reachable.")
def root():
  """Return a greeting so clients can verify the API is online."""
  return {"Hello": "world"}


@app.post("/download-model", summary="Generate and download UNet pickle", response_description="Binary pickle containing the UNet encoded by the chromosome.")
def download_model(chromosome: Annotated[list[float], Body(..., embed=True)]):
  """Create the UNet for the chromosome, persist it, and serve the pickle file for download."""
  model = Chromosome(chromosome=chromosome)

  save_pickle(model.get_unet(), file_path, model.get_binary(zip=True))

  return FileResponse(
    path=f"{file_path}/{model.get_binary(zip=True)}.pkl",
    media_type="application/octet-stream",
    filename=model.get_binary(zip=True) + ".pkl"
  )

# @app.post("/search")
# def run_search(params: SearchParams):
#   # Realiza la búsqueda
#   de = DiferentialEvolution(surrogate_model, **params.model_dump())
#   fitness_register = de.start()
#   vector = [random.randint(int(i/2), i) for i in range(100)]
#   vector.reverse()
#   json_data = {
#         "search_time": de.search_time,
#         "stop_reason": de.stop_reason,
#         "stop_gen": de.g,
#         "real_codification": de.best.tolist(),
#         "predicted_iou": float(de.best_fitness),
#         "trained": False,
#         "vector": fitness_register
#     }
#   # Retorna los resultados
#   result = {"params": params, "results": json_data}
#   return result

@app.post("/search")
async def run_search(params: SearchParams):
  # Realiza la búsqueda
  de = DiferentialEvolution(surrogate_model, **params.model_dump())

  return StreamingResponse(
    de.start(),
    media_type="text/event-stream"
  )

  # El streaming utiliza protocolo SSE, para leerlo en javaScript:
  # const source = new EventSource("/search");

  # source.addEventListener("iteration", (event) => {
  #   const data = JSON.parse(event.data);
  #   console.log("Iteración:", data.generation, data.best_fitness);
  # });

  # source.addEventListener("result", (event) => {
  #   const data = JSON.parse(event.data);
  #   console.log("Resultado final:", data);
  # });


# @app.post("/train")
# def train_network(args: TrainingArg):
#   model = Chromosome(chromosome=args.chromosome)
#   # Se entrena
#   results  = model.train_unet(data_loader = args.data_loader, dataset_len = args.dataset_len, epochs = args.epochs)

#   register = {
#     "training_time": results[0],
#     "last_epoch": results[1] + 1,
#     "training_iou": results[2]["train_iou"][-1],
#     "validation_iou": results[2]["val_iou"][-1]
#   }

#   file_path = "models"

#   save_pickle(model.get_unet(), file_path, model.get_binary(zip=True))

#   return {
#     "register": register,
#     "pickle_url": f"/download/{model.get_binary(zip=True)}"
#   }

@app.post("/train")
async def train_network(args: TrainingArg):
  model = Chromosome(chromosome=args.chromosome)
  # Se entrena
  return StreamingResponse(
      model.train_unet_stream(
          save_model_pickle=True,
          data_loader=args.data_loader,
          dataset_len=args.dataset_len,
          epochs=args.epochs),
      media_type="text/event-stream"
  )

  # El streaming utiliza protocolo SSE, para leerlo en javaScript:
  # const source = new EventSource("/train");

  # source.onmessage = (event) => {
  #     const data = JSON.parse(event.data); // parsear JSON
  #     console.log("Progreso de entrenamiento:", data);
  # };

@app.get("/download/{name}",summary="Download stored UNet",response_description="Binary pickle for the requested UNet filename.")
def download_file(name: str):
  """Serve the stored UNet pickle previously generated by the training or download endpoints."""
  return FileResponse(
    path=f"models/{name}.pkl",
    media_type="application/octet-stream",
    filename=f"{name}.pkl"
  )

@app.post("/json",summary="Export chromosome JSON",response_description="JSON payload describing the UNet architecture encoded by the chromosome.")
def get_json_chromosome(chromosome: Annotated[list[float], Body(..., embed=True)]):
  """Return the JSON representation of the chromosome for consumption by external tools."""
  model = Chromosome(chromosome=chromosome)
  return model.get_json()


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="", port=8000)
