from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Union
from fastapi import Body
import json

from codec import Chromosome
from search_algorithms.de_search import DiferentialEvolution
from search_algorithms.surrogate import SurrogateModel
from search_algorithms.evaluator import CombinedMetricEvaluator
from search_algorithms.mono_objective import DifferentialEvolution, GeneticAlgorithm

from utils import save_pickle

app = FastAPI(title = "NAS API", version = "1.0.0")
surrogate_model = SurrogateModel(model_path = r"./sustituto/xgboost_model.json")
file_path = "models"

class SearchParams_ant(BaseModel):
  population_size: int = Field(100, ge=10, le=100, description="Tamaño de la población")
  f: float = Field(0.9, ge=0.1, le=1.0, description="Factor de escala del diferencial")
  crossover_rate: float = Field(0.9, ge=0, le=1.0, description="Probabilidad de cruce")
  mutation_rate: float = Field(0.2, ge=0, le=1.0, description="Tasa de mutación")
  generations: int = Field(100, ge=2, le=100, description="Número máximo de generaciones")


class DEParams(BaseModel):
  algorithm: Literal["de"] = Field("de")
  dataset: str = Field(
      "carvana", description="Dataset to use: 'carvana' or 'road'"
  )

  n_pop: int = Field(25, ge=10, le=100, description="Tamaño de la población")
  max_gen: int = Field(50, ge=2, le=100, description="Número máximo de generaciones")
  F: float = Field(0.5, ge=0.1, le=1.0, description="Factor de escala del diferencial")
  crossover_rate: float = Field(0.9, ge=0.0, le=1.0, description="Probabilidad de cruce")


class GAParams(BaseModel):
  algorithm: Literal["ga"] = Field("ga")
  dataset: str = Field(
      "carvana", description="Dataset to use: 'carvana' or 'road'")

  n_pop: int = Field(25, ge=10, le=100, description="Tamaño de la población")
  max_gen: int = Field(50, ge=2, le=100, description="Número máximo de generaciones")
  mutation_rate: float = Field(0.2, ge=0.0, le=1.0, description="Tasa de mutación")
  crossover_rate: float = Field(0.8, ge=0.0, le=1.0, description="Probabilidad de cruce")

SearchParams = Union[DEParams, GAParams]

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


@app.post("/search_ant")
async def run_search_ant(params: SearchParams_ant):
  de = DiferentialEvolution(
    surrogate_model,
    pop_size=params.population_size,
    f=params.f,
    crossover_rate=params.crossover_rate,
    mutation_rate=params.mutation_rate,
    max_gen=params.generations
  )

  async def stream_results():
    async for result in de.start():
      # Serializa cada diccionario como JSON + salto de línea
      yield json.dumps(result) + "\n"

  return StreamingResponse(stream_results(), media_type="application/json")

@app.post("/search")
async def run_search(params: SearchParams):
  evaluator = CombinedMetricEvaluator(
    codification="real" if params.algorithm == "de" else "binary",
    dataset=params.dataset,
    beta=0.837 if params.dataset == "carvana" else 0.79,
  )

  search_params = params.model_dump(exclude={"algorithm", "dataset"})

  if params.algorithm == "de":
    search = DifferentialEvolution(
      base="random",
      n_differences=1,
      crossover="bin"
    )
    search.evaluator = evaluator
  else:  # GA
    search = GeneticAlgorithm(
      selection="tournament",
      crossover="uniform"
    )
    search.evaluator = evaluator

  async def stream_results():
    async for result in search.start(
      **search_params,
      diversity_min=0.01,
      target_fitness=0.846 if params.dataset == "carvana" else 0.79
    ):
      # Serializa cada diccionario como JSON + salto de línea
      yield json.dumps(result) + "\n"

  return StreamingResponse(stream_results(), media_type="application/json")


@app.post("/train")
async def train_network(args: TrainingArg):
  try:
    # Valida el cromosoma
    Chromosome(chromosome=args.chromosome)
  except Exception as e:
    return {"error": f"Cromosoma inválido: {str(e)}"}

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


@app.get("/download/{name}",summary="Download stored UNet",response_description="Binary pickle for the requested UNet filename.")
def download_file(name: str):
  """Serve the stored UNet pickle previously generated by the training or download endpoints."""
  return FileResponse(
    path=f"models/{name}.pkl",
    media_type="application/octet-stream",
    filename=f"{name}.pkl"
  )

@app.get("/image_results/{name}",summary="Download results from training",response_description="Image file showing training results for the requested UNet filename.")
def download_results(name: str):
  """Serve the stored training results image previously generated by the training endpoint."""
  return FileResponse(
    path=f"models/{name}.png",
    media_type="image/png",
    filename=f"{name}.png"
  )

@app.post("/json",summary="Export chromosome JSON",response_description="JSON payload describing the UNet architecture encoded by the chromosome.")
def get_json_chromosome(chromosome: Annotated[list[float], Body(..., embed=True)]):
  """Return the JSON representation of the chromosome for consumption by external tools."""
  model = Chromosome(chromosome=chromosome)
  return model.get_json()


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="", port=8000)
