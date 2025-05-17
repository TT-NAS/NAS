# NAS Repo

## Dependencias

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [PyTorch](https://pytorch.org/get-started/locally/). Instalar `torch`, `torchvision` y `torchaudio` desde el mismo link.
- Otras dependencias:

```bash
pip install -r requirements.txt
```

## Instrucciones previas

Para obtener los conjuntos de datos hace falta descargar la carpeta `data/` desde [este link](https://mega.nz/file/e3hQzbTB#l60DJyVcBs1XezSv4sEJ7QIO1EKhp3QYIEPHUhPza70) y descomprimirla en la carpeta raíz. La estructura final del directorio debería ser la siguiente:

```bash
- data/
  ├── car-dataset/
  ├── carvana-dataset/
  ├── coco-dataset-car/
  ├── coco-dataset-people/
  └── road-dataset/
- diagramas/
- results/
- src/
  ├── app/
  ├── codec/
  ├── pycocotools/
  └── ...
- sustituto/
- .gitignore
- README.md
- requirements.txt
```

La carpeta `pycocotools/` dentro de `src/` contiene la api para trabajar con el dataset COCO. En caso de dar problemas borrar la carpeta, descargarla desde el [repositorio oficial](https://github.com/cocodataset/cocoapi/) y reinstalarla.

## Instrucciones de uso

Una vez configurado el entorno y descargados los conjuntos de datos, se puede ejecutar el script `main.py` para iniciar la herramienta de NAS.

```bash
python src/main.py
```

Las arquitecturas encontradas se guardarán en la carpeta `output/` dentro de su propio directorio con un nombre elegido por el usuario, o por defecto la fecha y hora de la ejecución. Dentro de cada carpeta se guardarán datos como la codificación de la arquitectura, el rendimiento estimado, si se ha entrenado o no y los resultados del entrenamiento.
