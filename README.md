# NAS Repo

## Dependencias

- [PyTorch](https://pytorch.org/get-started/locally/), instalar los 3 (torch, torchvision y torchaudio) con CUDA si es posible.
- Otras dependencias:

```bash
pip install numpy
pip install pandas
pip install matplotlib
pip install pillow
pip install colorama
```

## Instrucciones previas

Descargar la carpeta `data/` de [este link](https://mega.nz/file/e3hQzbTB#l60DJyVcBs1XezSv4sEJ7QIO1EKhp3QYIEPHUhPza70) y descomprimir en la
carpeta raíz. La estructura final del directorio debería ser la siguiente:

```bash
- data/
  ├── car-dataset/
  ├── carvana-dataset/
  ├── coco-dataset-car/
  ├── coco-dataset-people/
  └── road-dataset/
- results/
- src/
  ├── codec/
  ├── pycocotools/
  ├── utils/
  ├── Pruebas.py
  └── Score_models.py
- .gitignore
- README.md
```

La carpeta `pycocotools/` contiene la api para trabajar con el dataset COCO. Ponerla en el directorio `src/` es la única forma que yo encontré
para instalarla, porque no pude instalarla con pip. Si les da problemas borrenla, ponganla en el `.gitignore` e instalenla por su cuenta,
[este es el link de la api](https://github.com/cocodataset/cocoapi/). Sea como sea que la instalen, no deberían tener que modificar nada en
el código para que funcione.

## Instrucciones de uso

Para empezar a entrenar se debe correr el script `Score_models.py`. Pero antes asegurense de ajustar los parámetros de la función `score_n_models`
en el main para indicar desde que semilla se va a empezar a entrenar (`idx_start`), y cuantos modelos se van a entrenar (`num`).

```python
if __name__ == "__main__":
    score_n_models(
        idx_start=0,
        num=10,
        alternative_datasets=["carvana", "car"]
    )
```

Los demás parámetros pueden dejarlos intactos. Luego simplemente corran el script y esperen a que termine. Los resultados se guardarán
en la carpeta `results/`.
