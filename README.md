# NAS Repo

Descargar la carpeta data de [este link](https://mega.nz/file/e3hQzbTB#l60DJyVcBs1XezSv4sEJ7QIO1EKhp3QYIEPHUhPza70) y descomprimir en la
carpeta data. La estructura final del directorio debería ser la siguiente:

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

La carpeta `pycocotools/` contiene la api para trabajar con el dataset COCO. Es la forma que yo encontré para instalarla, porque no pude instalarla con pip. Si les da problemas borrenla, ponganla en el `.gitignore` e instalenla por su cuenta, [este](https://github.com/cocodataset/cocoapi/)
es el link de la api. Sea como sea que la instalen, no deberían tener que modificar nada en el código para que funcione.

Luego hago un readme chido
