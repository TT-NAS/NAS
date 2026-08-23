# Visual Neural Architecture Search Tool

## A tool developed to automatically search for neural network architectures for semantic segmentation. The project uses Neural Architecture Search (NAS) together with bio-inspired optimization algorithms to find architectures that balance segmentation performance and the number of model parameters.

---

### About the project

The project is based on a **U-Net** architecture and allows different characteristics of the network to be modified, such as the number of layers, convolutions, filters, filter size, activation functions, and pooling operations.

Instead of manually designing and testing each architecture, the system uses search algorithms to explore different configurations and select those that achieve better results.

To reduce the cost of evaluating each architecture, a surrogate model was implemented to estimate an architecture's performance without having to fully train it.

---

### Main features

1. **Automatic architecture search:**
   Allows different neural network configurations to be explored in order to find architectures suitable for semantic segmentation.

2. **Genetic Algorithm:**
   Uses a binary representation of the architectures and a genetic algorithm to perform the search.

3. **Differential Evolution:**
   Uses a real-valued representation of the architectures and Differential Evolution as the search strategy.

4. **Performance estimation:**
   The system uses surrogate models to estimate the IoU of architectures without fully training them.

5. **Surrogate model optimization:**
   Different models were evaluated for performance estimation, including SynFlow, a multilayer perceptron, and XGBoost. XGBoost achieved the best results and was subsequently optimized using Optuna.

6. **Training of the selected architecture:**
   Once the search is complete, the selected architecture can be trained.

---

### Datasets

The project was tested using two datasets:

* **Carvana:** used for vehicle segmentation.
* **Road:** a custom dataset used for road segmentation.

---

### [Web application](https://github.com/TT-NAS/NASWeb)

The application provides an interface for configuring the search parameters and viewing the results.

---

### Examples

#### Search configuration

<img width="1234" height="999" alt="3" src="https://github.com/user-attachments/assets/e602399c-02c4-422a-96a6-5eac5ac1277e" />

#### Architecture search

<img width="1234" height="999" alt="4" src="https://github.com/user-attachments/assets/c4a86e86-29fa-4bf3-9602-3726e668dde5" />

#### Training the selected architecture

<img width="1234" height="999" alt="6" src="https://github.com/user-attachments/assets/08b6cf1e-eb20-420e-be3f-1bb9e9b54fb4" />

---

### Installation

#### Dependencies

* [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
* [PyTorch](https://pytorch.org/get-started/locally/). Install `torch`, `torchvision`, and `torchaudio` from the same link.
* Other dependencies:

```bash
pip install -r requirements.txt
```

#### Prerequisites

To obtain the datasets, download the `data/` folder from [this link](https://mega.nz/file/e3hQzbTB#l60DJyVcBs1XezSv4sEJ7QIO1EKhp3QYIEPHUhPza70) and extract it into the root directory. The final directory structure should be:

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

The `pycocotools/` folder inside `src/` contains the API for working with the COCO dataset. If it causes issues, delete the folder, download it from the [official repository](https://github.com/cocodataset/cocoapi/), and reinstall it.

#### Usage

Once the environment has been configured and the datasets have been downloaded, run the `main.py` script to start the NAS tool.

```bash
python src/main.py
```

The architectures found will be saved in the `output/` folder, inside their own directory, using a name chosen by the user or, by default, the date and time of the execution. Each folder contains information such as the architecture encoding, estimated performance, whether the architecture has been trained, and the training results.

---

## Authors

* [Manzano Rios Kevin Uriel](https://github.com/KevinUrielAdler)
* [Moran Orozco Kevin Jafet](https://github.com/Jafet5757)
* [Núñez Castillo Jaime](https://github.com/jnunez54)
