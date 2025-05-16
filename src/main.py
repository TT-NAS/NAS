from datetime import datetime
import json
import os
from time import sleep

from colorama import init, Fore, Style, Back
import numpy as np

from codec import Chromosome
from search_algorithms.de_search import DiferentialEvolution
from search_algorithms.surrogate import SurrogateModel
from app import ascii_art, params
init(autoreset=True)

surrogate_model = SurrogateModel(model_path = r"./sustituto/xgboost_model.json")
    
def get_input(valid_options):
    while True:
        try:
            value = int(input())
            if value in valid_options:
                return value
            else:
                colorama_print("Opción no disponible. Seleccione una opción válida: ", Back.RESET, Fore.RED)
                
        except ValueError:
            colorama_print("Entrada inválida. Seleccione una opción válida: ", Back.RESET, Fore.RED)
            
            
def colorama_print(text, bg_color=Back.RESET, text_color=Fore.RESET):
    print(f"{bg_color}{text_color}{text}{Style.RESET_ALL}", end='')

def display_options(options):
    for i, option in enumerate(options):
        
        colorama_print(f"{'\t' if i > 0 else ''}[{i + 1}]", 
                       Back.GREEN, Fore.RESET)
        colorama_print(f" {option}{'\n' if i == len(options)-1 else ' '}", Back.RESET, Fore.RESET)

def display_header(title="MENU PRINCIPAL"):
    os.system('cls' if os.name == 'nt' else 'clear')
    colorama_print(ascii_art + "\n", text_color=Fore.GREEN)
    colorama_print("CLI - Herramienta para Búsqueda de Arquitecturas Neuronales (NAS TOOL)\n\n", Back.RESET, Fore.GREEN)
    colorama_print(f"[{title}]", Back.GREEN, Fore.RESET)
    print("\n"*5)

def main_menu():
    display_header()

    display_options([
    "Iniciar Búsqueda de Arquitectura",
    "Mostrar o entrenar redes guardadas",
    "Salir"
    ])
    colorama_print("\nSeleccione una opción:", Back.GREEN, Fore.RESET)
    colorama_print(" ", Back.RESET, Fore.RESET)
    return get_input([1, 2, 3])

def architecture_search_menu():
    display_header("BÚSQUEDA DE ARQUITECTURA")
    display_options([
        "Ejecutar con parámetros por defecto",
        "Ingresar parámetros manualmente",
        "Regresar"
    ])
    colorama_print("\nSeleccione una opción:", Back.GREEN, Fore.RESET)
    colorama_print(" ", Back.RESET, Fore.RESET)
    return get_input([1, 2, 3])

def save_results(name, de):
    # Guardar en un json el best fitness y un string con la codificación real
    os.makedirs(r"./output/" + name, exist_ok=True)
    
    json_data = {
        "real_codification": de.best.tolist(),
        "predicted_iou": float(de.best_fitness),
        "trained": False
    }
    path = os.path.join(r"./output", name, "model.json")
    with open(path, 'w') as f:
        json.dump(json_data, f, indent=4)
    return path

def train_network(path):
    with open(path, 'r') as f:
        data = json.load(f)
    real_codification = data["real_codification"]
    if data["trained"]:
        colorama_print("La red ya ha sido entrenada.\n", Back.YELLOW, Fore.RESET)
        return True
    # Entrenar el modelo
    try:
        model = Chromosome(chromosome=real_codification)
        model.train_unet(data_loader="carvana", dataset_len=1000, epochs=15)
        model.show_results(data_loader="carvana", dataset_len = 32, 
                           path=path.replace("model.json", ""), save=True, name="test_results")
    except KeyboardInterrupt:
        colorama_print("\nEntrenamiento interrumpido por el usuario. Presione Enter para continuar...", Back.RED, Fore.RESET)
        
        return False
    except Exception as e:
        colorama_print(f"Error al entrenar la red: {e}\n", Back.RED, Fore.RESET)
        colorama_print("Presione Enter para continuar...\n", Back.RESET, Fore.GREEN)
        input()
        return False
    
    data["trained"] = True
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    colorama_print("Resultados guardados.\n", Back.GREEN, Fore.RESET)
    input()
    return True
    
def default_search():
    display_header("EJECUTANDO BÚSQUEDA DE ARQUITECTURA CON PARÁMETROS POR DEFECTO")
    de = DiferentialEvolution(surrogate_model)
    de.start()
    colorama_print("Búsqueda de arquitectura completada", Back.GREEN, Fore.RESET)
    colorama_print("\nIntroduzca el nombre para guardar los resultados (o presione Enter para usar la fecha y hora actuales): ", Back.RESET, Fore.GREEN)
    name = input()
    if name == "":
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    saved_path = save_results(name, de)
    colorama_print(f"Resultados guardados en output/{name}\n", Back.RESET, Fore.GREEN)
    # Preguntar si se debe entrenar la red  
    display_options([
        "Entrenar la red",
        "Continuar"
    ])
    training = get_input([1, 2])
    if training == 1:
        colorama_print("Entrenando la red...\n", Back.GREEN, Fore.RESET)
        # Entrenar la red
        train_network(saved_path)
        
        input()
    else:
        colorama_print("Continuando sin entrenar la red...\n", Back.GREEN, Fore.RESET)
        colorama_print("Presione Enter para continuar...\n", Back.RESET, Fore.GREEN)
        input()
    
def validate_input(value, type:str, range=None, default=None):
    if value == "":
        value = default
    if type == 'int':
        try:
            value = int(value)
            if range and (value < range[0] or value > range[1]):
                return False
            return True
        except ValueError:
            return False
    elif type == 'float':
        try:
            value = float(value)
            if range and (value < range[0] or value > range[1]):
                return False
            return True
        except ValueError:
            return False
        
def custom_search():
    display_header("EJECUTANDO BÚSQUEDA DE ARQUITECTURA CON PARÁMETROS MANUALES")
    for param, details in params.items():
        colorama_print(f"\nIntorducir valor (Enter para utilizar el valor por defecto): \n{details['name']} ({details['type']}, rango: {details['range']}, default: {details['default']}): ", Back.RESET, Fore.GREEN)
        value = input()
        while not validate_input(value, details["type"], details["range"], details["default"]):
            colorama_print(f"Valor inválido. Intente de nuevo: ", Back.RESET, Fore.RED)
            value = input()
        if value == "":
            value = details["default"]
        params[param]["value"] = float(value) if details["type"] == "float" else int(value)
    # Convertir los parámetros a un diccionario
    params_dict = {param: details["value"] for param, details in params.items()}
    
    
    de = DiferentialEvolution(surrogate_model, **params_dict)
    de.start()
    colorama_print("Búsqueda de arquitectura completada.", Back.GREEN, Fore.RESET)
    colorama_print("\nIntroduzca el nombre para guardar los resultados (o presione Enter para usar la fecha y hora actuales): ", Back.RESET, Fore.GREEN)
    name = input()
    if name == "":
        name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    saved_path = save_results(name, de)
    colorama_print(f"Resultados guardados en output/{name}\n", Back.RESET, Fore.GREEN)
    # Preguntar si se debe entrenar la red  
    display_options([
        "Entrenar la red",
        "Continuar"
    ])
    training = get_input([1, 2])
    if training == 1:
        colorama_print("Entrenando la red...\n", Back.GREEN, Fore.RESET)
        # Entrenar la red
        train_network(saved_path)
        
        input()
    else:
        colorama_print("Continuando sin entrenar la red...\n", Back.GREEN, Fore.RESET)
        colorama_print("Presione Enter para continuar...\n", Back.RESET, Fore.GREEN)
        input()
        
def load_saved_networks():
    # Cargar redes guardadas
    display_header(" REDES GUARDADAS")
    saved_networks = []
    for root, dirs, files in os.walk(r"./output"):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                with open(path, 'r') as f:
                    data = json.load(f)
                saved_networks.append((path, data))
    
    # Preguntar para entrenar una red
    if len(saved_networks) == 0:
        colorama_print("No hay redes guardadas.\n", Back.RED, Fore.RESET)
        colorama_print("Presione Enter para continuar...\n", Back.RESET, Fore.GREEN)
        input()
        return
    colorama_print("Seleccione una opción:\n", Back.GREEN, Fore.RESET)
    colorama_print("[0] Volver al menú principal\n", Back.RESET, Fore.MAGENTA)
    for i, (path, data) in enumerate(saved_networks):
        colorama_print(f"[{i+1}] {path} - Predicted IOU: {data['predicted_iou']}, Entrenada: {data['trained']}\n", Back.RESET, Fore.MAGENTA)
    colorama_print("Seleccione una red para entrenar: ", Back.RESET, Fore.GREEN)
    input_ = get_input([i for i in range(len(saved_networks) + 1)])
    
    if input_ == 0:
        colorama_print("Regresando al menú principal...", Back.RED, Fore.RESET)
        sleep(1)
        return
    try:
        index = input_ - 1
        path, data = saved_networks[index]
        # Entrenar la red
        trained = train_network(path)
        if trained:
            colorama_print("Red entrenada.\n", Back.GREEN, Fore.RESET)
            colorama_print("Presione Enter para continuar...\n", Back.RESET, Fore.GREEN)
            input()
    except Exception as e:
        colorama_print(f"Error al entrenar la red: {e}\n", Back.RED, Fore.RESET)
        colorama_print("Presione Enter para continuar...\n", Back.RESET, Fore.GREEN)
        input()
        return


state = 0
while state != 3:
    if state == 0:
        state = main_menu()
    elif state == 1:
        input_ = architecture_search_menu()
        if input_ == 1:
            default_search()
            state = 0
            
        elif input_ == 2:
            custom_search()
            state = 0
            
        elif input_ == 3:
            state = 0
        else:
            colorama_print("Opción no válida. Intente de nuevo.\n", Back.RESET, Fore.RED)
            # Esperar que el usuario presione una tecla
            input("\nPresione Enter para continuar...")
            state = 1
            
    elif state == 2:
        load_saved_networks()
        state = 0
        pass
   
    elif state == 3:
        colorama_print("Saliendo...\n", Back.RED, Fore.RESET)
        break
    else:
        colorama_print("Opción no válida. Intente de nuevo.\n", Back.RESET, Fore.RED)
        # Esperar que el usuario presione una tecla
        input("\nPresione Enter para continuar...")
        state = 0

    
