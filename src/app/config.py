ascii_art = """
███╗   ██╗ █████╗ ███████╗    ████████╗ ██████╗  ██████╗ ██╗     
████╗  ██║██╔══██╗██╔════╝    ╚══██╔══╝██╔═══██╗██╔═══██╗██║     
██╔██╗ ██║███████║███████╗       ██║   ██║   ██║██║   ██║██║     
██║╚██╗██║██╔══██║╚════██║       ██║   ██║   ██║██║   ██║██║     
██║ ╚████║██║  ██║███████║       ██║   ╚██████╔╝╚██████╔╝███████╗
╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝       ╚═╝    ╚═════╝  ╚═════╝ ╚══════╝                       
"""

# pop_size: int = 10, f: float = 0.9, 
# crossover_rate: float = 0.9, mutation_rate: float = 0.2, max_gen: int = 100)

# Datos, tipo y rango de los hiperparámetros
params = {
    "pop_size": {
        "name": "Tamaño de la población",
        "type": "int",
        "range": [10, 100],
        "default": 100
    },
    "f": {
        "name": "Factor de escala del diferencial",
        "type": "float",
        "range": [0.1, 1.0],
        "default": 0.9
    },
    "crossover_rate": {
        "name": "Probabilidad de cruce",
        "type": "float",
        "range": [0.1, 1.0],
        "default": 0.9
    },
    "mutation_rate": {
        "name": "Tasa de mutación",
        "type": "float",
        "range": [0.1, 1.0],
        "default": 0.2
    },
    "max_gen": {
        "name": "Número máximo de generaciones",
        "type": "int",
        "range": [2, 1000],
        "default": 100
    }
}