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
        "type": "int",
        "range": [10, 100],
        "default": 100
    },
    "f": {
        "type": "float",
        "range": [0.1, 1.0],
        "default": 0.9
    },
    "crossover_rate": {
        "type": "float",
        "range": [0.1, 1.0],
        "default": 0.9
    },
    "mutation_rate": {
        "type": "float",
        "range": [0.1, 1.0],
        "default": 0.2
    },
    "max_gen": {
        "type": "int",
        "range": [10, 1000],
        "default": 100
    }
}