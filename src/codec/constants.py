# Restricciones para el tamaño de las redes
MAX_LAYERS = 4
MAX_CONVS_PER_LAYER = 2

# Tamaños de las distintas partes de un cromosoma tanto en binario como en real
# 3 valores por convolución (filters, kernel_size, activation)
REAL_CONV_LEN = 3
REAL_POOLING_LEN = 1
REAL_CONCAT_LEN = 1
REAL_CONVS_LEN = REAL_CONV_LEN * MAX_CONVS_PER_LAYER
# convoluciones tanto en encoding como en decoding + pooling + concatenación
REAL_LAYER_LEN = REAL_CONVS_LEN * 2 + REAL_POOLING_LEN + REAL_CONCAT_LEN
# MAX_LAYERS capas + bottleneck
REAL_CHROMOSOME_LEN = REAL_LAYER_LEN * MAX_LAYERS + REAL_CONVS_LEN
# filters: 4 bits, kernel_size: 2 bits, activation: 4 bits = 10 bits
BIN_CONV_LEN = 10
BIN_POOLING_LEN = 2
BIN_CONCAT_LEN = 1
BIN_CONVS_LEN = BIN_CONV_LEN * MAX_CONVS_PER_LAYER
BIN_LAYER_LEN = BIN_CONVS_LEN * 2 + 3
# MAX_LAYERS capas + bottleneck
BIN_CHROMOSOME_LEN = BIN_LAYER_LEN * MAX_LAYERS + BIN_CONVS_LEN

# Valores que representan una capa de identidad en un cromosoma real y binario
IDENTITY_CONV_REAL = [0.01, 0.01, 0.01]  # f=None + s=1 + a=linear
IDENTITY_LAYER_REAL = (
    # identity_convs + p=None
    IDENTITY_CONV_REAL * MAX_CONVS_PER_LAYER +
    [0.01] +
    # identity_convs + concat=False
    IDENTITY_CONV_REAL * MAX_CONVS_PER_LAYER +
    [0.01]
)
IDENTITY_CONV_BIN = "0000" + "00" + "0000"  # f=None + s=1 + a=linear
IDENTITY_LAYER_BIN = (
    IDENTITY_CONV_BIN * MAX_CONVS_PER_LAYER + "00" +  # identity_convs + p=None
    IDENTITY_CONV_BIN * MAX_CONVS_PER_LAYER + "0"  # identity_convs + concat=False
)

# Variables de decisión para un cromosoma binario,
# la representación real se obtiene a partir de estas variables
VALID_FILTERS = {
    "0001": 1,
    "0011": 2,
    "0010": 4,
    "0110": 8,
    "0111": 16,
    "0101": 32,
    "0100": 64,
    "1100": 128,
    "1101": 256,
    "1111": 512,
    "1110": 1024,
}
FILTERS = {
    "0000": None,
} | VALID_FILTERS
KERNEL_SIZES = {
    "00": 1,
    "01": 3,
    "11": 5,
}
ACTIVATION_FUNCTIONS = {
    "0000": "linear",
    "0001": "relu",
    "0011": "softplus",
    "0010": "elu",
    "0110": "selu",
    "0111": "sigmoid",
    "0101": "tanh",
    "0100": "softsign",
    "1100": "softmax"
}

VALID_POOLINGS = {
    "01": "max",
    "11": "average",
}
POOLINGS = {
    "00": None
} | VALID_POOLINGS
CONCATENATION = {
    "0": False,
    "1": True,
}
