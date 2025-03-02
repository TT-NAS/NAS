FILTERS = {
    # TODO: Volver a hacer la base con la codificación en la que 0000 es 2**0 y no 0 :c
    # '0000': 0,
    '0001': 2**0,  # 1
    '0011': 2**1,  # 2
    '0010': 2**2,  # 4
    '0110': 2**3,  # 8
    '0111': 2**4,  # 16
    '0101': 2**5,  # 32
    '0100': 2**6,  # 64
    '1100': 2**7,  # 128
    '1101': 2**8,  # 256
    '1111': 2**9,  # 512
    '1110': 2**10,  # 1024
}
KERNEL_SIZES = {
    '00': 1,
    '01': 3,
    '11': 5,
}
ACTIVATION_FUNCTIONS = {
    '0000': 'relu',
    '0001': 'sigmoid',
    '0011': 'tanh',
    '0010': 'softmax',
    '0110': 'softplus',
    '0111': 'softsign',
    '0101': 'selu',
    '0100': 'elu',
    '1100': 'exponential',
    '1101': 'linear',
}
POOLINGS = {
    '0': 'max',
    '1': 'average',
}
CONCATENATION = {
    '0': False,
    '1': True,
}
