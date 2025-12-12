import torch.nn as nn
from .fft_layers import BlockCirculantLinear

def replace_linear_layers(model):
    """
    Beispiel: ersetzt alle nn.Linear Layer
    (spaeter selektiv fuer Llama anpassen)
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print("Found Linear:", name)
