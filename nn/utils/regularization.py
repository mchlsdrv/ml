import numpy as np
import torch

def stochastic_depth(inputs, min_survival_prop=0.5) -> torch.tensor:
    """
    Randomly skips certain layers
    :param inputs: Input Tensor
    :param min_survival_prop: The minimal probability that the layer will be dropped
    :return: outputs: The
    """
    dev = inputs.device
    btch_sz = inputs.shape[0]

    # - We want the input layer not to be discarded ever, as it holds low-level features. The second layer should be
    # discarded in a lower probability than the third layer, etc. We continue in such fashion until we arrive at the
    # limit set by the min_survival_prop
    survival_prop_vector = np.ones((inputs.shape[0], 1, 1, 1)) * min_survival_prop
    survival_prop_vector[0, 0, 0, 0] = 1.
    non_input_prop_vector = np.arange(0.9, min_survival_prop + btch_sz, -0.1)[:btch_sz]
    survival_prop_vector[1:len(non_input_prop_vector) + 1, 0, 0, 0] = non_input_prop_vector
    survival_prop_vector = torch.tensor(survival_prop_vector, dtype=torch.float32, device=dev)

    binary_tensor = torch.rand(inputs.shape[0], 1, 1, 1, device=dev) < survival_prop_vector

    outputs = torch.div(inputs, survival_prop_vector) * binary_tensor

    return outputs


def apply_stochastic_depth(x, inputs, survival_prop: float = 0.5, training: bool = True):
    if training:
        x = stochastic_depth(x, min_survival_prop=survival_prop) + inputs
    else:
        x = x + inputs
    return x
