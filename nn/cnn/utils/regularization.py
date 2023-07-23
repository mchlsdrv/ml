import torch


def stochastic_depth(model, inputs, survival_prop=0.5) -> torch.tensor:
    """
    Randomly skips certain layers
    :param model: The training model
    :param inputs: Input Tensor
    :param survival_prop:
    :return: outputs: The
    """
    if not model.training:
        return inputs

    binary_tensor = torch.randn(inputs.shape[0], 1, 1, 1, device=inputs.device) > survival_prop

    outputs = torch.div(inputs, survival_prop) * binary_tensor

    return outputs
