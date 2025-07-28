import torch


def tiny_value_of_dtype(dtype: torch.dtype) -> float:
    """
    Returns a moderately tiny value for a given floating point PyTorch
    data type that is used to avoid numerical issues such as division by zero.


    This is different from `torch.finfo(dtype).tiny` because it causes some NaN bugs.

    Reference:
    https://github.com/MaksymDel/allennlp-light/blob/main/allennlp_light/nn/util.py#L1999

    Args:
        dtype (torch.dtype): A torch float dtype.
    Returns:
        A small float value that is not too small to causes any NaN bugs.
    Raises:
        TypeError: If the dtype is not one of the following: torch.float64,
            torch.float32, torch.float16, or torch.bfloat16
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float32 or dtype == torch.float64 or dtype == torch.bfloat16:
        return 1e-13
    elif dtype == torch.float16:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))
