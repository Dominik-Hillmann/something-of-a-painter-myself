import torch
from torch import Tensor

import numpy as np

from typing import Union, Tuple

def scale(
    x: Union[np.ndarray, Tensor],
    from_range: Tuple[int, int] = (0, 255),
    to_range: Tuple[int, int] = (-1, 1)
) -> Union[np.ndarray, Tensor]:
    """Applies min-max-normalization to tensors with values between 0 and
    255.
    Output tensors will have values between -1 and 1.
    """
    if type(x) is np.ndarray:
        x = Tensor(x)
    
