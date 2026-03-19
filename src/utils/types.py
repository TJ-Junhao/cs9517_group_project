from typing import Union
from numpy.typing import ArrayLike
from cv2.typing import MatLike
from torch import Tensor

Tensors = Union[ArrayLike, MatLike, Tensor]
