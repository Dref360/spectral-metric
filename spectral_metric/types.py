from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass
class SimilarityArrays:
    sample_probability: Array
    sample_probability_norm: Array
