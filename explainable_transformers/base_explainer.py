from abc import ABC, abstractmethod
from typing import List


class BaseExplainer(ABC):
    def __init__(
        self,
        model_path: str = None,
        model=None,
        tokenizer=None,
    ):
        if (
            (model_path is None and (model is None or tokenizer is None))
            or (model_path is not None and (model is not None or tokenizer is not None))
            or (model_path is None and not (model is not None and tokenizer is not None))
        ):
            raise ValueError("either 'model_path' or 'model' and 'tokenizer' must be given but not both")

    @abstractmethod
    def get_shap_values(self, inputs: List[str]):
        pass
