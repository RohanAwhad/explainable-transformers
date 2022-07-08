"""Explain Text Classification module"""
import shap

from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class TextClassificationExplainer:
    """Explains Text Classification models using SHAP"""

    def __init__(
        self,
        model_path: str = None,
        model: AutoModelForSequenceClassification = None,
        tokenizer: AutoTokenizer = None,
    ):
        if (
            (model_path is None and (model is None or tokenizer is None))
            or (model_path is not None and (model is not None or tokenizer is not None))
            or (model_path is None and not (model is not None and tokenizer is not None))
        ):
            raise ValueError("either 'model_path' or 'model' and 'tokenizer' must be given but not both")

        # load the model and tokenizer
        if model_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # build a pipeline object to do predictions
        self.pred = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
        )

        self.explainer = shap.Explainer(self.pred)

    def get_shap_values(self, inputs):
        """
        Explains predictions of inputs using SHAP Explainer
        """
        return self.explainer(inputs)

    def get_predictions(self, inputs):
        """Return predictions from model"""

        predictions = self.pred(inputs)
        return {
            "labels": [output["label"] for output in predictions[0]],
            "scores": [output["score"] for output in predictions[0]],
        }
