"""Explain Text Classification module"""
import shap

from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from .base_explainer import BaseExplainer


class TextClassificationExplainer(BaseExplainer):
    """Explains Text Classification models using SHAP"""

    def __init__(
        self,
        model_path: str = None,
        model: AutoModelForSequenceClassification = None,
        tokenizer: AutoTokenizer = None,
    ):
        super().__init__(model_path=model_path, model=model, tokenizer=tokenizer)

        # load the model and tokenizer
        if model_path is not None:
            logger.info(f"model path: '{model_path}'")
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            logger.info("tokenizer loaded")
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            logger.info("model loaded")

        # build a pipeline object to do predictions
        self.pred = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
        )
        logger.info("text classification pipeline loaded")

        self.explainer = shap.Explainer(self.pred)
        logger.info("text classification explainer loaded")

    def get_shap_values(self, inputs):
        """
        Explains predictions of inputs using SHAP Explainer
        """
        logger.debug(f"explaining output for : {inputs}")
        return self.explainer(inputs)

    def get_predictions(self, inputs):
        """Return predictions from model"""

        predictions = self.pred(inputs)
        return {
            "labels": [output["label"] for output in predictions[0]],
            "scores": [output["score"] for output in predictions[0]],
        }
