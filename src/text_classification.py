"""Explain Text Classification module"""
import shap
import streamlit as st
import transformers


class TextClassificationExplainer:
    """Explains Text Classification models using SHAP"""

    def __init__(self, model_path: str):
        self._setup_explainer(model_path)

    def _setup_explainer(self, model_path: str):
        """
        Loads model and tokenizer in a pipeline and
        initialises SHAP Explainer using the pipeline
        """
        # load the model and tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path, use_fast=True
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_path
        )

        # build a pipeline object to do predictions
        pred = transformers.pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
        )

        self.explainer = shap.Explainer(pred)

    def explain(self, inputs, model_path=None):
        """
        Explains predictions of inputs using SHAP Explainer
        """
        return {"shap_values": self.explainer(inputs)}
