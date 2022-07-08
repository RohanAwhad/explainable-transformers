"""Explain Text Classification module"""
import pandas as pd
import shap
import streamlit as st
import transformers


class TextClassificationExplainer:
    """Explains Text Classification models using SHAP"""

    def __init__(self, model_path: str):
        # load the model and tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path, use_fast=True
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_path
        )

        # build a pipeline object to do predictions
        self.pred = transformers.pipeline(
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
        return {"shap_values": self.explainer(inputs)}

    def get_predictions(self, inputs):
        """Return predictions from model"""

        predictions = self.pred(inputs)
        return pd.DataFrame(
            {
                "labels": [output["label"] for output in predictions[0]],
                "scores": [output["score"] for output in predictions[0]],
            }
        )
