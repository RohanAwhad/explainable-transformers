import transformers
import torch
import shap

from typing import List


class QAExplainer:
    def __init__(self, model_path):
        self._setup_explainer(model_path)

    def _setup_explainer(self, model_path):
        """
        Loads model and tokenizer and initializes SHAP Explainer
        """
        self.model_path = model_path
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path, use_fast=True
        )
        self.model = transformers.AutoModelForQuestionAnswering.from_pretrained(
            model_path
        )

        self.explainer_start = shap.Explainer(
            self._start_shap_func, self.tokenizer
        )
        self.explainer_end = shap.Explainer(self._end_shap_func, self.tokenizer)

    def explain(self, inputs: List[str], model_path: str = None):
        """
        Explains predictions of inputs using SHAP
        and the model from model_path provided
        """
        if model_path and self.model_path != model_path:
            self._setup_explainer(model_path)

        return {
            "start_shap_values": self.explainer_start(inputs),
            "end_shap_values": self.explainer_end(inputs),
        }

    def _shap_func(self, inputs, start):
        """Processing of inputs to generate logits, that are ingested by shap"""
        outs = []
        for data in inputs:
            question, context = data.split("[SEP]")
            d = self.tokenizer(question, context)
            out = self.model.forward(
                **{k: torch.tensor(d[k]).reshape(1, -1) for k in d}
            )
            logits = out.start_logits if start else out.end_logits
            outs.append(logits.reshape(-1).detach().numpy())
        return outs

    def _start_shap_func(self, inputs):
        return self._shap_func(inputs, True)

    def _end_shap_func(self, inputs):
        return self._shap_func(inputs, False)
