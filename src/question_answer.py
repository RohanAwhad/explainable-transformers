import transformers
import torch
import shap
import pandas as pd

from typing import List


class QAExplainer:
    """Explains QA models using SHAP"""

    def __init__(self, model_path):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path, use_fast=True
        )
        self.model = transformers.AutoModelForQuestionAnswering.from_pretrained(
            model_path
        )
        self.pipeline = transformers.QuestionAnsweringPipeline(
            model=self.model, tokenizer=self.tokenizer
        )

        self._start_shap_func.__func__.output_names = self._out_names
        self._end_shap_func.__func__.output_names = self._out_names

        self.explainer_start = shap.Explainer(
            self._start_shap_func,
            self.tokenizer,
        )
        self.explainer_end = shap.Explainer(
            self._end_shap_func,
            self.tokenizer,
        )

    def get_shap_values(self, inputs: List[str]):
        """Explains predictions of inputs using SHAP and the model provided"""
        return {
            "start_shap_values": self.explainer_start(inputs),
            "end_shap_values": self.explainer_end(inputs),
        }

    def _shap_func(self, inputs, start):
        """Processing of inputs to generate logits, that are ingested by SHAP"""
        outs = []
        for data in inputs:
            question, context = data.split("[SEP]")
            d = self.tokenizer(question, context)
            with torch.no_grad():
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

    def _out_names(self, inputs):
        question, context = inputs.split("[SEP]")
        d = self.tokenizer(question, context)
        return [self.tokenizer.decode([id]) for id in d["input_ids"]]

    def get_predictions(self, inputs):
        question, context = inputs[0].split("[SEP]")
        predictions = self.pipeline(question=question, context=context)
        print(predictions)
        return pd.DataFrame(
            {
                "answer": [predictions["answer"]],
                "score": [predictions["score"]],
            }
        )
