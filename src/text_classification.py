"""Explain Text Classification module"""
import shap
import transformers


def get_explainer(model_path: str):
    """
    Loads model and tokenizer in a pipeline and creates and returns a
    SHAP Explainer from the pipeline
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

    return shap.Explainer(pred)
