import shap
import transformers


def get_explainer(model_path: str, task: str):
    # load the model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path,
                                                           use_fast=True)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_path)

    # build a pipeline object to do predictions
    pred = transformers.pipeline(task,
                                 model=model,
                                 tokenizer=tokenizer,
                                 return_all_scores=True)

    return shap.Explainer(pred)