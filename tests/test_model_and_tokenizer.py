import numpy as np

from explainable_transformers.text_classification import TextClassificationExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def test_model_and_tokenizer_instead_of_model_path_for_text_classification():
    model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    expected_explainer = TextClassificationExplainer(model_path=model_path)

    actual_explainer = TextClassificationExplainer(model=model, tokenizer=tokenizer)

    inp = ["I like you. I love you."]
    expected_shap_values = expected_explainer.get_shap_values(inp)
    actual_shap_values = actual_explainer.get_shap_values(inp)

    np.testing.assert_almost_equal(actual_shap_values.values, expected_shap_values.values)
    np.testing.assert_almost_equal(actual_shap_values.base_values, expected_shap_values.base_values)
    np.testing.assert_almost_equal(actual_shap_values.hierarchical_values, expected_shap_values.hierarchical_values)
