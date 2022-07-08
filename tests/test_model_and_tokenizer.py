import numpy as np
import pytest

from explainable_transformers.text_classification import TextClassificationExplainer
from explainable_transformers.question_answer import QAExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForQuestionAnswering


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


def test_model_and_tokenizer_instead_of_model_path_for_qa():
    model_path = "deepset/minilm-uncased-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    expected_explainer = QAExplainer(model_path=model_path)

    actual_explainer = QAExplainer(model=model, tokenizer=tokenizer)

    question = "Where do I live?"
    context = "My name is Wolfgang and I live in Berlin"
    inp = [question + "[SEP]" + context]

    expected_shap_values = expected_explainer.get_shap_values(inp)
    actual_shap_values = actual_explainer.get_shap_values(inp)

    np.testing.assert_almost_equal(
        actual_shap_values["start_shap_values"].values, expected_shap_values["start_shap_values"].values
    )
    np.testing.assert_almost_equal(
        actual_shap_values["end_shap_values"].values, expected_shap_values["end_shap_values"].values
    )


def test_raise_error_for_no_input():
    with pytest.raises(ValueError):
        _ = TextClassificationExplainer()


def test_raise_error_for_giving_all():
    with pytest.raises(ValueError):
        model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        _ = TextClassificationExplainer(model_path=model_path, model=model, tokenizer=tokenizer)


def test_raise_error_for_giving_model_but_no_tokenizer():
    with pytest.raises(ValueError):
        model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        _ = TextClassificationExplainer(model=model, tokenizer=None)


def test_raise_error_for_giving_model_path_and_model():
    with pytest.raises(ValueError):
        model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        _ = TextClassificationExplainer(model_path=model_path, model=model, tokenizer=None)
