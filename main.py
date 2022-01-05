"""main.py"""

import matplotlib
import requests
from requests.models import HTTPError
import shap

import streamlit as st
import streamlit.components.v1 as components

from src import question_answer
from src import shap_bugs
from src import text_classification


def st_shap(plot):
    if isinstance(plot, str):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot}</body>"
        components.html(shap_html, height=400, scrolling=True)
    elif isinstance(plot, matplotlib.figure.Figure):
        st.pyplot(plot)


@st.cache(allow_output_mutation=True)
def get_explainer(task, model_path):
    FACTORY = {
        "Text Classification": text_classification.TextClassificationExplainer,
        "Question Answering": question_answer.QAExplainer,
    }

    return FACTORY[task](model_path)


@st.cache(allow_output_mutation=True)
def get_shap_values_and_predictions(task, model_path, inputs):
    explainer = get_explainer(task, model_path)
    return explainer.get_shap_values(inputs), explainer.get_predictions(inputs)


st.title("Explainable Transformers")

st.write("### Input your arguments")
task = st.selectbox("Task", ["Text Classification", "Question Answering"])
model_path = st.text_input(
    "HuggingFace Model Path", "nateraw/bert-base-uncased-emotion"
)
dataset_link = st.text_input("Dataset link")
if task == "Text Classification":
    st.caption(
        "The link should point to a text file with each input example separated by newline (\\n)"
    )
    text = st.text_area("Enter input for model", "I am Happy!")
    inputs = [text] if text else []
elif task == "Question Answering":
    st.caption(
        "The link should point to a CSV with 2 columns: question and context."
    )
    question = st.text_area("Enter question")
    context = st.text_area("Enter context")

    inputs = [question + "[SEP]" + context] if question and context else []

try:
    # Validation Conditions
    if dataset_link:
        if task == "Text Classification":
            res = requests.get(dataset_link)
            if res.status_code != 200:
                raise HTTPError(
                    f"Error {res.status_code}: Unable to fetch file at {dataset_link}"
                )

            inputs = res.text.split("\n")

            while "" in inputs:
                inputs.remove("")

    if len(inputs) == 0:
        raise ValueError("Input is incomplete")

    # Operations
    shap_values, predictions = get_shap_values_and_predictions(
        task, model_path, inputs
    )

    # Presentation
    st.write("## Prediction")
    st.dataframe(predictions)

    st.write("## Text Plot")
    if task == "Text Classification":
        st_shap(shap.plots.text(shap_values["shap_values"][0], display=False))
    elif task == "Question Answering":
        st.write("### Start ")
        st_shap(
            shap.plots.text(shap_values["start_shap_values"][0], display=False)
        )
        st.write("### End ")
        st_shap(
            shap.plots.text(shap_values["end_shap_values"][0], display=False)
        )

    if task == "Text Classification":
        st.write("## Waterfall Plot")
        output_names = shap_values["shap_values"].output_names
        output_col, plot_col = st.columns([1, 3])
        col = output_col.radio("Select one output", output_names)
        with plot_col:
            st_shap(
                shap_bugs.waterfall(
                    shap_values["shap_values"][0, :, col], show=False
                )
            )
except ValueError as e:
    st.write(f"##### {str(e)}")
except EnvironmentError as e:
    st.write(f"##### {str(e)}")
