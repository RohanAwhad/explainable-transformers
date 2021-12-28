"""main.py"""

import shap

import streamlit as st
import streamlit.components.v1 as components

from src import text_classification
from src import question_answer


def st_shap(plot):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot}</body>"
    components.html(shap_html, height=400, scrolling=True)


@st.cache(allow_output_mutation=True)
def get_explainer(task, model_path):
    FACTORY = {
        "Text Classification": text_classification.TextClassificationExplainer,
        "Question Answering": question_answer.QAExplainer,
    }

    return FACTORY[task](model_path)


st.title("Explainable Transformers")

# with st.form("user-inputs"):
st.write("Input your arguments")
model_path = st.text_input(
    "HuggingFace Model Path", "nateraw/bert-base-uncased-emotion"
)
task = st.selectbox("Task", ["Text Classification", "Question Answering"])
if task == "Text Classification":
    inputs = [st.text_area("Enter input for model")]
elif task == "Question Answering":
    question = st.text_area("Enter question")
    context = st.text_area("Enter context")
    inputs = [question + "[SEP]" + context]

# submitted = st.form_submit_button("Run")
submitted = st.button("Run")

if submitted:
    shap_values = get_explainer(task, model_path).explain(inputs)

    if task == "Text Classification":
        st_shap(shap.plots.text(shap_values["shap_values"][0], display=False))
    elif task == "Question Answering":
        st_shap(
            shap.plots.text(shap_values["start_shap_values"][0], display=False)
        )
        st_shap(
            shap.plots.text(shap_values["end_shap_values"][0], display=False)
        )
