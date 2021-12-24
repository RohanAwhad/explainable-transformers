from src import text_classification

model_path = "nateraw/bert-base-uncased-emotion"
task = "text-classification"
input_text = [
    "it's good!",
    ("i can go from feeling so hopeless to so damned hopeful just from being "
     "around someone who cares and is awake")
]

explainer = text_classification.get_explainer(model_path)
shap_values = explainer(input_text)
print(shap_values)