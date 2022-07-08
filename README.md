# Explainable Transformers

Install by running following command:

```
pip install explainable-transformers==0.2.0
```

## Usage

```
from explainable_transformers.text_classification import TextClassificationExplainer

model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
explainer = TextClassificationExplainer(model_path=model_path)

inp = ["I like you. I love you."]
shap_values = explainer.get_shap_values(inp)

print(shap_values.base_values)
print(shap_values.values)
```

## Future Scope:

-   Want to give an option to enter either a single example or an entire dataset!

    For single example:

    -   Output is for local
    -   Text Plot is enough

    Entire Dataset:
    _ Reason: To explain model outputs on a global scale
    _ Output:

    1.  A plot for global values (Required): Waterfall Plot!
    2.  Download option for shap values of dataset

-   Support for multiple model types and NLP tasks!

