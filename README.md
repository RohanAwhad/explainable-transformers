# Explainable Transformers

To run

```
python3.7 -m venv env
source env/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
streamlit run main.py
```

## Future Scope:

-   Want to give an option to enter either a single example or an entire dataset!
    For single example:

    -   Output is for local

    Entire Dataset:
    _ Reason: To explain model outputs on a global scale
    _ Output:

    1.  A plot for global values (Required)
    2.  Download option for shap values of dataset

-   Support for multiple model types and NLP tasks!
