"""
This module defines the following routines used by the 'transform' step:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""
def transformer_fn():
  """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """
  from sklearn.preprocessing import StandardScaler
  return Pipeline(
    steps=[
      (
        "scale_features",
        StandardScaler(),
      )
    ]
  )