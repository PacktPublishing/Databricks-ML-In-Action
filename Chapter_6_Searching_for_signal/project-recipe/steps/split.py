import logging
from pandas import DataFrame

_logger = logging.getLogger(__name__)

def custom_split(df: DataFrame):
  """
  Uses a Straified split to create the training and test datasets.
  Mark rows of the ingested datasets to be split into training, validation, and test datasets.
  :param dataset: The dataset produced by the data ingestion step.
  :return: Series of strings with each element to be either 'TRAINING', 'VALIDATION' or 'TEST'
  """
  from pandas import Series
  from sklearn.model_selection import StratifiedGroupKFold

  target_col = 'StartHesitation'
  best_fold = 0

  sgkf = StratifiedGroupKFold(n_splits=8, random_state=416, shuffle=True)
  splits = sgkf.split(X=df['id'], y=df[target_col], groups=df['Subject'])
  for fold, (train_index, test_index) in enumerate(splits):
    if fold == best_fold:
      print(f"Training label distribution {df.loc[train_index].groupby([target_col]).size()/(1.0*len(train_index))*100}")
      print(f"Testing label distribution {df.loc[test_index].groupby([target_col]).size()/(1.0*len(test_index))*100}")
      break

  splits = Series("TRAINING", index=range(len(df)))
  
  # validation data not the real validation set --- TODO fix this
  splits[test_index] = "VALIDATION"

  # testing data
  splits[test_index] = "TEST"

  return splits