import os
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

project_dir = Path(__file__).resolve().parents[2]

log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)

tgt = 'label'

def main(input_file_path, n_splits=5):
    input_file_name = os.path.join(input_file_path, "train.csv")
    df = pd.read_csv(input_file_name)

    y = df.loc[~df[tgt].isna(), tgt]
    X = df.loc[~df[tgt].isna(), :].drop(["well_id", tgt, 'row_id'], axis=1)

    groups = df.loc[~df[tgt].isna(), 'well_id']
    print(groups.unique())

    cv = GroupKFold(n_splits)

    X["folds"] = -1
    for k, (_, test_index) in enumerate(cv.split(X, y, groups)):
        X.loc[test_index, "folds"] = k

    X[tgt] = list(y)
    X.to_csv(input_file_name.replace(".csv", "_.csv"), index=False)

if __name__ == "__main__":
    # not used in this stub but often useful for finding various files

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    input_file_path = os.path.join(project_dir, "data", "processed")
    os.makedirs(input_file_path, exist_ok=True)
    main(input_file_path)
