import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

project_dir = Path(__file__).resolve().parents[2]

log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)

exclude_cols = ['row_id']
tgt='label'

def main(input_file_path, output_file_path, n_splits=5):
    input_file_name = os.path.join(input_file_path, "train.pck")
    #input_file_name_test = os.path.join(input_file_path, "Test_final.pck")
   # output_file_name = os.path.join(output_file_path, f"models_lgbm.pck")
    df = pd.read_pickle(input_file_name)

    models = []
    scores = []
    scores_dm = []

    y = df.loc[~df[tgt].isna(), tgt]
    X = df.loc[~df[tgt].isna(), :].drop(["well_id", tgt,'row_id'], axis=1)
    groups = df.loc[~df[tgt].isna(), 'well_id']
    preds_holdout = np.zeros((df.shape[0], 5))
    cv= GroupKFold(n_splits)

    for k, (train_index, test_index) in enumerate(cv.split(X, y, groups)):

        X_train, X_holdout = X.iloc[train_index, :], X.iloc[test_index, :]
        model = LGBMClassifier(
            objective="multiclass",
            random_state=k,
            n_jobs=-1,
        )

        y_train, y_holdout = y.iloc[train_index], y.iloc[test_index]

        model.fit(
            X_train,
            y_train,
            eval_set=(X_holdout, y_holdout),
            early_stopping_rounds=50,
            verbose=200,
        )
        # model.fit(X_train, y_train)
        score = accuracy_score(y_holdout, model.predict(X_holdout))
        models.append(model)
        scores.append(score)
        logging.info(f"{k} - Holdout score = {score}")
        preds_holdout[test_index, :] = model.predict_proba(X_holdout)

        interim_file_path = os.path.join(project_dir, "data", "interim")
        os.makedirs(interim_file_path, exist_ok=True)
        df_preds = pd.DataFrame(preds_holdout,columns = [f'label_{x}' for x in range(5)],index=df.index)
        df_preds = pd.concat([df,df_preds],axis = 1)
        df_preds['pred'] = np.argmax(preds_holdout,axis=1)
        df_preds.to_pickle(os.path.join(interim_file_path,'holdout_lgbm.pck'))


if __name__ == "__main__":
    # not used in this stub but often useful for finding various files

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    input_file_path = os.path.join(project_dir, "data", "processed")
    output_file_path = os.path.join(project_dir, "models")
    os.makedirs(input_file_path, exist_ok=True)
    os.makedirs(output_file_path, exist_ok=True)

    preds_wsc = main(input_file_path, output_file_path)

    df = preds_wsc