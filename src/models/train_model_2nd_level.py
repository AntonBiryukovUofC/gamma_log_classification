import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

project_dir = Path(__file__).resolve().parents[2]

log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)

tgt='label'

def main(input_file_path, output_file_path, n_splits=5):
    X=[]
    shifts = np.arange(-50,50,5)
    input_file_name = os.path.join(input_file_path, "holdout_lgbm.pck")
    df = pd.read_pickle(input_file_name)
    grouped = df.groupby('well_id')
    colnames = [f'label_{x}' for x in range(5)]
    for s in shifts:
        s_colnames = [f'{x}_{s}' for x in colnames]
        df[s_colnames] = grouped[colnames].shift(s)
        X.append(df.copy())
    models = []
    scores = []
    f1_scores = []
    scores_dm = []

    y = df.loc[~df[tgt].isna(), tgt]
    X = df.loc[~df[tgt].isna(), :].drop(["well_id", tgt,'row_id'], axis=1)

    groups = df.loc[~df[tgt].isna(), 'well_id']
    print(groups.unique())
    preds_holdout = np.ones((df.shape[0], 5))*(-50)

    cv= GroupKFold(n_splits)

    for k, (train_index, test_index) in enumerate(cv.split(X, y, groups)):

        X_train, X_holdout = X.iloc[train_index, :], X.iloc[test_index, :]
        model = LGBMClassifier(n_estimators=500,
                               learning_rate=0.08,
                               feature_fraction=0.1,
            objective="multiclassova",
            is_unbalance = True,
            num_leaves=32,
            random_state=k,
            n_jobs=-1,


        )

        y_train, y_holdout = y.iloc[train_index], y.iloc[test_index]

        model.fit(
            X_train,
            y_train,
            verbose=200,
            #eval_set=(X_holdout,y_holdout),
            #early_stopping_rounds=50
        )
        # model.fit(X_train, y_train)
        score = accuracy_score(y_holdout, model.predict(X_holdout))
        f1_sc = f1_score(y_holdout, model.predict(X_holdout),labels = [1,2,3,4],average='weighted')
        f1_scores.append(f1_sc)
        models.append(model)
        scores.append(score)
        logging.info(f"{k} - Holdout score = {score}, f1 = {f1_sc}")
        preds_holdout[test_index, :] = model.predict_proba(X_holdout)
        #preds_test[k, :, :] = model.predict_proba(X_test)

        interim_file_path = os.path.join(project_dir, "data", "interim")
        os.makedirs(interim_file_path, exist_ok=True)
        df_preds = pd.DataFrame(preds_holdout,columns = [f'label_{x}' for x in range(5)],index=df.index)
        df_preds = pd.concat([df,df_preds],axis = 1)
        df_preds['pred'] = np.argmax(preds_holdout,axis=1)
        #df_preds.to_pickle(os.path.join(interim_file_path,f'holdout_lgbm_{k}.pck'))

    logging.info(f" Holdout score = {np.mean(scores)} , std = {np.std(scores)}")
    logging.info(f" Holdout F1 score = {np.mean(f1_scores)} , std = {np.std(f1_scores)}")

    logging.info(f" OOF Holdout score = {accuracy_score(y,np.argmax(preds_holdout,axis=1))} ")
    logging.info(f" OOF Holdout F1 score = {f1_score(y,np.argmax(preds_holdout,axis=1),labels = [1,2,3,4],average='weighted')}")

   # preds_df = df_test[['row_id','well_id']]
   # preds_df['label'] = np.argmax(preds_test.sum(axis=0), axis=1)
    return 0

if __name__ == "__main__":
    # not used in this stub but often useful for finding various files

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    input_file_path = os.path.join(project_dir, "data", "interim")
    output_file_path = os.path.join(project_dir, "models")
    os.makedirs(input_file_path, exist_ok=True)
    os.makedirs(output_file_path, exist_ok=True)

    preds_test = main(input_file_path, output_file_path)
   # preds_test.to_csv(os.path.join(input_file_path,'submit.csv'),index=False)