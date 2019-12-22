import pandas as pd
sub = pd.read_csv("data/raw/CAX_LogFacies_Submission_File.csv")
df = pd.read_csv("data/result/LSTM_submit.csv")
sub["label"] = list(df["label"])
sub.to_csv("data/result/submit.csv",index=False)
