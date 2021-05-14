from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory


url="https://raw.githubusercontent.com/benbot8/azure_capstone/f54d4da8ea6c2cd5c9082b6703271c25d5f16b40/starter_file/train.csv"

df = TabularDatasetFactory.from_delimited_files(path=url)

#df = pd.read_csv("train.csv")

def clean_data(data):
    #clean data and convert categorical to indicator variables 
    x_df = data.dropna()
    x_df.reset_index(drop=True, inplace=True)
    x_df.drop(['state', 'account_length', 'area_code'], axis = 1, inplace=True)
    x_df['international_plan'] = x_df.international_plan.apply(lambda s: 1 if s == "yes" else 0)
    x_df['voice_mail_plan'] = x_df.voice_mail_plan.apply(lambda s: 1 if s == "yes" else 0)
    x_df['churn'] = x_df.churn.apply(lambda s: 1 if s == "yes" else 0)
    x_df.rename(columns={"churn": "y"}, inplace=True)
    y_df = x_df.pop("y")
    return x_df, y_df


x,y = clean_data(df)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
run = Run.get_context()


def main():
    
     # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=2000, help="Maximum number of iterations to converge")
    parser.add_argument('--solver', type=str, default='lbfgs', help="chose the algorithm to train the model")

    args = parser.parse_args()

    run.log("Regularization Strength:", float(args.C))
    run.log("Max iterations:", int(args.max_iter))
    run.log("Algorithm ", args.solver)


    model = LogisticRegression(solver=args.solver, C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    AUC_weighted = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1], average="weighted")
    run.log("AUC_weighted", float(AUC_weighted))



if __name__ == '__main__':
    main()