from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.core import Workspace, Dataset, Datastore


workspace = Workspace.from_config()

found = False
key = "customer-churn"

if key in workspace.datasets.keys(): 
        found = True
        dataset = workspace.datasets[key] 

df = dataset.to_pandas_dataframe()

def clean_data(data):
    x_df = data.dropna()

    x_df["international_plan"] = x_df.international_plan.apply(lambda s: 1 if s == "yes" else 0)
    x_df["voice_mail_plan"] = x_df.voice_mail_plan.apply(lambda s: 1 if s == "yes" else 0)
    x_df.drop("state", inplace=True, axis=1)
    x_df.drop("area_code", inplace=True, axis=1)
    y_df = x_df.pop("churn").apply(lambda s: 1 if s == "yes" else 0)
    return x_df, y_df

run = Run.get_context()

x,y = clean_data(df)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=20, random_state=42)


def main():
    
     # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")
    parser.add_argument('--solver', type=str, default='lbfgs', help="chose the algorithm to train the model")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    run.log("Algorithm ", args.solver)

    model = LogisticRegression(solver=args.solver, C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    print( accuracy)



if __name__ == '__main__':
    main()