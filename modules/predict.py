import dill
import os
import json
import pandas as pd
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '.')
model_path = f'{path}/data/models'
test_path = f'{path}/data/test'

def predict():
    model_sort = sorted(os.listdir(model_path))[-1]

    with open(model_path + '/' + model_sort, 'rb') as file:
        model = dill.load(file)

    df_pred = pd.DataFrame(columns=['id', 'pred'])

    test_files = os.listdir(test_path)
    for file in test_files:
        with open(test_path + '/' + file, 'rb') as file_test:
            form = json.load(file_test)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)
            X = {'id': df.id, 'pred': y}
            df1 = pd.DataFrame(X)
            df_pred = pd.concat([df_pred,df1], axis=0)

    df_pred.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv')


if __name__ == '__main__':
    predict()
