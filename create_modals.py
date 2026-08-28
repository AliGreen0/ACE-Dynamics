import pandas as pd
from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE

for i in range(1, 4):
    dataset_name = 'KIPAN'  # 'BRCA', 'ROSMAP', 'LGG', 'KIPAN'
    dataset_file = f"./{dataset_name}/{i}_tr.csv"
    y_train_file = f"./{dataset_name}/labels_tr.csv"
    dataset_file_test = f"./{dataset_name}/{i}_te.csv"


    X_train = pd.read_csv(dataset_file, header=None)
    y_train = pd.read_csv(y_train_file, header=None).iloc[:, 0]
    X_test = pd.read_csv(dataset_file_test, header=None)


    correlation_matrix = pd.DataFrame(X_train).corr()

    multiplied_result = pd.DataFrame(X_train).dot(correlation_matrix)
    multiplied_result_test = pd.DataFrame(X_test).dot(correlation_matrix)


    multiplied_result.to_csv(f"./{dataset_name}/{i + 3}_tr.csv", header=False, index=False)
    multiplied_result_test.to_csv(f"./{dataset_name}/{i + 3}_te.csv", header=False, index=False)

