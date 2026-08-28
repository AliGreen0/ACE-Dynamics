import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

for i in range(1, 4):
    dataset_name = 'ROSMAP'
    dataset_file = f"./{dataset_name}/{i}_tr.csv"
    y_train_file = f"./{dataset_name}/labels_tr.csv"
    dataset_file_test = f"./{dataset_name}/{i}_te.csv"

    # Load the datasets
    X_train = pd.read_csv(dataset_file, header=None)
    y_train = pd.read_csv(y_train_file, header=None).iloc[:, 0]
    X_test = pd.read_csv(dataset_file_test, header=None)

    # Normalize (fit on train, transform both train and test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE oversampling
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

    # Compute the correlation matrix on the resampled data
    correlation_matrix = pd.DataFrame(X_resampled).corr()

    # Multiply resampled data and test data by the correlation matrix
    multiplied_result = pd.DataFrame(X_resampled).dot(correlation_matrix)
    multiplied_result_test = pd.DataFrame(X_test_scaled).dot(correlation_matrix)

    # Save results
    X_resampled.to_csv(f"{i}_tr.csv", header=False, index=False)
    multiplied_result.to_csv(f"{i + 3}_tr.csv", header=False, index=False)
    multiplied_result_test.to_csv(f"{i + 3}_te.csv", header=False, index=False)
    pd.Series(y_resampled).to_csv("labels_tr.csv", header=False, index=False)
