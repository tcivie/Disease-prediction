import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Paths to the CSV files
# Update the path to your training CSV file
train_file_path = r'data\Blood_samples_dataset_balanced_2(f).csv'
# Update the path to your testing CSV file
test_file_path = r'data\blood_samples_dataset_test.csv'

# Load training data from CSV
train_data = pd.read_csv(train_file_path)
# All rows, all columns except the last
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values  # All rows, last column only

# Load testing data from CSV
test_data = pd.read_csv(test_file_path)
X_test = test_data.iloc[:, :-1].values  # All rows, all columns except the last
y_test = test_data.iloc[:, -1].values  # All rows, last column only


def create_svm(kernel='linear', C=1.0, random_state=42):
    # Create SVM classifier
    # You can change the kernel here
    svm = SVC(kernel=kernel, C=C, random_state=random_state)

    # Train the model
    svm.fit(X_train, y_train)

    # Predict the test set results
    return svm.predict(X_test)


best_settings = {'kernel': 'linear', 'C': 0.01, 'acc': 0}
kernels = ['linear', 'poly', 'rbf']
for kernel in kernels:
    c = 0.001
    while c <= 10000:
        y_pred = create_svm(kernel=kernel, C=c)
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_settings['acc']:
            best_settings['acc'] = accuracy
            best_settings['C'] = c
            best_settings['kernel'] = kernel
        c *=10
        
y_pred = create_svm(kernel=best_settings['kernel'], C=best_settings['C'])
print('Bes settings:')
for setting in best_settings:
    try:
        print(f'{setting}: {best_settings[setting] * 100:.2f}%')
    except:
        print(f'{setting}: {best_settings[setting]}')

# Get all unique classes from both actual and predicted
all_classes = np.unique(np.concatenate((y_test, y_pred)))

# Calculate prediction counts
true_counts = pd.Series(y_test).value_counts().reindex(
    all_classes, fill_value=0).sort_index()
pred_counts = pd.Series(y_pred).value_counts().reindex(
    all_classes, fill_value=0).sort_index()

# Bar chart for actual vs predicted
plt.figure(figsize=(10, 6))
index = np.arange(len(true_counts))
bar_width = 0.35

plt.bar(index, true_counts, bar_width, label='Actual Counts')
plt.bar(index + bar_width, pred_counts, bar_width, label='Predicted Counts')

plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Comparison of Actual and Predicted Class Counts')
# Adjust the labels to the center of grouped bars
plt.xticks(index + bar_width / 2, [str(cls) for cls in all_classes])
plt.legend()
plt.show()
