import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.tree import ExtraTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, hamming_loss, classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from collections import Counter
import joblib

warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv('AIR.csv')
del df['StationId']
del df['Datetime']
df = df.dropna()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['AQI_Bucket'] = le.fit_transform(df['AQI_Bucket']).astype(int)

x1 = df.drop(labels='AQI_Bucket', axis=1)
y1 = df.loc[:, 'AQI_Bucket']

# Oversample data
ros = RandomOverSampler(random_state=42)
x, y = ros.fit_resample(x1, y1)
print("Original dataset count: ", Counter(y1))
print("After oversampling: ", Counter(y))

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)
print("Train dataset size:", len(x_train))
print("Test dataset size:", len(x_test))

# Create and train model
ETC = ExtraTreeClassifier()
ETC.fit(x_train, y_train)

# Predict and evaluate
predicted = ETC.predict(x_test)
cm = confusion_matrix(y_test, predicted)
print('Confusion Matrix:\n', cm)

accuracy = accuracy_score(y_test, predicted)
print("Accuracy Score:", accuracy)

hamming = hamming_loss(y_test, predicted)
print("Hamming Loss:", hamming)

report = classification_report(y_test, predicted)
print("Classification Report:\n", report)

# Display confusion matrix plot
from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ETC.classes_)
disp.plot()
plt.show()

# Plot accuracy bar chart
def graph():
    data = [accuracy]
    alg = "EXTRA TREE CLASSIFIER"
    plt.figure(figsize=(5,5))
    b = plt.bar(alg, data, color=("skyblue"))
    plt.title("Accuracy Score of Extra Tree Classifier")
    plt.legend(b, data, fontsize=9)
    plt.show()
graph()

# Save model
joblib.dump(ETC, 'MODEL.pkl')
