#entrenamiento red neuronal

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
import joblib

# Custom transformer for feature engineering
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_['age_group'] = pd.cut(X_['age'], bins=[0, 18, 30, 45, 60, 75, 100], labels=['0-18', '19-30', '31-45', '46-60', '61-75', '75+'])
        X_['bmi_category'] = pd.cut(X_['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        X_['glucose_category'] = pd.cut(X_['avg_glucose_level'], bins=[0, 70, 100, 125, 200, 300], labels=['Low', 'Normal', 'Prediabetes', 'Diabetes', 'High'])
        X_['age_glucose_interaction'] = X_['age'] * X_['avg_glucose_level']
        X_['bmi_age_interaction'] = X_['bmi'] * X_['age']
        X_['health_score'] = X_['hypertension'] + X_['heart_disease'] + (X_['bmi'] > 30).astype(int) + (X_['age'] > 60).astype(int)
        return X_

# Load data
file_path = "stroke_dataset.csv"
df = pd.read_csv(file_path)

# Split data
X = df.drop('stroke', axis=1)
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define features
numeric_features = ['age', 'avg_glucose_level', 'bmi']
categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

# Create preprocessor
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create full pipeline including feature engineering
full_pipeline = Pipeline([
    ('feature_engineer', FeatureEngineer()),
    ('preprocessor', preprocessor)
])

# Apply preprocessing
X_train_preprocessed = full_pipeline.fit_transform(X_train)
X_test_preprocessed = full_pipeline.transform(X_test)


# Define the model
def create_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        Dense(8, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Create and train the model
n_features = X_train_preprocessed.shape[1]
model = KerasClassifier(model=create_model, input_dim=n_features, epochs=200, batch_size=32, verbose=1)

history = model.fit(X_train_preprocessed, y_train, validation_split=0.2)

# Evaluate the model
y_train_pred = model.predict(X_train_preprocessed)
y_test_pred = model.predict(X_test_preprocessed)
train_accuracy = np.mean(y_train_pred == y_train)
test_accuracy = np.mean(y_test_pred == y_test)

print(f"Accuracy de entrenamiento: {train_accuracy:.4f}")
print(f"Accuracy de prueba: {test_accuracy:.4f}")

# Detailed evaluation
y_pred_proba = model.predict_proba(X_test_preprocessed)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

y_pred = (y_pred_proba >= optimal_threshold).astype(int)

print("\nUmbral óptimo:", optimal_threshold)
print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))


# Calculate training and test accuracies
y_train_pred = model.predict(X_train_preprocessed)
y_test_pred = model.predict(X_test_preprocessed)
train_accuracy = np.mean(y_train_pred == y_train)
test_accuracy = np.mean(y_test_pred == y_test)

# Calculate overfitting percentage
overfitting_percentage = ((train_accuracy - test_accuracy) / train_accuracy) * 100

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Overfitting Percentage: {overfitting_percentage:.2f}%")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot using Matplotlib
plt.figure(figsize=(10, 5))

# Matplotlib subplot
plt.subplot(1, 2, 1)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Matplotlib')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['No Stroke', 'Stroke'], rotation=45)
plt.yticks(tick_marks, ['No Stroke', 'Stroke'])
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Add text annotations
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

# Seaborn subplot
plt.subplot(1, 2, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Seaborn')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()

plt.show()

# Save the model and preprocessor
joblib.dump(full_pipeline, 'full_pipeline_nn.joblib')
model.model_.save('keras_model_nn.keras')

# Function to predict stroke risk
def predict_stroke(new_data):
    loaded_pipeline = joblib.load('full_pipeline_nn.joblib')
    loaded_model = load_model('keras_model_nn.keras')

    new_data_preprocessed = loaded_pipeline.transform(new_data)
    predictions = loaded_model.predict(new_data_preprocessed)
    predicted_classes = (predictions >= 0.06).astype(int)

    return predicted_classes

# Example usage
new_patient = pd.DataFrame({
    'age': [77],
    'gender': ['Male'],
    'hypertension': [1],
    'heart_disease': [0],
    'ever_married': ['Yes'],
    'work_type': ['Private'],
    'Residence_type': ['Urban'],
    'avg_glucose_level': [90.5],
    'bmi': [28.1],
    'smoking_status': ['formerly smoked']
})

result = predict_stroke(new_patient)
print("Stroke prediction:", "Stroke" if result[0] == 1 else "Normal")