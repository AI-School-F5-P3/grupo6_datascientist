import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve

# Cargar los datos
df = pd.read_csv('stroke_dataset.csv')

# Ingeniería de características (mantenemos la que ya tenías)
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 30, 45, 60, 75, 100], labels=['0-18', '19-30', '31-45', '46-60', '61-75', '75+'])
df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
df['glucose_category'] = pd.cut(df['avg_glucose_level'], bins=[0, 70, 100, 125, 200, 300], labels=['Low', 'Normal', 'Prediabetes', 'Diabetes', 'High'])
df['age_glucose_interaction'] = df['age'] * df['avg_glucose_level']
df['bmi_age_interaction'] = df['bmi'] * df['age']
df['health_score'] = df['hypertension'] + df['heart_disease'] + (df['bmi'] > 30).astype(int) + (df['age'] > 60).astype(int)

# Separar características y variable objetivo
X = df.drop('stroke', axis=1)
y = df['stroke']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Definir columnas numéricas y categóricas
numeric_features = ['age', 'avg_glucose_level', 'bmi', 'age_glucose_interaction', 'bmi_age_interaction', 'health_score']
categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'age_group', 'bmi_category', 'glucose_category']

# Crear preprocesadores para características numéricas y categóricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar preprocesadores
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Aplicar el preprocesamiento
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Definir el modelo de red neuronal
def create_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Crear el modelo
n_features = X_train_preprocessed.shape[1]
model = KerasClassifier(model=create_model, input_dim=n_features, epochs=200, batch_size=32, verbose=0)

# Entrenar el modelo
history = model.fit(X_train_preprocessed, y_train, validation_split=0.2)

# Calcular accuracy de entrenamiento
y_train_pred = model.predict(X_train_preprocessed)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Calcular accuracy de prueba
y_test_pred = model.predict(X_test_preprocessed)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Accuracy de entrenamiento: {train_accuracy:.4f}")
print(f"Accuracy de prueba: {test_accuracy:.4f}")

# Evaluar el modelo con más detalle
y_pred_proba = model.predict_proba(X_test_preprocessed)[:, 1]

# Encontrar el umbral óptimo
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

y_pred = (y_pred_proba >= optimal_threshold).astype(int)

print("\nUmbral óptimo:", optimal_threshold)
print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))