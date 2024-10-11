import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import cross_val_score
import optuna

# Cargar los datos
data = pd.read_csv('stroke_dataset.csv')

# Separar características y variable objetivo
X = data.drop('stroke', axis=1)
y = data['stroke']

# Definir columnas categóricas y numéricas
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
numerical_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

# Crear preprocesadores
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
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar preprocesamiento
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Definir la función objetivo para Optuna
def objective(trial):
    # Hiperparámetros para XGBoost
    xgb_params = {
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300),
        'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('xgb_gamma', 0, 5),
        'scale_pos_weight': trial.suggest_float('xgb_scale_pos_weight', 1, 10)
    }
    
    # Hiperparámetros para Random Forest
    rf_params = {
        'n_estimators': trial.suggest_int('rf_n_estimators', 50, 300),
        'max_depth': trial.suggest_int('rf_max_depth', 3, 10),
        'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
        'class_weight': {0: 1, 1: trial.suggest_float('rf_class_weight', 1, 10)}
    }
    
    # Hiperparámetros para LightGBM
    lgbm_params = {
        'num_leaves': trial.suggest_int('lgbm_num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('lgbm_n_estimators', 50, 300),
        'subsample': trial.suggest_float('lgbm_subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('lgbm_colsample_bytree', 0.5, 1.0),
        'scale_pos_weight': trial.suggest_float('lgbm_scale_pos_weight', 1, 10)
    }

    # Crear los modelos
    xgb = XGBClassifier(**xgb_params, random_state=42)
    rf = RandomForestClassifier(**rf_params, random_state=42)
    lgbm = LGBMClassifier(**lgbm_params, random_state=42)

    # Crear el ensamblaje (Stacking)
    estimators = [('xgb', xgb), ('rf', rf), ('lgbm', lgbm)]
    stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)

    # Validación cruzada para evaluación del modelo
    score = cross_val_score(stack, X_train_resampled, y_train_resampled, cv=5, scoring='f1').mean()
    
    return score

# Ejecutar la optimización de Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Obtener los mejores parámetros
best_params = study.best_params

# Crear el modelo final con los mejores parámetros
xgb_final = XGBClassifier(**{k[4:]: v for k, v in best_params.items() if k.startswith('xgb_')}, random_state=42)
rf_final = RandomForestClassifier(**{k[3:]: v for k, v in best_params.items() if k.startswith('rf_')}, random_state=42)
lgbm_final = LGBMClassifier(**{k[5:]: v for k, v in best_params.items() if k.startswith('lgbm_')}, random_state=42)

# Crear el ensamblaje final (Stacking)
estimators_final = [('xgb', xgb_final), ('rf', rf_final), ('lgbm', lgbm_final)]
stack_final = StackingClassifier(estimators=estimators_final, final_estimator=LogisticRegression(), cv=5)

# Entrenar el modelo final
stack_final.fit(X_train_resampled, y_train_resampled)

# Predecir probabilidades
y_pred_proba = stack_final.predict_proba(X_test_preprocessed)[:, 1]

# Encontrar el mejor umbral
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

# Aplicar el umbral óptimo
y_pred_final = (y_pred_proba >= optimal_threshold).astype(int)

# Evaluar el modelo final
print("Informe de clasificación:")
print(classification_report(y_test, y_pred_final))
print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred_final))