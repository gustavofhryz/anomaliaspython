import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, confusion_matrix
import xgboost as xgb
import shap

# 1. CARREGAMENTO DOS DADOS
# Usaremos um dataset público clássico de transações de cartão de crédito (Credit Card Fraud Detection)
print("Carregando os dados...")
url = "https://raw.githubusercontent.com/nsethi/Credit-Card-Fraud-Detection/master/creditcard.csv"
try:
    df = pd.read_csv(url)
except Exception as e:
    # Fallback caso haja falha na URL, criando um DataFrame sintético para demonstração estrutural
    print("Erro ao carregar URL externa, gerando base sintética para exemplo:", e)
    from sklearn.datasets import make_classification
    X_synth, y_synth = make_classification(n_samples=5000, n_features=30, weights=[0.995, 0.005], random_state=42)
    df = pd.DataFrame(X_synth, columns=[f"V{i}" for i in range(1, 31)])
    df['Class'] = y_synth
    df['Amount'] = np.random.uniform(1.0, 1000.0, size=len(df))

print(f"Dimensões do Dataset: {df.shape}")
print(f"Distribuição de Fraudes:\n{df['Class'].value_counts(normalize=True) * 100}%")

# 2. PREPARAÇÃO DOS DADOS
# 'Time' e 'Amount' precisam de tratamento. O Amount costuma passar por log-transform para suavizar assimetria.
if 'Time' in df.columns:
    df = df.drop(columns=['Time'])

if 'Amount' in df.columns:
    df['Amount_Log'] = np.log1p(df['Amount'])
    df = df.drop(columns=['Amount'])

X = df.drop(columns=['Class'])
y = df['Class']

# Divisão de treino e teste estratificada (mantendo a proporção de fraudes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Padronização das variáveis numéricas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. TREINAMENTO DE MODELOS (Baseline + Avançado)
print("\nTreinando Regressão Logística (Baseline)...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

print("Treinando XGBoost...")
xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=199) # Ajuste básico para desbalanceamento
xgb_model.fit(X_train_scaled, y_train)

# 4. AVALIAÇÃO E AJUSTE DE LIMIAR (THRESHOLD)
print("\nAvaliando Modelos...")
y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Curva Precision-Recall para encontrar o melhor limiar de decisão
precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Melhor Limiar (Threshold) encontrado por F1-Score: {best_threshold:.4f}")

# Aplicando o novo limiar
y_pred_custom = (y_pred_proba >= best_threshold).astype(int)

print("\nRelatório de Classificação (XGBoost com Threshold Ajustado):")
print(classification_report(y_test, y_pred_custom))
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# 5. EXPLICABILIDADE COM SHAP
print("\nCalculando valores SHAP para interpretabilidade...")
# Usando uma amostra menor para agilizar o cálculo do SHAP
explainer = shap.Explainer(xgb_model, X_train_scaled)
shap_values = explainer(X_test_scaled[:100])

print("Pipeline executado com sucesso! Pronto para geração do relatório.")