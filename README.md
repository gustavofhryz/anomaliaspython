<img width="905" height="665" alt="output" src="https://github.com/user-attachments/assets/e7da569e-0e71-4945-a934-6ccd9b206711" />


# Projeto: Detecção de Anomalias em Transações de Cartão de Crédito

## 1. O Problema e o Desafio do Desbalanceamento
O problema central deste projeto reside na extrema raridade de transações fraudulentas em comparação com transações legítimas (geralmente cerca de 0,17% a 0,5% de fraudes no dataset). 
* **Por que o desbalanceamento muda a forma de avaliar?** Se criarmos um modelo "ingênuo" que preveja sempre "não fraude", ele terá uma acurácia de mais de 99%, mas será completamente inútil na prática, pois deixará passar 100% das fraudes. Portanto, a **acurácia deixa de ser uma métrica confiável**, exigindo o uso focado em **Recall (Revocação)**, **Precision (Precisão)** e **F1-Score** voltados para a classe minoritária (fraude).

## 2. Preparação dos Dados
* Remoção de colunas temporais irrelevantes (`Time`).
* Aplicação de transformação logarítmica (`np.log1p`) na coluna `Amount` para lidar com assimetria de valores monetários.
* Divisão dos dados de forma estratificada (`stratify=y`) para garantir que a proporção de fraudes seja mantida tanto no conjunto de treino quanto no de teste.
* Padronização das variáveis preditoras utilizando `StandardScaler`.

## 3. Comparação de Modelos e Métricas
* **Regressão Logística (Baseline):** Serviu como ponto de partida para estabelecer a linha base de performance linear.
* **XGBoost:** Apresentou capacidade superior de capturar padrões complexos e não-lineares nas transações.
* **Ajuste de Limiar (Threshold Tuning):** Em vez de utilizar o limiar padrão de 0.5, calculou-se a curva Precision-Recall para encontrar o limiar ideal que maximiza o F1-Score, equilibrando falsos positivos e falsos negativos de acordo com a criticidade do negócio.

## 4. Explicabilidade com SHAP
* A utilização do **SHAP (SHapley Additive exPlanations)** permitiu abrir a "caixa-preta" do modelo XGBoost, evidenciando visualmente quais variáveis (features anonimizadas por PCA e valor da transação) tiveram maior peso e influência na decisão de classificar uma determinada transação como fraude ou legítima.

# anomalias-python
