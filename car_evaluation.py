import pandas as pd
import os
import urllib.request
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import CategoricalNB
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# configuração e definição

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
DADOS = "data"
FILE_PATH = os.path.join(DADOS, "car.data")

COLUMNS = [
    "buying", "maint", "doors", "persons", "lug_boot", "safety", "class"
]

# download do dataset

os.makedirs(DADOS, exist_ok=True)

if not os.path.exists(FILE_PATH):
    print("Baixando o conjunto de dados Car Evaluation...")
    urllib.request.urlretrieve(DATA_URL, FILE_PATH)
    print("Download concluído.")
else:
    print("ARQUIVO JÁ EXISTENTE. CARREGANDO...")

# análise exploratória (EDA)

df = pd.read_csv(FILE_PATH, header=None, names=COLUMNS)

print("\n" + "="*60)
print("VERIFICAÇÃO DE INTEGRIDADE DOS DADOS")
print("="*60)
print(f"Dimensão do DataFrame: {df.shape}")
print(f"Duplicatas encontradas: {df.duplicated().sum()}")
print(f"Valores nulos totais: {df.isnull().sum().sum()}")

plt.figure(figsize=(8, 5))
ax = sns.countplot(
    x="class",
    data=df,
    order=df["class"].value_counts().index
)
plt.title("Distribuição das Classes Originais")
plt.xlabel("Classe")
plt.ylabel("Contagem")

total = len(df)
for p in ax.patches:
    percentage = f"{100 * p.get_height() / total:.1f}%"
    ax.annotate(
        percentage,
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()

# preparação dos dados

map_buying_maint = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
map_doors = {'2': 2, '3': 3, '4': 4, '5more': 5}
map_persons = {'2': 2, '4': 4, 'more': 5}
map_lug = {'small': 0, 'med': 1, 'big': 2}
map_safety = {'low': 0, 'med': 1, 'high': 2}

df_encoded = df.copy()
df_encoded['buying'] = df_encoded['buying'].map(map_buying_maint)
df_encoded['maint'] = df_encoded['maint'].map(map_buying_maint)
df_encoded['doors'] = df_encoded['doors'].map(map_doors)
df_encoded['persons'] = df_encoded['persons'].map(map_persons)
df_encoded['lug_boot'] = df_encoded['lug_boot'].map(map_lug)
df_encoded['safety'] = df_encoded['safety'].map(map_safety)

df_focus = df_encoded.copy()
df_focus['class'] = df_focus['class'].apply(lambda x: 1 if x in ['good', 'vgood'] else 0)

X = df_focus.drop('class', axis=1)
y = df_focus['class']

X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\nDivisão realizada com sucesso!")
print(f"Conjunto para Validação Cruzada (Treino): {X_train_full.shape[0]} amostras")
print(f"Conjunto para Teste Final: {X_test_final.shape[0]} amostras")

# validação cruzada manual com otimização de threshold 

print("\n" + "="*60)
print("INICIANDO VALIDAÇÃO CRUZADA ESTRATIFICADA (5 FOLDS)")
print("="*60)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
thresholds = np.arange(0.30, 0.80, 0.05)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("model", CategoricalNB(alpha=1.0, class_prior=[0.80, 0.20]))
])

best_global_f1 = -1
best_params = {'alpha': 1.0, 'threshold': 0.60}

# Grid Search Manual

for alpha in alphas:
    pipeline.set_params(model__alpha=alpha)
    
    thr_scores = {}
    
    for thr in thresholds:
        f1s = []
        for train_idx, val_idx in kf.split(X_train_full, y_train_full):
            X_fold_train = X_train_full.iloc[train_idx]
            X_fold_val = X_train_full.iloc[val_idx]
            y_fold_train = y_train_full.iloc[train_idx]
            y_fold_val = y_train_full.iloc[val_idx]

            pipeline.fit(X_fold_train, y_fold_train)
            probs = pipeline.predict_proba(X_fold_val)[:, 1]
            preds = (probs >= thr).astype(int)

            f1s.append(f1_score(y_fold_val, preds, pos_label=1))
        thr_scores[thr] = np.mean(f1s)
    
    current_best_thr = max(thr_scores, key=thr_scores.get)
    current_best_f1 = thr_scores[current_best_thr]
    
    if current_best_f1 > best_global_f1:
        best_global_f1 = current_best_f1
        best_params['alpha'] = alpha
        best_params['threshold'] = current_best_thr

best_alpha = best_params['alpha']
best_threshold = best_params['threshold']

print(f"Melhores parâmetros encontrados: Alpha={best_alpha}, Threshold={best_threshold:.2f}")

# reconfiguração do pipeline final

pipeline.set_params(model__alpha=best_alpha)

cv_metrics = {
    "accuracy": [],
    "balanced_accuracy": [],
    "f1_pos": []
}

for train_idx, val_idx in kf.split(X_train_full, y_train_full):
    X_fold_train = X_train_full.iloc[train_idx]
    X_fold_val = X_train_full.iloc[val_idx]
    y_fold_train = y_train_full.iloc[train_idx]
    y_fold_val = y_train_full.iloc[val_idx]

    pipeline.fit(X_fold_train, y_fold_train)
    probs = pipeline.predict_proba(X_fold_val)[:, 1]
    preds = (probs >= best_threshold).astype(int)

    cv_metrics["accuracy"].append(accuracy_score(y_fold_val, preds))
    cv_metrics["balanced_accuracy"].append(balanced_accuracy_score(y_fold_val, preds))
    cv_metrics["f1_pos"].append(f1_score(y_fold_val, preds, pos_label=1))

print(f"Acurácia Média (CV):            {np.mean(cv_metrics['accuracy']):.4f} (+/- {np.std(cv_metrics['accuracy']):.4f})")
print(f"Acurácia Balanceada Média (CV): {np.mean(cv_metrics['balanced_accuracy']):.4f}")
print(f"F1-Score Oportunidade (CV):     {np.mean(cv_metrics['f1_pos']):.4f}")

# treinamento e teste final

pipeline.set_params(model__alpha=best_alpha)
pipeline.fit(X_train_full, y_train_full)

X_test_final_imp = X_test_final.copy()
probs_final = pipeline.predict_proba(X_test_final_imp)[:, 1]
preds_final = (probs_final >= best_threshold).astype(int)

acc_final = accuracy_score(y_test_final, preds_final)
bal_acc_final = balanced_accuracy_score(y_test_final, preds_final)
f1_final = f1_score(y_test_final, preds_final, pos_label=1)

print("\n" + "="*60)
print("RESULTADO FINAL NO TESTE (HOLDOUT 30%)")
print("="*60)
print(f"Acurácia Final:            {acc_final:.4f}")
print(f"Acurácia Balanceada Final: {bal_acc_final:.4f}")
print(f"F1-Score Final:            {f1_final:.4f}")

# documentação e visualização

cm = confusion_matrix(y_test_final, preds_final)
tn, fp, fn, tp = cm.ravel()

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Não Oportunidade', 'Oportunidade'],
    yticklabels=['Não Oportunidade', 'Oportunidade']
)
plt.title('Matriz de Confusão Final (Teste)')
plt.ylabel('Real')
plt.xlabel('Predito')
plt.tight_layout()
plt.savefig("Matriz_Confusao_Final.png")
plt.show()

with open("Relatorio_Final.txt", "w", encoding="utf-8") as f:
    f.write("RELATÓRIO DE MODELAGEM - CAR EVALUATION\n")
    f.write("="*60 + "\n")
    f.write("Técnica: Naive Bayes Categórico + Validação Cruzada Manual\n")
    f.write(f"Melhor Hiperparâmetro (Alpha): {best_alpha}\n")
    f.write(f"Melhor Threshold (Corte): {best_threshold:.2f}\n\n")

    f.write("METRICAS DE VALIDAÇÃO CRUZADA (Média 5 Folds):\n")
    f.write(f"- Acurácia: {np.mean(cv_metrics['accuracy']):.2%}\n")
    f.write(f"- Acurácia Balanceada: {np.mean(cv_metrics['balanced_accuracy']):.2%}\n")
    f.write(f"- F1-Score: {np.mean(cv_metrics['f1_pos']):.2%}\n\n")

    f.write("METRICAS DE TESTE FINAL (30% Isolado):\n")
    f.write(f"- Acurácia: {acc_final:.2%}\n")
    f.write(f"- Acurácia Balanceada: {bal_acc_final:.2%}\n")
    f.write(f"- F1-Score: {f1_final:.2%}\n\n")

    f.write("MATRIZ DE CONFUSÃO (TESTE):\n")
    f.write(f"- TN: {tn}\n- TP: {tp}\n- FP: {fp}\n- FN: {fn}\n")

print("\n[SUCESSO] Relatório e Imagem gerados.")