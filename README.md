# Car Evaluation – Classificação de Oportunidades de Compra

## Descrição do Projeto

Este projeto tem como objetivo desenvolver e avaliar um modelo de aprendizado de máquina capaz de identificar **oportunidades de compra de veículos** com base em características técnicas e econômicas.

O modelo utiliza o conjunto de dados **Car Evaluation**, disponibilizado pela **UCI Machine Learning Repository**, e é posteriormente avaliado com **dados reais coletados em uma concessionária**, permitindo verificar seu desempenho em um cenário prático e realista.

---

## Objetivo

Classificar veículos em duas categorias:

- **Oportunidade (1)**
- **Não Oportunidade (0)**

A classificação é baseada nos seguintes atributos:

- `buying` — preço de compra
- `maint` — custo de manutenção
- `doors` — número de portas
- `persons` — capacidade de passageiros
- `lug_boot` — tamanho do porta-malas
- `safety` — nível de segurança

---

## Fonte dos Dados

### Dataset Principal

- **Nome:** Car Evaluation
- **Origem:** UCI Machine Learning Repository
- **Instâncias:** 1.728 veículos
- **Atributos:** 6 atributos categóricos + 1 classe
- **Classes Originais:** `unacc`, `acc`, `good`, `vgood`

Para este projeto, as classes foram convertidas para um problema **binário**, conforme abaixo:

- `good`, `vgood` → **Oportunidade (1)**
- `unacc`, `acc` → **Não Oportunidade (0)**

### Dados Externos

Além do dataset original, foram utilizados **dados reais coletados manualmente em uma concessionária**, armazenados no arquivo:

- `carros_externos.xlsx`

Esses dados seguem o mesmo padrão de atributos do dataset original e incluem a coluna `class_real`, utilizada **exclusivamente para avaliação**, e não para treinamento do modelo.

---

## Metodologia

- **Modelo:** Naive Bayes Categórico (`CategoricalNB`)
- **Pré-processamento:**
  - Codificação manual de variáveis categóricas
  - Tratamento de valores ausentes com imputação
- **Validação:**
  - Validação cruzada estratificada com **5 folds**
- **Otimização:**
  - Ajuste manual de hiperparâmetros:
    - `alpha` (suavização de Laplace)
    - `threshold` (limiar de decisão para classificação)
- **Pipeline:**
  - Imputação → Modelo probabilístico → Decisão por threshold

---

## Avaliação do Modelo

### Métricas Utilizadas

- Acurácia
- Acurácia Balanceada
- F1-Score
- Matriz de Confusão

### Avaliação Interna (Dataset Original)

O modelo é avaliado em duas etapas:

1. **Validação Cruzada (5 folds)** — usada para seleção de hiperparâmetros
2. **Teste Final (Holdout 30%)** — conjunto isolado para avaliação final

Essas etapas garantem que o modelo seja avaliado de forma robusta e sem vazamento de dados.

### Avaliação com Dados Externos (Concessionária)

Os dados externos são carregados a partir do arquivo Excel e avaliados pelo modelo treinado, gerando:

- Classificação individual de cada veículo
- Matriz de confusão específica para os dados reais
- Métricas de desempenho:
  - Acurácia
  - Acurácia Balanceada
  - F1-Score

Essa etapa permite verificar **a capacidade de generalização do modelo** para dados do mundo real.

---

## Como Executar

1. Instale as dependências:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
```
2. Coloque o arquivo carros_externos.xlsx na mesma pasta do projeto.

3. Execute o script principal:

```bash
python car_evaluation.py
```

Ao final da execução, serão gerados automaticamente:
- Relatório textual com métricas (Relatorio_Final.txt)
- Matrizes de confusão em formato de imagem
- Resultados de classificação dos dados externos
