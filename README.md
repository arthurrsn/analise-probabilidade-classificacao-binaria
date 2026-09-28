# Análise de probabilidade em classificação binária

Trabalho A2 de Data Science.

**Integrantes:**

| Integrante | Parte |
|---|---|
| Arthur Ribeiro | Parte 1: programa de histograma e faixas de decisão |
| Eduardo Bernardes | Parte 2: domínio Doença Cardíaca (Heart Failure) |
| Gabriel Asserman | Parte 2: domínio AVC (Stroke) |
| Marcos Vitor | Parte 2: domínio Qualidade de Vinho Tinto |

O trabalho tem duas partes:

1. **Parte 1:** um programa reutilizável que desenha um histograma das probabilidades previstas por um modelo de classificação binária e ajuda a escolher dois pontos de corte (`t1 < t2`). Esses cortes dividem os casos em três faixas: negativo automático, análise manual e positivo automático.
2. **Parte 2:** construção e avaliação de modelos em três domínios diferentes, aplicando o programa da Parte 1 ao melhor modelo de cada um.

Tudo está no notebook `histograma-classificacao-binaria.ipynb`, com a explicação de cada decisão em células de texto antes e depois do código.

## Estrutura do repositório

```
histograma-classificacao-binaria.ipynb   notebook com a Parte 1 e os três domínios
data/
  heart.csv                              base do domínio Doença Cardíaca
  healthcare-dataset-stroke-data.csv     base do domínio AVC
  winequality-red.csv                    base do domínio Vinho (original da UCI)
  vinho_tinto_qualidade.csv              base do Vinho já com a variável-alvo
  vinho_previsoes_validacao.csv          y_true e y_prob do melhor modelo do Vinho (validação)
  vinho_previsoes_teste.csv              y_true e y_prob do melhor modelo do Vinho (teste)
modelo_avc_pronto.pkl                    modelo final do domínio AVC, salvo com joblib
```

O notebook lê as bases pelo caminho relativo `./data/`, então ele precisa ser executado a partir da pasta raiz do repositório.

## Como executar

### Opção 1: Google Colab (recomendado)

Os resultados que aparecem no notebook foram gerados no Google Colab.

1. Abrir o notebook no Colab:
   https://colab.research.google.com/github/arthurrsn/analise-probabilidade-classificacao-binaria/blob/main/histograma-classificacao-binaria.ipynb
2. O Colab abre só o notebook, sem a pasta `data/`. Para trazer os arquivos, criar uma célula de código no início e executar:
   ```python
   !git clone https://github.com/arthurrsn/analise-probabilidade-classificacao-binaria.git repo
   %cd repo
   ```
3. Executar tudo: menu **Ambiente de execução > Executar tudo**. A execução completa leva alguns minutos, porque cada domínio treina vários modelos com validação cruzada.

Todas as bibliotecas usadas já vêm instaladas no Colab.

### Opção 2: no computador

Requisitos: Python 3 e Jupyter.

```bash
git clone https://github.com/arthurrsn/analise-probabilidade-classificacao-binaria.git
cd analise-probabilidade-classificacao-binaria
pip install pandas numpy scikit-learn==1.6.1 matplotlib shap joblib notebook
jupyter notebook histograma-classificacao-binaria.ipynb
```

Depois, executar todas as células em ordem.

### Ordem de execução

As células precisam rodar de cima para baixo. As funções da Parte 1 ficam no início do notebook e são usadas pelos três domínios, então elas precisam ser executadas antes de qualquer domínio.

### Observação sobre reprodutibilidade

As divisões de dados e os modelos usam `random_state=42`, então os resultados se repetem a cada execução no mesmo ambiente. Em versões diferentes do scikit-learn, modelos com sorteio interno (como o Random Forest) podem dar números um pouco diferentes dos que estão escritos no texto do notebook. Por isso recomendamos o Colab.

## Bibliotecas

| Biblioteca | Uso |
|---|---|
| pandas | leitura das bases e tabelas de resultados |
| numpy | cálculos com vetores |
| scikit-learn | separação dos dados, pré-processamento, modelos, validação cruzada e métricas |
| matplotlib | gráficos (histogramas, curvas ROC e Precision-Recall, matriz de confusão) |
| shap | interpretação dos modelos (gráfico beeswarm) |
| joblib | exportação do modelo do domínio AVC |

## Parte 1: programa de faixas de decisão

O programa funciona com qualquer base e qualquer modelo de classificação binária que devolva a probabilidade da classe positiva. Ele recebe só dois vetores:

- `y_true`: o rótulo real de cada instância (0 ou 1)
- `y_prob`: a probabilidade estimada da classe positiva para **todas** as instâncias, qualquer que seja o rótulo real

| Função | O que faz |
|---|---|
| `assign_decision_band(y_prob, t1, t2)` | coloca cada instância em uma das três faixas: `negativo_automatico` (probabilidade < `t1`), `analise_manual` (`t1` ≤ probabilidade < `t2`) e `positivo_automatico` (probabilidade ≥ `t2`) |
| `plot_histogram(y_true, y_prob, t1, t2, positive_class_definition, bin_width_pct=10)` | desenha o histograma: eixo X com a probabilidade de 0% a 100%, eixo Y com o percentual de instâncias em cada intervalo. Azul = todas as instâncias; vermelho = instâncias cujo rótulo real é positivo. As duas usam os mesmos intervalos e o mesmo denominador (o total de instâncias). A largura dos intervalos é configurável com `bin_width_pct` (precisa dividir 100, por exemplo 5, 10, 20 ou 25) |
| `decision_band_report(y_true, y_prob, t1, t2)` | mostra, para cada faixa, a quantidade e o percentual da população, e a quantidade e a proporção de positivos e negativos reais. Também mostra a proporção de positivos classificados automaticamente como negativos (denominador: total de positivos reais) e a de negativos classificados automaticamente como positivos (denominador: total de negativos reais) |

Exemplo de uso:

```python
plot_histogram(y_val, y_prob_val, t1=0.15, t2=0.60, positive_class_definition="é vinho de alta qualidade?", bin_width_pct=5)
decision_band_report(y_teste, y_prob_teste, t1=0.15, t2=0.60)
```

Os cortes `t1` e `t2` são escolhidos olhando os dados de **validação** e depois aplicados ao conjunto de **teste** sem novos ajustes.

## Parte 2: domínios

| | Doença Cardíaca | AVC | Qualidade de Vinho Tinto |
|---|---|---|---|
| Responsável | Eduardo Bernardes | Gabriel Asserman | Marcos Vitor |
| Base | Heart Failure Prediction (918 pacientes) | Stroke Prediction (5.110 pacientes) | Wine Quality, vinhos tintos (1.599 vinhos) |
| Fonte | [Kaggle, fedesoriano](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) | [Kaggle, fedesoriano](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality) (Cortez et al., 2009) |
| Licença | Open Database License (ODbL) | Data Files © Original Authors (conforme a página da base no Kaggle) | CC BY 4.0 |
| Classe positiva | "o paciente tem doença cardíaca?" | "o paciente teve AVC?" | "o vinho é de alta qualidade?" (nota ≥ 7) |
| % de positivos | 55,3% (balanceada) | 4,9% (muito desbalanceada) | 13,6% (desbalanceada) |
| Algoritmos comparados | Regressão Logística, Árvore de Decisão, Random Forest | Regressão Logística, Random Forest, Hist Gradient Boosting | Regressão Logística, Árvore de Decisão, Random Forest |
| Métrica da otimização | ROC-AUC | Average Precision | Average Precision |
| Melhor modelo | Random Forest | Regressão Logística | Random Forest |
| Cortes (`t1` / `t2`) | 0,10 / 0,65 | 0,211 / 0,910 | 0,15 / 0,60 |

O que é igual nos três domínios:

- divisão estratificada em treino (60%), validação (20%) e teste (20%), com `random_state=42`
- pré-processamento dentro de um `Pipeline`, ajustado só nos dados de treino (inclusive dentro de cada divisão da validação cruzada), para evitar vazamento de dados
- otimização de hiperparâmetros com validação cruzada estratificada de 5 partes, só no treino
- avaliação com acurácia, precisão, recall, F1, AUC-ROC, AUC-PR, matriz de confusão e curvas ROC e Precision-Recall. As métricas que dependem de uma decisão sim ou não usam o limiar de 0,5
- gráfico SHAP beeswarm do melhor modelo
- escolha dos cortes `t1` e `t2` na validação e avaliação das três faixas no teste

As justificativas de cada escolha (métrica principal, algoritmos, cortes e resultados) estão explicadas no próprio notebook, na seção de cada domínio.
