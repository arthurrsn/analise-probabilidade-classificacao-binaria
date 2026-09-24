import matplotlib
matplotlib.use("Agg")  # sem interface grafica, so pra testar sem travar

import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(y_true, y_prob, positive_class_definition, bin_width_pct=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    assert len(y_true) == len(y_prob), "cada instância precisa ter rótulo real e probabilidade correspondente"
    assert 100 % bin_width_pct == 0, "bin_width_pct precisa dividir 100 igualmente (ex: 5, 10, 20, 25)"

    y_prob_pct = y_prob * 100
    total_instancias = len(y_prob_pct)

    n_bins = 100 // bin_width_pct
    bin_edges = np.linspace(0, 100, n_bins + 1)

    contagem_total, _ = np.histogram(y_prob_pct, bins=bin_edges)
    contagem_positiva, _ = np.histogram(y_prob_pct[y_true == 1], bins=bin_edges)

    pct_total = contagem_total / total_instancias * 100
    pct_positiva = contagem_positiva / total_instancias * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(bin_edges[:-1], pct_total, where="post", color="blue", linewidth=2, label="Todas as instâncias")
    ax.step(bin_edges[:-1], pct_positiva, where="post", color="red", linewidth=2,
            label=f"Classe positiva ({positive_class_definition})")
    ax.set_xlabel("Probabilidade estimada da classe positiva (%)")
    ax.set_ylabel("Instâncias do conjunto avaliado (%)")
    ax.set_xlim(0, 100)
    ax.set_xticks(bin_edges)
    ax.legend()
    fig.savefig(r"C:\Users\arthu\AppData\Local\Temp\claude\D--Projects-squad\17b976ec-990c-40a3-ae57-a8d5a6b3f1cc\scratchpad\test_histogram.png")
    plt.close(fig)

    return fig, ax


def assign_decision_band(y_prob, t1, t2):
    assert t1 < t2, "t1 precisa ser menor que t2"
    y_prob = np.asarray(y_prob)
    faixa = np.full(len(y_prob), "analise_manual", dtype=object)
    faixa[y_prob < t1] = "negativo_automatico"
    faixa[y_prob >= t2] = "positivo_automatico"
    return faixa


import pandas as pd


def decision_band_report(y_true, y_prob, t1, t2):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    faixa = assign_decision_band(y_prob, t1, t2)

    total = len(y_prob)
    total_positivos_reais = (y_true == 1).sum()
    total_negativos_reais = (y_true == 0).sum()

    linhas = []
    for nome_faixa in ["negativo_automatico", "analise_manual", "positivo_automatico"]:
        na_faixa = faixa == nome_faixa
        n = na_faixa.sum()
        pos = ((y_true == 1) & na_faixa).sum()
        neg = ((y_true == 0) & na_faixa).sum()
        linhas.append({
            "Faixa": nome_faixa,
            "Instâncias": n,
            "% da população (n / total_teste)": round(n / total * 100, 2),
            "Positivos reais": pos,
            "% positivos na faixa (pos / n)": round(pos / n * 100, 2) if n else float("nan"),
            "Negativos reais": neg,
            "% negativos na faixa (neg / n)": round(neg / n * 100, 2) if n else float("nan"),
        })
    tabela = pd.DataFrame(linhas)

    positivos_perdidos = ((y_true == 1) & (faixa == "negativo_automatico")).sum()
    prop_positivos_perdidos = positivos_perdidos / total_positivos_reais

    negativos_mal_classificados = ((y_true == 0) & (faixa == "positivo_automatico")).sum()
    prop_negativos_mal_classificados = negativos_mal_classificados / total_negativos_reais

    print("Relatório por faixa de decisão (conjunto de teste):")
    print(tabela.to_string(index=False))
    print(
        f"\nPositivos reais classificados automaticamente como negativos: "
        f"{positivos_perdidos}/{total_positivos_reais} "
        f"({prop_positivos_perdidos * 100:.2f}%; denominador = total de positivos reais)"
    )
    print(
        f"Negativos reais classificados automaticamente como positivos: "
        f"{negativos_mal_classificados}/{total_negativos_reais} "
        f"({prop_negativos_mal_classificados * 100:.2f}%; denominador = total de negativos reais)"
    )

    return tabela


# ---- teste de reusabilidade: dataset e modelo diferentes do professor ----
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
# no load_breast_cancer, 0 = maligno, 1 = benigno -> definindo "maligno" como classe positiva
y_positivo_maligno = 1 - y

X_train, X_temp, y_train, y_temp = train_test_split(X, y_positivo_maligno, test_size=0.4, random_state=42, stratify=y_positivo_maligno)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

modelo = LogisticRegression(max_iter=5000)
modelo.fit(X_train, y_train)

indice_positivo = list(modelo.classes_).index(1)
y_prob_val = modelo.predict_proba(X_val)[:, indice_positivo]
y_prob_test = modelo.predict_proba(X_test)[:, indice_positivo]

print("=== Teste 1: plot_histogram (validação) ===")
plot_histogram(y_val, y_prob_val, positive_class_definition="é maligno?", bin_width_pct=10)
print("Gráfico salvo em test_histogram.png\n")

print("=== Teste 2: assign_decision_band ===")
faixas = assign_decision_band(y_prob_test, t1=0.2, t2=0.8)
print("Contagem por faixa:", {f: int((faixas == f).sum()) for f in set(faixas)})
print()

print("=== Teste 3: decision_band_report (teste) ===")
decision_band_report(y_test, y_prob_test, t1=0.2, t2=0.8)

print("\n=== Teste 4: asserts de validação (devem falhar como esperado) ===")
try:
    assign_decision_band(y_prob_test, t1=0.8, t2=0.2)
    print("ERRO: deveria ter lançado AssertionError")
except AssertionError as e:
    print(f"OK, capturou corretamente: {e}")

try:
    plot_histogram(y_test[:-1], y_prob_test, positive_class_definition="teste")
    print("ERRO: deveria ter lançado AssertionError")
except AssertionError as e:
    print(f"OK, capturou corretamente: {e}")

print("\nTODOS OS TESTES RODARAM SEM ERRO.")
