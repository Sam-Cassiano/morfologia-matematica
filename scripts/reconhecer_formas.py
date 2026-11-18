"""Reconhecimento de formas: descritores morfológicos + classificação das folhas.

Uso (na raiz do projeto):
    python scripts/reconhecer_formas.py              # usa o cache se existir
    python scripts/reconhecer_formas.py --recalcular # refaz a extração dos atributos

Saídas em resultados/:
    formas_folhas.csv        atributos morfológicos de cada folha (cache)
    fig6_formas_regioes.png  manchas x nervuras reconhecidas pela forma
    fig7_granulometria.png   espectro de padrões médio por classe
    fig8_matriz_confusao.png matriz de confusão da árvore (5-fold)
Os números da classificação aparecem no terminal e no relatório (scripts/gerar_relatorio.py).
"""
import glob
import os
import sys
from multiprocessing import Pool

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, export_text

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))
from formas import RAIOS_GRANULOMETRIA, atributos_folha  # noqa: E402
from morfologia import binarizar_folha, binarizar_verde, kernel, limpar_mascara, segmentar_lesoes  # noqa: E402

DATA = os.path.join(RAIZ, "data")
OUT = os.path.join(RAIZ, "resultados")
CACHE = os.path.join(OUT, "formas_folhas.csv")
CLASSES = {"tomate_saudavel": "Tomate saudável",
           "tomate_pinta_preta": "Tomate – pinta-preta",
           "maca_sarna": "Maçã – sarna"}
AZUL, LARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA2, GRADE = "#1f1f1e", "#6b6a64", "#e4e3dc"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
                     "xtick.color": TINTA2, "ytick.color": TINTA2, "text.color": TINTA,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})

GRUPOS = {
    "Forma das lesões": ["n_manchas", "pct_manchas", "area_media_mancha", "maior_mancha_pct",
                         "circularidade_manchas", "solidez_manchas", "alongamento_manchas",
                         "n_nervuras", "pct_nervuras"] + [f"granulo_r{r}" for r in RAIOS_GRANULOMETRIA],
    "Forma da folha": ["folha_circularidade", "folha_solidez", "folha_alongamento", "folha_recortes"],
}
GRUPOS["Lesões + folha"] = GRUPOS["Forma das lesões"] + GRUPOS["Forma da folha"]
TODOS = GRUPOS["Lesões + folha"]


# ------------------------------------------------------------- extração
def processar(args):
    """Pipeline de uma folha + atributos (precisa ser de topo para o multiprocessing)."""
    classe, arq = args
    img = cv2.imread(os.path.join(DATA, "color", classe, arq))
    folha = limpar_mascara(binarizar_folha(img), kernel(5))["final"]
    les = segmentar_lesoes(binarizar_verde(img), folha, kernel(3))["limpa"]
    at, _, _ = atributos_folha(folha, les)
    at.update(classe=classe, arquivo=arq)
    return at


def extrair():
    tarefas = [(c, os.path.basename(f)) for c in CLASSES
               for f in sorted(glob.glob(os.path.join(DATA, "color", c, "*.jpg")))]
    print(f"extraindo atributos de {len(tarefas)} folhas...", flush=True)
    with Pool(max(1, os.cpu_count() or 1)) as pool:
        linhas = list(pool.imap(processar, tarefas, chunksize=32))
    df = pd.DataFrame(linhas)
    df = df[["classe", "arquivo"] + [c for c in df.columns if c not in ("classe", "arquivo")]]
    df.to_csv(CACHE, index=False)
    return df


# ------------------------------------------------------------- avaliação
def arvore():
    return DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, class_weight="balanced", random_state=0)


def cv5():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


def avaliar(X, y):
    pred = cross_val_predict(arvore(), X, y, cv=cv5())
    return pred, accuracy_score(y, pred), balanced_accuracy_score(y, pred)


def regra_unica(df):
    """Saudável x doente com um limiar em % de manchas, escolhido só nos folds de treino."""
    doente = (df["classe"].values != "tomate_saudavel").astype(int)
    x = df["pct_manchas"].values
    pred = np.zeros_like(doente)
    limiares = []
    for tr, te in cv5().split(x, doente):
        cand = np.unique(np.round(x[tr], 2))
        melhor = max(cand, key=lambda v: balanced_accuracy_score(doente[tr], (x[tr] >= v).astype(int)))
        limiares.append(melhor)
        pred[te] = (x[te] >= melhor).astype(int)
    return float(np.median(limiares)), accuracy_score(doente, pred), balanced_accuracy_score(doente, pred)


# --------------------------------------------------------------- figuras
def fig_regioes(df):
    """Exemplos: manchas (vermelho) x nervuras (azul) reconhecidas pela forma."""
    com_nervura = df[(df.n_nervuras > 0) & (df.n_manchas > 0)].sort_values("pct_nervuras", ascending=False)
    exemplos = [tuple(r) for r in com_nervura[com_nervura.classe == "maca_sarna"][["classe", "arquivo"]].head(2).values]
    exemplos += [tuple(r) for r in com_nervura[com_nervura.classe == "tomate_pinta_preta"][["classe", "arquivo"]].head(1).values]
    exemplos += [("tomate_pinta_preta", "0001.jpg")]
    fig, eixos = plt.subplots(len(exemplos), 3, figsize=(8.4, 2.8 * len(exemplos)))
    for linha, (c, a) in zip(eixos, exemplos):
        img = cv2.imread(os.path.join(DATA, "color", c, a))
        folha = limpar_mascara(binarizar_folha(img), kernel(5))["final"]
        les = segmentar_lesoes(binarizar_verde(img), folha, kernel(3))["limpa"]
        at, manchas, nervuras = atributos_folha(folha, les)
        over = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        over[manchas > 0] = (230, 40, 40)
        over[nervuras > 0] = (40, 110, 230)
        for ax, (t, im) in zip(linha, [(f"{CLASSES[c]} {a}", cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                                       ("Regiões candidatas", les),
                                       (f"{at['n_manchas']} manchas · {at['n_nervuras']} nervuras", over)]):
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None)
            ax.set_title(t, fontsize=10)
            ax.axis("off")
    fig.suptitle("Vermelho = mancha (forma compacta) · Azul = nervura (fina e alongada)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_formas_regioes.png"), dpi=150)
    plt.close(fig)


def fig_granulometria(df):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cols = [f"granulo_r{r}" for r in RAIOS_GRANULOMETRIA]
    for (c, nome), cor, mk in zip(CLASSES.items(), (AZUL, LARANJA, AQUA), ("o", "s", "^")):
        sub = df[(df.classe == c) & (df.n_manchas > 0)]
        ax.plot(RAIOS_GRANULOMETRIA, sub[cols].mean().values, color=cor, lw=2, marker=mk, ms=7,
                label=f"{nome} (n={len(sub)})")
    ax.set_xlabel("Raio do disco da abertura (px)")
    ax.set_ylabel("Fração da área de manchas que sobrevive")
    ax.set_xticks(RAIOS_GRANULOMETRIA)
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=GRADE, lw=0.8)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_granulometria.png"), dpi=150)
    plt.close(fig)


def fig_confusao(y, pred, acc):
    ordem = list(CLASSES)
    nomes = [CLASSES[c].replace(" – ", "\n") for c in ordem]
    cm = confusion_matrix(y, pred, labels=ordem)
    pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{100 * pct[i, j]:.1f}%", ha="center", va="center",
                    color="white" if pct[i, j] > 0.55 else TINTA, fontsize=11)
    ax.set_xticks(range(3), nomes)
    ax.set_yticks(range(3), nomes)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title(f"Árvore de decisão · {100 * acc:.1f}%", fontsize=11)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig8_matriz_confusao.png"), dpi=150)
    plt.close(fig)
    return cm


# ------------------------------------------------------------- resultado
def classificar(df):
    """Avalia a árvore por grupo de atributos, só tomate e a regra única. Retorna um dicionário."""
    y = df["classe"].values
    grupos = {nome: avaliar(df[cols].values, y) for nome, cols in GRUPOS.items()}
    pred, acc, bal = grupos["Lesões + folha"]
    t = df[df.classe.str.startswith("tomate")]
    _, acc_tom, _ = avaliar(t[GRUPOS["Forma das lesões"]].values, t.classe.values)
    limiar, acc_bin, bal_bin = regra_unica(df)
    final = arvore().fit(df[TODOS].values, y)
    return {"y": y, "pred": pred, "acc": acc, "bal": bal,
            "grupos": {n: (a, b) for n, (_, a, b) in grupos.items()},
            "cm": confusion_matrix(y, pred, labels=list(CLASSES)),
            "acc_tomate": acc_tom, "limiar": limiar, "acc_regra": acc_bin, "bal_regra": bal_bin,
            "regras": export_text(final, feature_names=TODOS, decimals=3)}


def carregar_atributos():
    """Lê o cache formas_folhas.csv (ou extrai de novo se não existir / --recalcular)."""
    usar_cache = os.path.exists(CACHE) and "--recalcular" not in sys.argv
    df = pd.read_csv(CACHE) if usar_cache else extrair()
    return df if set(TODOS).issubset(df.columns) else extrair()


# ------------------------------------------------------------------ main
def main():
    os.makedirs(OUT, exist_ok=True)
    if not glob.glob(os.path.join(DATA, "color", "*", "*.jpg")):
        sys.exit("Nenhuma imagem em data/color. Rode antes: python scripts/baixar_dataset.py")
    df = carregar_atributos()
    print("avaliando a árvore de decisão (5-fold)...", flush=True)
    r = classificar(df)
    fig_regioes(df)
    fig_granulometria(df)
    fig_confusao(r["y"], r["pred"], r["acc"])

    print(f"\n{len(df)} folhas · árvore de decisão (profundidade ≤ 4) · validação cruzada 5-fold")
    for nome, (a, b) in r["grupos"].items():
        print(f"  {nome:18s} acurácia {100 * a:5.1f}%   balanceada {100 * b:5.1f}%")
    print(f"  Só tomate (forma das lesões): {100 * r['acc_tomate']:.1f}%")
    print(f"  Regra única: doente se % de manchas >= {r['limiar']:.2f}% -> "
          f"acurácia {100 * r['acc_regra']:.1f}% (balanceada {100 * r['bal_regra']:.1f}%)")
    print("Figuras 6–8 salvas em", OUT)


if __name__ == "__main__":
    main()
