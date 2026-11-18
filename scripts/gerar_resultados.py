"""Segmentação: gera as métricas e as figuras fig1–fig5 do projeto.

Uso (na raiz do repositório):
    python scripts/gerar_resultados.py
"""
import csv
import glob
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))
from morfologia import (binarizar_folha, binarizar_verde, iou, kernel, limpar_mascara,  # noqa: E402
                        mapa_distancia_fundo, mascara_gabarito, n_componentes, operadores_basicos,
                        pct_area, segmentar_lesoes)

DATA = os.path.join(RAIZ, "data")
OUT = os.path.join(RAIZ, "resultados")
os.makedirs(OUT, exist_ok=True)
if not glob.glob(os.path.join(DATA, "color", "*", "*.jpg")):
    sys.exit("Nenhuma imagem em data/color. Rode antes: python scripts/baixar_dataset.py")

CLASSES = {
    "tomate_saudavel": "Tomate saudável",
    "tomate_pinta_preta": "Tomate – pinta-preta",
    "maca_sarna": "Maçã – sarna",
}
K_LIMPEZA = kernel(5, "elipse")
K_LESAO = kernel(3, "elipse")

# paleta categórica (skill dataviz, slots 1-3) + tinta de texto
AZUL, LARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA2, GRADE = "#1f1f1e", "#6b6a64", "#e4e3dc"
plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "text.color": TINTA,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def ler(classe, arq):
    img = cv2.imread(os.path.join(DATA, "color", classe, arq))
    gt = mascara_gabarito(cv2.imread(os.path.join(DATA, "segmented", classe, arq)))
    return img, gt


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def grade_imagens(paineis, linhas, colunas, arquivo, titulos_linha=None, tam=2.6):
    fig, eixos = plt.subplots(linhas, colunas, figsize=(colunas * tam, linhas * tam + 0.3))
    eixos = np.atleast_2d(eixos)
    for ax, (titulo, im) in zip(eixos.flat, paineis):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=255)
        ax.set_title(titulo, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    if titulos_linha:
        for ax, t in zip(eixos[:, 0], titulos_linha):
            ax.set_ylabel(t, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, arquivo), dpi=150)
    plt.close(fig)


def sobrepor(img, mask, cor=(230, 40, 40)):
    out = rgb(img).copy()
    out[mask > 0] = (0.35 * out[mask > 0] + 0.65 * np.array(cor)).astype(np.uint8)
    return out


# ---------------------------------------------------------------- métricas
linhas = []
brutas = []  # (máscara bruta, gabarito) guardados para a varredura de kernels
for classe in CLASSES:
    for f in sorted(glob.glob(os.path.join(DATA, "color", classe, "*.jpg"))):
        arq = os.path.basename(f)
        img, gt = ler(classe, arq)
        bruta = binarizar_folha(img)
        brutas.append((bruta, gt))
        final = limpar_mascara(bruta, K_LIMPEZA)["final"]
        les = segmentar_lesoes(binarizar_verde(img), final, K_LESAO)
        if len(linhas) % 500 == 0:
            print("métricas:", len(linhas), flush=True)
        linhas.append({
            "classe": classe, "arquivo": arq,
            "iou_bruta": round(iou(bruta, gt), 4),
            "iou_limpa": round(iou(final, gt), 4),
            "comp_bruta": n_componentes(bruta),
            "comp_limpa": n_componentes(final),
            "n_lesoes": les["n_lesoes"],
            "pct_lesao": round(pct_area(les["limpa"], final), 2),
        })

with open(os.path.join(OUT, "metricas.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
    w.writeheader(); w.writerows(linhas)

resumo = ["| Classe | IoU bruta | IoU limpa | Componentes (bruta → limpa) | Nº lesões | % área lesionada |",
          "|---|---|---|---|---|---|"]
for classe, nome in CLASSES.items():
    L = [r for r in linhas if r["classe"] == classe]
    m = lambda k: np.mean([r[k] for r in L])  # noqa: E731
    resumo.append(f"| {nome} | {m('iou_bruta'):.3f} | {m('iou_limpa'):.3f} | "
                  f"{m('comp_bruta'):.1f} → {m('comp_limpa'):.1f} | {m('n_lesoes'):.1f} | {m('pct_lesao'):.2f} |")
print("\n".join(resumo))

# ------------------------------------------- fig 1: operadores básicos
img, _ = ler("tomate_pinta_preta", "0001.jpg")
mask = binarizar_folha(img)
ops = operadores_basicos(mask, kernel(9, "elipse"))
grade_imagens([("Original", rgb(img)), ("Máscara (Otsu)", mask),
               ("Erosão", ops["erosao"]), ("Dilatação", ops["dilatacao"]),
               ("Abertura", ops["abertura"]), ("Fechamento", ops["fechamento"])],
              2, 3, "fig1_operadores_basicos.png")

# ------------------------------------------------------ fig 2: pipeline
paineis = []
for classe in CLASSES:
    img, _ = ler(classe, "0001.jpg")
    dist = mapa_distancia_fundo(img)
    bruta = binarizar_folha(img)
    final = limpar_mascara(bruta, K_LIMPEZA)["final"]
    les = segmentar_lesoes(binarizar_verde(img), final, K_LESAO)
    paineis += [("Original", rgb(img)), ("Distância ao fundo", dist),
                ("Otsu", bruta), ("Após morfologia", final),
                (f"Lesões: {les['n_lesoes']}", sobrepor(img, les["limpa"]))]
grade_imagens(paineis, 3, 5, "fig2_pipeline.png", titulos_linha=list(CLASSES.values()))

# ------------------------------------- fig 3: efeito do tamanho do kernel
img, _ = ler("tomate_pinta_preta", "0001.jpg")
mask = binarizar_folha(img)
paineis = []
for op, nome in ((cv2.MORPH_OPEN, "Abertura"), (cv2.MORPH_CLOSE, "Fechamento")):
    for t in (3, 9, 21):
        paineis.append((f"{nome} {t}×{t}", cv2.morphologyEx(mask, op, kernel(t, "elipse"))))
grade_imagens(paineis, 2, 3, "fig3_tamanho_kernel.png")

# --------------------------------- fig 4: IoU x tamanho/forma do kernel
tamanhos = [1, 3, 5, 7, 9, 11, 15, 21, 31]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
curvas = {}
for (forma, cor, marc) in (("elipse", AZUL, "o"), ("retangulo", LARANJA, "s"), ("cruz", AQUA, "^")):
    ys = []
    print("varredura:", forma, flush=True)
    for t in tamanhos:
        k = kernel(t, forma)
        ys.append(np.mean([iou(limpar_mascara(b, k)["final"], gt) for b, gt in brutas]))
    curvas[forma] = ys
    ax.plot(tamanhos, ys, color=cor, lw=2, marker=marc, ms=7, label=forma.replace("retangulo", "retângulo"))
    ax.annotate(forma.replace("retangulo", "retângulo"), (tamanhos[-1], ys[-1]),
                xytext=(6, 0), textcoords="offset points", va="center", color=TINTA2)
ax.set_xlabel("Tamanho do elemento estruturante (px)")
ax.set_ylabel("IoU médio vs. gabarito")
ax.set_xticks(tamanhos)
ax.grid(axis="y", color=GRADE, lw=0.8)
ax.legend(frameon=False, loc="lower left")
ax.set_xlim(0, 36)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_iou_kernel.png"), dpi=150)
plt.close(fig)
with open(os.path.join(OUT, "iou_por_kernel.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["tamanho"] + list(curvas))
    for i, t in enumerate(tamanhos):
        w.writerow([t] + [round(curvas[f][i], 4) for f in curvas])

# ----------------------------------------- fig 5: % área lesionada/classe
Y_MAX = 30  # eixo cortado para não esmagar a maioria; pontos acima são contados
fig, ax = plt.subplots(figsize=(7.5, 4.0))
rng = np.random.default_rng(0)
for i, (classe, nome) in enumerate(CLASSES.items()):
    v = [r["pct_lesao"] for r in linhas if r["classe"] == classe]
    ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=8, color=AZUL, alpha=0.3,
               linewidth=0, zorder=3)
    ax.hlines(np.mean(v), i - 0.28, i + 0.28, color=TINTA, lw=2.5, zorder=4)
    ax.annotate(f"média {np.mean(v):.2f}%\nn = {len(v)}", (i + 0.3, max(np.mean(v), 2.5)), va="center", color=TINTA2, fontsize=10)
    fora = sum(x > Y_MAX for x in v)
    if fora:
        ax.annotate(f"{fora} acima de {Y_MAX}%", (i, Y_MAX), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", color=TINTA2, fontsize=9)
ax.set_ylim(-1, Y_MAX)
ax.set_xticks(range(len(CLASSES)), list(CLASSES.values()))
ax.set_ylabel("% da folha com lesão")
ax.grid(axis="y", color=GRADE, lw=0.8)
ax.set_xlim(-0.5, len(CLASSES) - 0.05)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_lesoes_por_classe.png"), dpi=150)
plt.close(fig)

print("Figuras salvas em", OUT)
