"""Gera o relatório do projeto em um único PDF: resultados/relatorio.pdf.

Uso (na raiz do projeto, depois de gerar_resultados.py e reconhecer_formas.py):
    python scripts/gerar_relatorio.py
    python scripts/gerar_relatorio.py --autor "Seu Nome"

Lê resultados/metricas.csv, resultados/iou_por_kernel.csv, resultados/formas_folhas.csv
e as figuras fig1–fig8. Usa só matplotlib (nenhuma biblioteca extra).
"""
import argparse
import datetime as dt
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
from reconhecer_formas import CACHE, CLASSES, OUT, classificar  # noqa: E402

A4 = (8.27, 11.69)
AZUL, TINTA, TINTA2, GRADE, FUNDO = "#2a78d6", "#1f1f1e", "#6b6a64", "#d9d8d0", "#eef4fc"


def num(x, casas=3):
    return f"{x:.{casas}f}".replace(".", ",")


def pct(x, casas=1):
    return num(100 * x, casas) + "%"


# ------------------------------------------------------------- página
class Pagina:
    """Página A4 com um cursor vertical (y em fração da altura, de cima para baixo)."""

    def __init__(self, pdf, n):
        self.pdf, self.n = pdf, n
        self.fig = plt.figure(figsize=A4)
        self.y = 0.94
        self.fig.text(0.08, 0.035, "PDSI · Projeto 6 — Morfologia Matemática Aplicada", fontsize=7.5, color=TINTA2)
        self.fig.text(0.92, 0.035, str(n), fontsize=7.5, color=TINTA2, ha="right")

    def titulo(self, texto):
        self.fig.text(0.08, self.y, texto, fontsize=15, weight="bold", color=TINTA, va="top")
        self.y -= 0.032
        self.fig.add_artist(plt.Line2D([0.08, 0.92], [self.y, self.y], color=AZUL, lw=1.5))
        self.y -= 0.018

    def subtitulo(self, texto):
        self.fig.text(0.08, self.y, texto, fontsize=11.5, weight="bold", color=AZUL, va="top")
        self.y -= 0.026

    def paragrafo(self, texto, largura=98, tam=9.5):
        linhas = textwrap.wrap(texto, largura)
        self.fig.text(0.08, self.y, "\n".join(linhas), fontsize=tam, color=TINTA, va="top", linespacing=1.45)
        self.y -= 0.0172 * len(linhas) * tam / 9.5 + 0.012

    def itens(self, lista, largura=94):
        for item in lista:
            linhas = textwrap.wrap(item, largura)
            self.fig.text(0.09, self.y, "•", fontsize=9.5, color=AZUL, va="top")
            self.fig.text(0.11, self.y, "\n".join(linhas), fontsize=9.5, color=TINTA, va="top", linespacing=1.45)
            self.y -= 0.0172 * len(linhas) + 0.006
        self.y -= 0.006

    def tabela(self, cab, linhas, larg_col=None, negrito_ultima=False):
        h = 0.024 * (len(linhas) + 1)
        ax = self.fig.add_axes([0.08, self.y - h, 0.84, h])
        ax.axis("off")
        tb = ax.table(cellText=linhas, colLabels=cab, loc="upper center", cellLoc="center",
                      colWidths=larg_col, bbox=[0, 0, 1, 1])
        tb.auto_set_font_size(False)
        tb.set_fontsize(8.6)
        for (i, j), c in tb.get_celld().items():
            c.set_edgecolor(GRADE)
            if i == 0:
                c.set_facecolor(FUNDO)
                c.set_text_props(weight="bold")
            if j == 0 and i > 0:
                c._loc = "left"
                c.PAD = 0.03
            if negrito_ultima and i == len(linhas):
                c.set_text_props(weight="bold")
        self.y -= h + 0.02

    def imagem(self, arquivo, largura=0.84, legenda=None):
        img = mpimg.imread(os.path.join(OUT, arquivo))
        alt_px, larg_px = img.shape[:2]
        h = largura * A4[0] * alt_px / larg_px / A4[1]
        x0 = 0.08 + (0.84 - largura) / 2
        ax = self.fig.add_axes([x0, self.y - h, largura, h])
        ax.imshow(img)
        ax.axis("off")
        self.y -= h + 0.008
        if legenda:
            self.fig.text(0.5, self.y, legenda, fontsize=8, color=TINTA2, ha="center", va="top", style="italic")
            self.y -= 0.022

    def fechar(self):
        self.pdf.savefig(self.fig)
        plt.close(self.fig)


# ------------------------------------------------------------ relatório
def gerar(autor):
    m = pd.read_csv(os.path.join(OUT, "metricas.csv"))
    kern = pd.read_csv(os.path.join(OUT, "iou_por_kernel.csv")).set_index("tamanho")
    df = pd.read_csv(CACHE)
    print("avaliando a classificação para o relatório...", flush=True)
    r = classificar(df)

    n = len(m)
    piorou = int((m.iou_limpa < m.iou_bruta).sum())
    melhor_t, melhor_f = kern.stack().idxmax()
    melhor_v = kern.stack().max()
    iou_k1 = kern.loc[1, "elipse"]   # 1×1 = sem abertura/fechamento (só preenchimento + maior componente)
    iou_k5 = kern.loc[5, "elipse"]   # kernel usado no pipeline
    n_fmt = f"{n:,}".replace(",", ".")
    g = m.groupby("classe", sort=False)
    granulo = df[df.n_manchas > 0].groupby("classe")["granulo_r2"].mean()
    destino = os.path.join(OUT, "relatorio.pdf")

    with PdfPages(destino) as pdf:
        # ------------------------------------------------ 1. capa + resumo
        p = Pagina(pdf, 1)
        p.fig.text(0.08, 0.95, "PDSI · PROJETO 6", fontsize=9, color=AZUL, weight="bold", va="top")
        p.fig.text(0.08, 0.925, "Morfologia Matemática Aplicada", fontsize=22, weight="bold", color=TINTA, va="top")
        p.fig.text(0.08, 0.885, "Limpeza de máscaras e reconhecimento de formas em folhas do PlantVillage",
                   fontsize=11.5, color=TINTA2, va="top")
        p.fig.text(0.08, 0.86, f"{autor} · IFAL · {dt.date.today():%d/%m/%Y}", fontsize=9.5, color=TINTA2, va="top")
        p.y = 0.82
        p.subtitulo("Objetivo")
        p.paragrafo("Aplicar erosão, dilatação, abertura e fechamento em imagens reais de folhas para (1) corrigir "
                    "falhas nas máscaras binárias obtidas por limiarização e (2) reconhecer formas: separar manchas "
                    "de nervuras e classificar a folha pela forma das lesões.")
        p.subtitulo("Dados")
        p.tabela(["Classe", "Pasta no PlantVillage", "Imagens"],
                 [[CLASSES[c], o, f"{len(g.get_group(c)):,}".replace(",", ".")] for c, o in
                  (("tomate_saudavel", "Tomato___healthy"), ("tomate_pinta_preta", "Tomato___Early_blight"),
                   ("maca_sarna", "Apple___Apple_scab"))] + [["Total", "", f"{n:,}".replace(",", ".")]],
                 [0.38, 0.42, 0.2], negrito_ultima=True)
        p.paragrafo("Fotos 256 × 256 px, uma folha por imagem. A versão “segmented” do PlantVillage (folha sem fundo) "
                    "é usada como gabarito para medir a qualidade das máscaras (IoU). Fonte: "
                    "github.com/spMohanty/PlantVillage-Dataset (Hughes & Salathé, 2015).", tam=8.8)
        p.subtitulo("Método")
        p.itens(["Binarização: distância de cor (a*, b* do Lab) até a cor do fundo, seguida de Otsu — a sombra, "
                 "que tem a cor do fundo, fica fora da folha.",
                 "Limpeza: abertura → fechamento (elipse 5×5) → preenchimento de buracos → maior componente conexo.",
                 "Lesões: pixels da folha que não são verdes (Otsu no canal a*), abertura 3×3 e descarte de regiões "
                 "com menos de 15 px.",
                 "Formas: cada região recebe descritores (circularidade, solidez, alongamento, espessura e fração que "
                 "sobrevive a uma abertura 5×5); finas e alongadas são nervuras, as demais são manchas. Por folha: "
                 "granulometria das manchas e forma da folha, usadas numa árvore de decisão (5-fold)."])
        p.subtitulo("Principais resultados")
        p.itens([f"A limpeza deixa cada máscara com um único objeto: {num(m.comp_bruta.mean(), 1)} → "
                 f"{num(m.comp_limpa.mean(), 1)} componentes em média; IoU {num(m.iou_bruta.mean())} → "
                 f"{num(m.iou_limpa.mean())}.",
                 f"Melhor elemento estruturante na varredura: {melhor_f.replace('retangulo', 'retângulo')} "
                 f"{melhor_t}×{melhor_t} (IoU {num(melhor_v, 4)}); o pipeline usa elipse 5×5 (IoU {num(iou_k5, 4)}), "
                 f"diferença desprezível. Acima de ≈ 7 px o IoU cai.",
                 f"Árvore de decisão com atributos de forma: {pct(r['acc'])} de acurácia nas 3 classes; "
                 f"regra única “manchas ≥ {num(r['limiar'], 2)}% → doente”: {pct(r['acc_regra'])}."])
        p.fechar()

        # ------------------------------------------------ 2. segmentação
        p = Pagina(pdf, 2)
        p.titulo("1. Correção das máscaras binárias")
        p.paragrafo("A máscara bruta (Otsu) costuma ter ruído no fundo, buracos nas lesões e pedaços soltos. "
                    "A sequência abertura → fechamento → preenchimento → maior componente resolve os três problemas. "
                    "A tabela compara a máscara antes e depois da limpeza com o gabarito.")
        linhas = []
        for c in CLASSES:
            x = g.get_group(c)
            linhas.append([CLASSES[c], num(x.iou_bruta.mean()), num(x.iou_limpa.mean()),
                           f"{num(x.comp_bruta.mean(), 1)} → {num(x.comp_limpa.mean(), 1)}",
                           num(x.n_lesoes.mean(), 1), num(x.pct_lesao.mean(), 2) + "%"])
        linhas.append(["Todas", num(m.iou_bruta.mean()), num(m.iou_limpa.mean()),
                       f"{num(m.comp_bruta.mean(), 1)} → {num(m.comp_limpa.mean(), 1)}",
                       num(m.n_lesoes.mean(), 1), num(m.pct_lesao.mean(), 2) + "%"])
        p.tabela(["Classe", "IoU bruta", "IoU limpa", "Componentes", "Nº lesões", "% lesionada"], linhas,
                 [0.26, 0.13, 0.13, 0.18, 0.13, 0.17], negrito_ultima=True)
        p.paragrafo(f"O ganho em IoU é pequeno porque o Otsu já acerta a maior parte da folha; o ganho principal é "
                    f"topológico (um objeto só). A maior melhora é na pinta-preta, cujas lesões escuras abrem buracos "
                    f"na máscara. Em {piorou} de {n:,} imagens o IoU caiu um pouco, em geral porque a abertura apaga "
                    f"pontas finas da folha.".replace(",", "."))
        p.imagem("fig2_pipeline.png", 0.84, "Figura 1 — original → distância ao fundo → Otsu → após morfologia → lesões.")
        p.fechar()

        # ------------------------------------------------ 3. operadores
        p = Pagina(pdf, 3)
        p.titulo("2. Operadores e elemento estruturante")
        p.paragrafo("Erosão encolhe o objeto e aumenta os buracos; dilatação faz o contrário. A abertura (erosão → "
                    "dilatação) remove ruído menor que o elemento estruturante; o fechamento (dilatação → erosão) "
                    "fecha buracos e fendas — ambos quase sem mudar o tamanho da folha.")
        p.imagem("fig1_operadores_basicos.png", 0.7, "Figura 2 — os quatro operadores com elemento elíptico 9×9.")
        p.imagem("fig3_tamanho_kernel.png", 0.66, "Figura 3 — abertura e fechamento com elementos de 3, 9 e 21 px.")
        p.fechar()

        p = Pagina(pdf, 4)
        p.subtitulo("Varredura de 27 elementos estruturantes")
        p.paragrafo(f"Foram testadas 3 formas (elipse, retângulo, cruz) × 9 tamanhos (1 a 31 px) em todas as {n_fmt} "
                    f"folhas. O melhor resultado é {melhor_f.replace('retangulo', 'retângulo')} {melhor_t}×{melhor_t} "
                    f"(IoU {num(melhor_v, 4)}); o pipeline usa elipse 5×5 (IoU {num(iou_k5, 4)}), praticamente igual. "
                    f"Elementos grandes apagam pontas e dentes da folha, e a cruz degrada menos que o retângulo por "
                    f"remover menos pixels nos cantos.")
        p.paragrafo(f"Com 1×1 (sem abertura nem fechamento, só preenchimento de buracos e maior componente) o IoU já é "
                    f"{num(iou_k1, 4)}: o ganho de IoU vem do preenchimento e do maior componente. A abertura e o "
                    f"fechamento pequenos quase não mudam o IoU; o papel deles é tirar ruído e unir pedaços da folha "
                    f"antes desses dois passos, e elementos grandes passam a prejudicar.")
        p.tabela(["Tamanho (px)"] + ["Elipse", "Retângulo", "Cruz"],
                 [[str(t), num(kern.loc[t, "elipse"]), num(kern.loc[t, "retangulo"]), num(kern.loc[t, "cruz"])]
                  for t in kern.index], [0.25, 0.25, 0.25, 0.25])
        p.imagem("fig4_iou_kernel.png", 0.66, "Figura 4 — IoU médio × tamanho e forma do elemento estruturante.")
        p.fechar()

        # ------------------------------------------------ 5. lesões
        p = Pagina(pdf, 5)
        p.titulo("3. Lesões")
        p.paragrafo(f"A porcentagem da folha com lesão já separa as classes: média de "
                    f"{num(g.get_group('tomate_saudavel').pct_lesao.mean(), 2)}% nas folhas saudáveis contra "
                    f"{num(g.get_group('tomate_pinta_preta').pct_lesao.mean(), 2)}% na pinta-preta e "
                    f"{num(g.get_group('maca_sarna').pct_lesao.mean(), 2)}% na sarna da maçã.")
        p.imagem("fig5_lesoes_por_classe.png", 0.8, "Figura 5 — % da folha com lesão, por classe (cada ponto é uma folha).")
        p.paragrafo("As lesões são os pixels da folha que não são tecido verde. Antes de procurá-las, a folha é erodida "
                    "com uma elipse 7×7 (≈ 3 px) para ignorar a borda (onde a sombra e o contorno criariam falsas lesões); depois, uma abertura "
                    "3×3 remove pontos isolados e regiões com menos de 15 px são descartadas.")
        p.fechar()

        p = Pagina(pdf, 6)
        p.titulo("4. Reconhecimento de formas")
        p.paragrafo("Nem toda região “não verde” é doença: na maçã, a nervura central também aparece. A regra de forma "
                    "separa as duas: região fina (no máximo 35% da área sobrevive à abertura 5×5, ou espessura ≤ 7 px) e alongada "
                    "(alongamento ≥ 3) é nervura; o resto é mancha.")
        p.imagem("fig6_formas_regioes.png", 0.46, "Figura 6 — vermelho = mancha (compacta), azul = nervura (fina e alongada).")
        p.subtitulo("Granulometria")
        p.paragrafo("A granulometria mede o tamanho das manchas sem separá-las: é a fração da área que sobrevive a "
                    "aberturas com discos de raio crescente. Na abertura de raio 2 sobrevivem "
                    f"{num(granulo['tomate_pinta_preta'], 2)} das manchas da pinta-preta, "
                    f"{num(granulo['maca_sarna'], 2)} da sarna e {num(granulo['tomate_saudavel'], 2)} das "
                    "manchas (falsas, pequenas) das folhas saudáveis.")
        p.imagem("fig7_granulometria.png", 0.5, "Figura 7 — espectro de padrões médio por classe.")
        p.fechar()

        # ------------------------------------------------ 7. classificação
        p = Pagina(pdf, 7)
        p.subtitulo("Classificação das folhas")
        p.paragrafo("Árvore de decisão de profundidade ≤ 4 (regras legíveis), validação cruzada estratificada 5-fold.")
        p.tabela(["Atributos", "Acurácia", "Acurácia balanceada"],
                 [[nome, pct(a), pct(b)] for nome, (a, b) in r["grupos"].items()]
                 + [["Só tomate (saudável × pinta-preta)", pct(r["acc_tomate"]), "—"],
                    [f"Regra única: manchas ≥ {num(r['limiar'], 2)}%", pct(r["acc_regra"]), pct(r["bal_regra"])]],
                 [0.5, 0.2, 0.3])
        p.imagem("fig8_matriz_confusao.png", 0.55, "Figura 8 — matriz de confusão da árvore (lesões + folha).")
        p.fechar()

        # ------------------------------------------------ 8. conclusões
        p = Pagina(pdf, 8)
        p.titulo("5. Conclusões e limitações")
        p.itens([f"Abertura + fechamento + preenchimento + maior componente transformam a máscara bruta em um objeto "
                 f"único em todas as {n_fmt} imagens, com IoU médio {num(m.iou_limpa.mean())}.",
                 "Elementos estruturantes pequenos (3–7 px) funcionam melhor; o tamanho deve ser menor que os "
                 "detalhes do contorno que se quer preservar.",
                 "Descritores morfológicos simples — sobrevivência à abertura, espessura, granulometria — bastam para "
                 "separar manchas de nervuras e para classificar a folha com regras interpretáveis "
                 f"({pct(r['acc'])} com a árvore, {pct(r['acc_regra'])} com uma única regra).",
                 "Limitações: lesões escuras na borda da folha ficam fora (a folha é erodida ≈ 3 px, elipse 7×7, antes de "
                 "procurar lesões); as manchas verde-oliva da sarna quase não aparecem no canal a*; as fotos são de laboratório, "
                 "com fundo uniforme.",
                 "Próximos passos: reconstrução morfológica para lesões na borda, esqueletização das nervuras e "
                 "top-hat/black-hat para as manchas escuras da sarna."])
        p.subtitulo("Como reproduzir")
        p.paragrafo("python scripts/baixar_dataset.py  →  python scripts/gerar_resultados.py  →  "
                    "python scripts/reconhecer_formas.py  →  python scripts/gerar_relatorio.py", tam=8.8)
        p.subtitulo("Regras da árvore de decisão (treinada em todas as folhas)")
        regras = r["regras"].replace("tomate_pinta_preta", "pinta-preta").replace("tomate_saudavel", "saudável") \
                            .replace("maca_sarna", "sarna").replace("class:", "classe:")
        p.fig.text(0.08, p.y, regras, fontsize=6.3, family="monospace", color=TINTA, va="top", linespacing=1.25)
        p.fechar()

        info = pdf.infodict()
        info["Title"] = "Morfologia Matemática Aplicada — Relatório"
        info["Author"] = autor
    return destino


def main():
    ap = argparse.ArgumentParser(description="Gera resultados/relatorio.pdf")
    ap.add_argument("--autor", default="Samuel Cassiano dos Santos")
    a = ap.parse_args()
    for f in ("metricas.csv", "iou_por_kernel.csv", "fig5_lesoes_por_classe.png"):
        if not os.path.exists(os.path.join(OUT, f)):
            sys.exit(f"Falta resultados/{f}. Rode antes: python scripts/gerar_resultados.py")
    for f in ("formas_folhas.csv", "fig8_matriz_confusao.png"):
        if not os.path.exists(os.path.join(OUT, f)):
            sys.exit(f"Falta resultados/{f}. Rode antes: python scripts/reconhecer_formas.py")
    print("Relatório salvo em", gerar(a.autor))


if __name__ == "__main__":
    main()
