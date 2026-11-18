# Morfologia Matemática Aplicada – PDSI (Projeto 6)

Uso de **erosão, dilatação, abertura e fechamento** para corrigir falhas em máscaras
binárias de folhas e destacar lesões de doenças em imagens do **PlantVillage**.

## Pipeline

```
imagem colorida
  → distância de cor (a*, b*) até o fundo → Otsu            = máscara bruta da folha
  → abertura → fechamento → preenchimento → maior componente = máscara da folha
  → Otsu no canal a* (tecido verde)
  → lesões = folha AND NOT verde → abertura 3×3 → componentes ≥ 15 px
```

## Resultados (3.221 imagens, 3 classes)

| Classe | Imagens | IoU bruta | IoU limpa | Componentes (bruta → limpa) | Nº lesões | % área lesionada |
|---|---|---|---|---|---|---|
| Tomate saudável | 1.591 | 0.958 | 0.960 | 3.1 → 1.0 | 1.0 | 0.32 |
| Tomate – pinta-preta | 1.000 | 0.876 | 0.893 | 13.9 → 1.0 | 8.3 | 8.00 |
| Maçã – sarna | 630 | 0.947 | 0.955 | 3.1 → 1.0 | 6.7 | 1.82 |

Elemento estruturante: na varredura (3 formas × 9 tamanhos) o melhor é elipse 3×3 (IoU 0,9388); o pipeline usa
elipse 5×5 (0,9382), praticamente igual. Com 1×1 (sem abertura/fechamento) o IoU já é 0,9385: o ganho de IoU vem do
preenchimento de buracos e do maior componente; acima de ≈ 7 px o IoU cai. Tempo: ≈ 4 ms por imagem;
`gerar_resultados.py` completo (métricas + varredura de 27 kernels) leva ≈ 2–3 min.

IoU calculado contra a versão `segmented` do PlantVillage (gabarito).


## Reconhecimento de formas

Cada região candidata recebe descritores (área, circularidade, solidez, alongamento, espessura, fração que
sobrevive a uma abertura 5×5). Regiões finas e alongadas → **nervura**; demais → **mancha**. Por folha, somam-se a
quantidade, o tamanho e a forma das manchas, a **granulometria** (espectro de padrões) e a forma da folha.
Árvore de decisão (profundidade ≤ 4), validação cruzada 5-fold, 3 classes:

| Atributos | Acurácia |
|---|---|
| Forma das lesões | 76,6% |
| Forma da folha | 66,6% |
| **Lesões + folha** | **86,0%** |

Só tomate (saudável × pinta-preta), só com a forma das lesões: 93,9%.
Regra única saudável × doente (% de manchas ≥ 0,41%): 86,6%.
Granulometria: fração das manchas que sobrevive à abertura de raio 2 — pinta-preta 0,88, sarna 0,69, saudável 0,47.

## Estrutura

```
data/
  color/<classe>/NNNN.jpg     imagens originais
  segmented/<classe>/NNNN.jpg versão segmentada (gabarito)
  origem.csv                  nome original de cada arquivo no PlantVillage
src/morfologia.py             funções (binarização, operadores, limpeza, lesões, métricas)
src/formas.py                 descritores de forma, regra mancha x nervura, granulometria
scripts/reconhecer_formas.py  extrai atributos de forma e classifica as folhas (árvore de decisão)
scripts/baixar_dataset.py     baixa o dataset (git sparse-checkout do PlantVillage)
scripts/gerar_resultados.py   métricas da segmentação e figuras 1–5
scripts/gerar_relatorio.py    gera resultados/relatorio.pdf (relatório completo)
notebooks/01_morfologia.ipynb passo a passo com visualizações
resultados/                   gerado ao rodar: relatorio.pdf, fig1–fig8, CSVs (fora do Git)
```

## Como rodar

Com Python 3.10–3.12 e Git (o `requirements.txt` fixa o scikit-learn 1.7.2, versão com que os números acima foram gerados):

```bash
git clone https://github.com/Sam-Cassiano/Sam-Cassiano-morfologia-matematica.git
cd Sam-Cassiano-morfologia-matematica
python -m venv .venv
.venv\Scripts\activate                # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
jupyter notebook notebooks/01_morfologia.ipynb
```

No notebook, rode tudo (**Run All**). A seção 0 baixa o dataset e roda os scripts na primeira vez
(≈ 4 min no total); depois disso, reaproveita o que já existe. A última seção gera `resultados/relatorio.pdf`.

Os scripts também podem ser rodados direto no terminal, na raiz do projeto:

```bash
python scripts/baixar_dataset.py       # baixa as 3 classes do PlantVillage (≈ 100 MB, ≈ 1 min)
python scripts/gerar_resultados.py     # segmentação: métricas e figuras 1–5 (≈ 2 min)
python scripts/reconhecer_formas.py    # formas + classificação, figuras 6–8 (≈ 1 min)
python scripts/gerar_relatorio.py      # resultados/relatorio.pdf (≈ 5 s)
```

## Dados

Todas as imagens de três classes (`Tomato___healthy` 1.591, `Tomato___Early_blight` 1.000,
`Apple___Apple_scab` 630 — 3.221 no total, ≈ 99 MB com as versões color e segmented) do repositório
[spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
(Hughes & Salathé, 2015). As imagens **não ficam no GitHub**: `scripts/baixar_dataset.py` as baixa
e numera (`0001.jpg`, …); `data/origem.csv` guarda o nome original de cada uma.

## Próximos passos

- Reconstrução morfológica para lesões na borda da folha
- Esqueletização das nervuras
- Top-hat / black-hat para realçar as manchas escuras da sarna da maçã
