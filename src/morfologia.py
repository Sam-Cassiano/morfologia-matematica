"""Funções do projeto "Morfologia Matemática Aplicada" (PDSI).

Pipeline:
    imagem colorida -> distância de cor ao fundo -> Otsu -> máscara bruta da folha
    -> limpeza morfológica (abertura, fechamento, preenchimento, maior componente)
    -> tecido verde (Otsu no canal a*) -> lesões = folha AND NOT verde -> abertura
"""
import cv2
import numpy as np

FORMAS = {
    "retangulo": cv2.MORPH_RECT,
    "elipse": cv2.MORPH_ELLIPSE,
    "cruz": cv2.MORPH_CROSS,
}


def kernel(tamanho=5, forma="elipse"):
    """Elemento estruturante quadrado de lado `tamanho`."""
    return cv2.getStructuringElement(FORMAS[forma], (tamanho, tamanho))


# ----------------------------------------------------------------- binarização
def mapa_distancia_fundo(img_bgr, borda=5):
    """Distância de cromaticidade (a*, b*) de cada pixel à cor do fundo.

    A cor do fundo é estimada pela mediana dos pixels da borda da imagem.
    Usar só a*/b* (sem L*) faz a sombra, que tem a mesma cor do fundo porém
    mais escura, ficar com distância baixa. Retorna uint8 normalizado 0-255.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = borda
    pix_borda = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                                lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    ref = np.median(pix_borda, axis=0)
    d = np.sqrt(((lab[..., 1:] - ref[1:]) ** 2).sum(axis=-1))
    return cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def binarizar_folha(img_bgr):
    """Folha inteira (tecido verde + lesões) vs. fundo: Otsu na distância ao fundo."""
    _, mask = cv2.threshold(mapa_distancia_fundo(img_bgr), 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def binarizar_verde(img_bgr):
    """Apenas o tecido verde: Otsu no canal a* do Lab (verde = a* baixo)."""
    a = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[..., 1]
    _, mask = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask


def mascara_gabarito(seg_bgr, limiar=20):
    """Máscara de referência a partir da versão 'segmented' do PlantVillage."""
    gray = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2GRAY)
    return np.where(gray > limiar, 255, 0).astype(np.uint8)


# ------------------------------------------------------- operadores básicos
def operadores_basicos(mask, k):
    """Aplica os quatro operadores morfológicos básicos."""
    return {
        "erosao": cv2.erode(mask, k),
        "dilatacao": cv2.dilate(mask, k),
        "abertura": cv2.morphologyEx(mask, cv2.MORPH_OPEN, k),
        "fechamento": cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k),
    }


# ------------------------------------------------------------------ limpeza
def preencher_buracos(mask):
    """Preenche buracos internos (flood fill a partir da borda da imagem)."""
    h, w = mask.shape
    pad = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = pad.copy()
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
    buracos = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, buracos)


def maior_componente(mask):
    """Mantém só o maior componente conexo (a folha)."""
    n, rot, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    maior = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(rot == maior, 255, 0).astype(np.uint8)


def limpar_mascara(mask, k):
    """Abertura -> fechamento -> preenchimento de buracos -> maior componente.

    Retorna um dicionário com cada etapa para facilitar a visualização.
    """
    etapas = {"bruta": mask}
    etapas["abertura"] = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    etapas["fechamento"] = cv2.morphologyEx(etapas["abertura"], cv2.MORPH_CLOSE, k)
    etapas["preenchida"] = preencher_buracos(etapas["fechamento"])
    etapas["final"] = maior_componente(etapas["preenchida"])
    return etapas


# ------------------------------------------------------------------ lesões
def segmentar_lesoes(mask_verde, mask_folha, k, margem=7, area_min=15):
    """Lesões = pixels da folha que NÃO são verdes.

    - `mask_folha` é a folha completa (já preenchida);
    - erodimos a folha com uma elipse `margem`×`margem` (7×7 tira ≈ 3 px da borda)
      para ignorar o contorno, onde a sombra criaria falsas lesões;
    - abertura com `k` remove pontos isolados;
    - componentes menores que `area_min` px são descartados.
    """
    interior = cv2.erode(mask_folha, kernel(margem, "elipse"))
    bruta = cv2.bitwise_and(interior, cv2.bitwise_not(mask_verde))
    limpa = cv2.morphologyEx(bruta, cv2.MORPH_OPEN, k)
    n, rot, stats, _ = cv2.connectedComponentsWithStats(limpa, connectivity=8)
    manter = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= area_min]
    final = np.isin(rot, manter).astype(np.uint8) * 255
    return {"bruta": bruta, "limpa": final, "n_lesoes": len(manter)}


# ------------------------------------------------------------------ métricas
def iou(a, b):
    a, b = a > 0, b > 0
    uniao = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / uniao) if uniao else 1.0


def n_componentes(mask):
    return cv2.connectedComponents(mask, connectivity=8)[0] - 1


def pct_area(mask_parte, mask_todo):
    todo = (mask_todo > 0).sum()
    return 100.0 * (mask_parte > 0).sum() / todo if todo else 0.0
