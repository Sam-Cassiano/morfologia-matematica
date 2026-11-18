"""Reconhecimento de formas com morfologia matemática.

1. Descritores de cada região (lesão candidata): área, circularidade, solidez,
   alongamento, espessura e fração que sobrevive a uma abertura 5×5.
2. Regra de forma: região fina e alongada = NERVURA; o resto = MANCHA.
3. Atributos da folha inteira: quantidade/tamanho/forma das manchas,
   granulometria (espectro de padrões) e forma da folha.
"""
import cv2
import numpy as np

from morfologia import kernel

RAIOS_GRANULOMETRIA = (1, 2, 3, 5, 8)


# --------------------------------------------------------------- por região
def descritores_regiao(mask_regiao):
    """Descritores de forma de UMA região binária (uint8 0/255)."""
    area = int((mask_regiao > 0).sum())
    cnts, _ = cv2.findContours(mask_regiao, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    perim = cv2.arcLength(c, True)
    casca = np.zeros_like(mask_regiao)
    cv2.fillPoly(casca, [cv2.convexHull(np.vstack(cnts))], 255)
    hull = int((casca > 0).sum())
    ys, xs = np.nonzero(mask_regiao)
    if area >= 3:
        l1, l2 = sorted(np.linalg.eigvalsh(np.cov(np.vstack([xs, ys]))), reverse=True)
        alongamento = float(np.sqrt(l1 / max(l2, 1e-6)))
    else:
        alongamento = 1.0
    dist = cv2.distanceTransform(mask_regiao, cv2.DIST_L2, 3)
    aberta = cv2.morphologyEx(mask_regiao, cv2.MORPH_OPEN, kernel(5, "elipse"))
    return {
        "area": area,
        "circularidade": float(min(1.0, 4 * np.pi * area / perim ** 2)) if perim > 0 else 0.0,
        "solidez": float(min(1.0, area / hull)) if hull > 0 else 1.0,
        "alongamento": alongamento,
        "espessura": float(2 * dist.max()),
        "sobrevive_abertura": float((aberta > 0).sum() / area) if area else 0.0,
    }


def eh_nervura(d, along_min=3.0, esp_max=7.0, sobrevive_max=0.35):
    """Regra de forma: nervura é FINA (some na abertura 5×5) e ALONGADA."""
    fina = d["espessura"] <= esp_max or d["sobrevive_abertura"] <= sobrevive_max
    return bool(fina and d["alongamento"] >= along_min)


def classificar_regioes(mask_lesoes):
    """Separa as regiões em manchas e nervuras. Retorna (manchas, nervuras, lista de descritores)."""
    n, rot, _, _ = cv2.connectedComponentsWithStats(mask_lesoes, connectivity=8)
    manchas = np.zeros_like(mask_lesoes)
    nervuras = np.zeros_like(mask_lesoes)
    regioes = []
    for i in range(1, n):
        reg = np.where(rot == i, 255, 0).astype(np.uint8)
        d = descritores_regiao(reg)
        d["tipo"] = "nervura" if eh_nervura(d) else "mancha"
        (nervuras if d["tipo"] == "nervura" else manchas)[rot == i] = 255
        regioes.append(d)
    return manchas, nervuras, regioes


# ------------------------------------------------------------ granulometria
def granulometria(mask, raios=RAIOS_GRANULOMETRIA):
    """Fração da área que sobrevive a aberturas com discos de raio crescente.

    É o 'espectro de padrões' da morfologia: mede o TAMANHO das formas sem
    precisar separá-las. Manchas grandes sobrevivem a raios maiores.
    """
    total = (mask > 0).sum()
    if total == 0:
        return [0.0] * len(raios)
    return [float((cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel(2 * r + 1, "elipse")) > 0).sum() / total)
            for r in raios]


# ------------------------------------------------------------- folha inteira
def descritores_folha(mask_folha):
    """Forma da folha. Recortes = área que o fechamento 15×15 acrescenta (dentes/lóbulos)."""
    d = descritores_regiao(mask_folha)
    fechada = cv2.morphologyEx(mask_folha, cv2.MORPH_CLOSE, kernel(15, "elipse"))
    area = max(1, (mask_folha > 0).sum())
    return {
        "folha_circularidade": d["circularidade"],
        "folha_solidez": d["solidez"],
        "folha_alongamento": d["alongamento"],
        "folha_recortes": float(((fechada > 0) & (mask_folha == 0)).sum() / area),
    }


def atributos_folha(mask_folha, mask_lesoes):
    """Vetor de atributos morfológicos de uma folha. Retorna (atributos, manchas, nervuras)."""
    manchas, nervuras, regioes = classificar_regioes(mask_lesoes)
    ms = [r for r in regioes if r["tipo"] == "mancha"]
    area_folha = max(1, (mask_folha > 0).sum())
    area_m = sum(r["area"] for r in ms)
    peso = lambda k: (sum(r[k] * r["area"] for r in ms) / area_m) if area_m else 0.0  # noqa: E731
    at = {
        "n_manchas": len(ms),
        "n_nervuras": len(regioes) - len(ms),
        "pct_manchas": 100.0 * area_m / area_folha,
        "pct_nervuras": 100.0 * (nervuras > 0).sum() / area_folha,
        "area_media_mancha": area_m / len(ms) if ms else 0.0,
        "maior_mancha_pct": 100.0 * max((r["area"] for r in ms), default=0) / area_folha,
        "circularidade_manchas": peso("circularidade"),
        "solidez_manchas": peso("solidez"),
        "alongamento_manchas": peso("alongamento"),
    }
    for r, v in zip(RAIOS_GRANULOMETRIA, granulometria(manchas)):
        at[f"granulo_r{r}"] = v
    at.update(descritores_folha(mask_folha))
    return at, manchas, nervuras
