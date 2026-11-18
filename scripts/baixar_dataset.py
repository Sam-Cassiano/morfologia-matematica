"""Baixa as 3 classes do PlantVillage usadas no projeto (versões color e segmented).

Fonte: https://github.com/spMohanty/PlantVillage-Dataset (pasta raw/).
Método padrão: git sparse-checkout (baixa só as 6 pastas necessárias, ≈ 100 MB).
Se o git falhar, usa download arquivo por arquivo pela API do GitHub.

Uso (na raiz do repositório):
    python scripts/baixar_dataset.py              # todas as imagens (3.221 pares)
    python scripts/baixar_dataset.py --max 50     # só 50 por classe (teste rápido)
    python scripts/baixar_dataset.py --metodo http

Saída: data/color/<classe>/NNNN.jpg, data/segmented/<classe>/NNNN.jpg, data/origem.csv
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(RAIZ, "data")
REPO = "https://github.com/spMohanty/PlantVillage-Dataset.git"
API = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset"
RAW = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/"
CLASSES = {"Tomato___healthy": "tomate_saudavel",
           "Tomato___Early_blight": "tomate_pinta_preta",
           "Apple___Apple_scab": "maca_sarna"}


def nome_segmentado(nome_color):
    return os.path.splitext(nome_color)[0] + "_final_masked.jpg"


def pares(lista_color, conjunto_seg, maximo):
    """Pares (color, segmented) em ordem alfabética, numerados 0001, 0002…"""
    saida = []
    for nome in sorted(lista_color):
        seg = nome_segmentado(nome)
        if seg in conjunto_seg:
            saida.append((nome, seg))
            if maximo and len(saida) >= maximo:
                break
    return saida


def via_git(maximo):
    tmp = tempfile.mkdtemp(prefix="plantvillage_")
    try:
        print("clonando (sparse) o PlantVillage...", flush=True)
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, tmp],
                       check=True, stdout=subprocess.DEVNULL)
        pastas = [f"raw/{v}/{c}" for c in CLASSES for v in ("color", "segmented")]
        print("baixando as pastas:", ", ".join(pastas), flush=True)
        subprocess.run(["git", "-C", tmp, "sparse-checkout", "set", "--no-cone", *pastas], check=True)
        linhas = []
        for c, alias in CLASSES.items():
            dc, ds = os.path.join(tmp, "raw", "color", c), os.path.join(tmp, "raw", "segmented", c)
            lista = pares(os.listdir(dc), set(os.listdir(ds)), maximo)
            for k, (nc, ns) in enumerate(lista, 1):
                fn = f"{k:04d}.jpg"
                for origem, v in ((os.path.join(dc, nc), "color"), (os.path.join(ds, ns), "segmented")):
                    destino = os.path.join(DATA, v, alias, fn)
                    os.makedirs(os.path.dirname(destino), exist_ok=True)
                    shutil.copyfile(origem, destino)
                linhas.append([alias, fn, nc])
            print(f"  {alias}: {len(lista)} pares", flush=True)
        return linhas
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _json(url):
    return json.loads(urllib.request.urlopen(url, timeout=60).read())


def via_http(maximo):
    arv = {e["path"]: e["sha"] for e in _json(f"{API}/contents/raw")}
    top = {v: {e["path"]: e["sha"] for e in _json(f"{API}/git/trees/{arv[v]}")["tree"]} for v in ("color", "segmented")}
    tarefas, linhas = [], []
    for c, alias in CLASSES.items():
        col = [e["path"] for e in _json(f"{API}/git/trees/{top['color'][c]}")["tree"] if e["type"] == "blob"]
        seg = {e["path"] for e in _json(f"{API}/git/trees/{top['segmented'][c]}")["tree"] if e["type"] == "blob"}
        lista = pares(col, seg, maximo)
        for k, (nc, ns) in enumerate(lista, 1):
            fn = f"{k:04d}.jpg"
            tarefas.append((RAW + f"color/{c}/" + urllib.parse.quote(nc), os.path.join(DATA, "color", alias, fn)))
            tarefas.append((RAW + f"segmented/{c}/" + urllib.parse.quote(ns), os.path.join(DATA, "segmented", alias, fn)))
            linhas.append([alias, fn, nc])
        print(f"  {alias}: {len(lista)} pares", flush=True)

    def um(t):
        url, dst = t
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return True
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        for _ in range(4):
            try:
                dados = urllib.request.urlopen(url, timeout=60).read()
                open(dst, "wb").write(dados)
                return True
            except Exception:
                pass
        return False

    print(f"baixando {len(tarefas)} arquivos...", flush=True)
    with ThreadPoolExecutor(24) as ex:
        ok = list(ex.map(um, tarefas))
    if not all(ok):
        print(f"AVISO: {ok.count(False)} arquivos falharam; rode de novo para completar.")
    return linhas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max", type=int, default=0, help="máximo de imagens por classe (0 = todas)")
    ap.add_argument("--metodo", choices=["git", "http"], default="git")
    a = ap.parse_args()
    if a.metodo == "git" and shutil.which("git"):
        try:
            linhas = via_git(a.max)
        except Exception as e:  # noqa: BLE001
            print("git falhou (", e, "), usando http...", flush=True)
            linhas = via_http(a.max)
    else:
        linhas = via_http(a.max)
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "origem.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["classe", "arquivo", "nome_original_plantvillage"])
        w.writerows(linhas)
    print(f"pronto: {len(linhas)} pares em {DATA}")


if __name__ == "__main__":
    sys.exit(main())
