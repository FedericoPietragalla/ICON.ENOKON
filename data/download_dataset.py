"""
Scarica il dataset UCI "Wine Quality" e lo salva come  data/wine_quality.csv.

    python data/download_dataset.py                  # scarica da archive.ics.uci.edu
    python data/download_dataset.py --raw-dir DIR    # usa file già scaricati a mano
                                                     # (winequality-red.csv, winequality-white.csv)

Il CSV prodotto contiene le 11 misure chimiche, il punteggio `quality` e la colonna
`wine_type` ('red' | 'white'): rossi e bianchi sono uniti in un unico dataset.

Fonte e citazione: P. Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis (2009),
"Modeling wine preferences by data mining from physicochemical properties",
Decision Support Systems 47(4):547-553.  UCI Machine Learning Repository:
https://doi.org/10.24432/C56S3T   (pagina del dataset: https://archive.ics.uci.edu/dataset/186/wine+quality)
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg  # noqa: E402

CSV_URLS = {
    "red": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "white": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
}
ZIP_URL = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"
EXPECTED_ROWS = {"red": 1599, "white": 4898}
FILE_NAMES = {"red": "winequality-red.csv", "white": "winequality-white.csv"}


def _fetch(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ENOKON dataset downloader)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _parse(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), sep=";")
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df


def _from_dir(raw_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for kind, name in FILE_NAMES.items():
        f = raw_dir / name
        if not f.exists():
            raise FileNotFoundError(f"File mancante: {f}")
        out[kind] = _parse(f.read_bytes())
    return out


def _from_web() -> dict[str, pd.DataFrame]:
    try:                                          # 1) due CSV separati
        return {k: _parse(_fetch(u)) for k, u in CSV_URLS.items()}
    except Exception as exc:                      # noqa: BLE001
        print(f"  download dei CSV fallito ({exc}); provo con lo zip...")
    with zipfile.ZipFile(io.BytesIO(_fetch(ZIP_URL))) as zf:   # 2) archivio zip
        return {k: _parse(zf.read(n)) for k, n in FILE_NAMES.items()}


def build(out_path: Path = cfg.DATA_PATH, raw_dir: Path | None = None, force: bool = False) -> Path:
    out_path = Path(out_path)
    if out_path.exists() and not force:
        print(f"Dataset già presente: {out_path}")
        return out_path
    try:
        parts = _from_dir(Path(raw_dir)) if raw_dir else _from_web()
    except Exception as exc:                      # noqa: BLE001
        raise SystemExit(
            f"\nImpossibile ottenere il dataset: {exc}\n\n"
            "Scaricalo a mano da https://archive.ics.uci.edu/dataset/186/wine+quality\n"
            "(file winequality-red.csv e winequality-white.csv), mettili in una cartella e rilancia:\n"
            "    python data/download_dataset.py --raw-dir <cartella>\n")
    frames = []
    for kind, df in parts.items():
        missing = [c for c in cfg.RAW_FEATURES + [cfg.QUALITY_COLUMN] if c not in df.columns]
        if missing:
            raise SystemExit(f"Colonne mancanti nel file '{kind}': {missing}")
        if len(df) != EXPECTED_ROWS[kind]:
            print(f"  attenzione: {kind} ha {len(df)} righe (attese {EXPECTED_ROWS[kind]})")
        df = df[cfg.RAW_FEATURES + [cfg.QUALITY_COLUMN]].copy()
        df[cfg.TYPE_COLUMN] = kind
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(out_path, index=False)
    print(f"Salvato {out_path}  ({len(full)} vini: "
          f"{(full[cfg.TYPE_COLUMN] == 'red').sum()} rossi, {(full[cfg.TYPE_COLUMN] == 'white').sum()} bianchi)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=None, help="cartella con i due CSV originali UCI")
    ap.add_argument("--force", action="store_true", help="riscrive il CSV anche se esiste")
    ap.add_argument("--out", type=Path, default=cfg.DATA_PATH)
    a = ap.parse_args()
    build(a.out, a.raw_dir, a.force)
