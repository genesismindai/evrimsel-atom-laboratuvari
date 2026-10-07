#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 genesismindai (Evrimsel Atom Laboratuvarı)
"""
lineage.py — SIRALI İLERLEME (rekor zinciri).

Tasarım kuralı (kullanıcı talebi): "rastgele tohum" yoktur. Her koşu, önceki
koşunun bıraktığı REKORUN üstüne koyar; ilerleme sıralı ve monotonik'tir:

  * Her benchmark için tek bir "en iyi kayıt" (rekor) tutulur.
  * Bir sonraki koşu, rekorun ağacını (ve şampiyonlar listesini) başlangıç
    bireyi olarak devralır — arama sıfırdan başlamaz.
  * Yeni sonuç rekoru geçemezse rekor KORUNUR (asla geriye gitme yok).
  * Karşılaştırma ölçütü dürüstlük kurallarına göre sıralıdır:
        (1) kuantum jürisinden geçen (DOĞRULANMIŞ),
        (2) en az jüri ihlali,
        (3) en iyi uyum (max(train, val) — aşırı uyuma karşı),
        (4) en ucuz (maliyet birimi).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.environ.get("EA_LINEAGE") or os.path.join(ROOT, "data", "lineage.json")


# ------------------------------------------------------------------ yardımcılar
def _to_tree(obj):
    """
    Kayıt biçimini (JSON listesi ya da repr metni) ağaç demetine çevirir.
    Düğüm biçimi: ('c', sabit) | ('x', değişken_idx) | ('f', op, *argümanlar)
    """
    if isinstance(obj, str):
        import ast as _ast
        try:
            obj = _ast.literal_eval(obj)
        except Exception:
            return None
    if isinstance(obj, (tuple, list)):
        if len(obj) == 2 and obj[0] in ("c", "x"):
            return (str(obj[0]), obj[1])
        if len(obj) >= 2 and obj[0] == "f":
            args = tuple(_to_tree(a) for a in obj[2:])
            if any(a is None for a in args):
                return None
            return ("f", str(obj[1])) + args
        return tuple(_to_tree(a) for a in obj) if obj else None
    return obj


def _ranking(rec: dict, floor: float = 0.0) -> tuple:
    """
    Rekor karşılaştırma anahtarı (küçük olan daha iyi). Dürüst-MDL:
    hata, ÖLÇÜM GÜRÜLTÜ TABANI ile kapılanır — tabanın altındaki fark bilgi
    taşımaz, bu yüzden eşit bilgi taşıyan daha UCUZ form rekor olur.
    """
    loss = max(float(rec.get("loss_val") or 1e9), float(rec.get("loss_train") or 0.0),
               float(floor or 0.0))
    return (
        0 if rec.get("verified") else 1,
        float(rec.get("violation") or 0.0),
        loss,
        float(rec.get("cost") or 1e9),
    )


def better(cand: dict, cur: Optional[dict], floor: float = 0.0) -> bool:
    if cur is None:
        return True
    return _ranking(cand, floor) < _ranking(cur, floor)


def load(path: Optional[str] = None) -> dict:
    path = path or PATH
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"iteration": 0, "benchmarks": {}, "steps": []}


def save(state: dict, path: Optional[str] = None) -> None:
    path = path or PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)


def next_iteration(state: dict) -> int:
    return int(state.get("iteration", 0)) + 1


def bench_entry(state: dict, key: str) -> dict:
    return (state.get("benchmarks") or {}).get(key) or {}


def resume_trees(state: dict, key: str, limit: int = 3) -> List[tuple]:
    """Bir sonraki koşunun devralacağı ağaçlar (rekor + eski rekorlar)."""
    ent = bench_entry(state, key)
    out: List[tuple] = []
    rec = ent.get("record") or {}
    cands = []
    if rec.get("tree"):
        cands.append(rec["tree"])
    cands += [st.get("tree") for st in (ent.get("steps") or [])[-8:] if st.get("tree")]
    for sc in cands:
        if len(out) >= limit:
            break
        t = _to_tree(sc)
        if not t:
            continue
        if repr(t) not in {repr(x) for x in out}:
            out.append(t)
    return out[:limit]


def merge(state: dict, key: str, iteration: int, run_id: str,
          champ: Optional[dict], extra: Optional[dict] = None,
          noise_rel: float = 0.0) -> dict:
    """
    Koşu sonucunu rekor zincirine işler. Rekor YALNIZCA iyileşirse güncellenir.
    Dönüş: {"new_record": bool, "previous": {...}|None, "iteration": int}
    """
    bms = state.setdefault("benchmarks", {})
    ent = bms.setdefault(key, {"record": None, "steps": []})
    prev = ent.get("record")
    info = {"new_record": False, "previous": prev, "iteration": iteration}
    if not champ or not champ.get("python"):
        return info

    rec = {
        "iteration": iteration, "run_id": run_id,
        "formula": champ.get("formula"), "python": champ.get("python"),
        "tree": champ.get("tree"),
        "cost": champ.get("cost"), "loss_train": champ.get("loss_train"),
        "loss_val": champ.get("loss_val"), "loss_test": champ.get("loss_test"),
        "verified": champ.get("verified"), "violation": champ.get("violation"),
        "at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    ent["steps"] = (ent.get("steps") or [])[-60:]
    ent["steps"].append({k: rec[k] for k in ("iteration", "run_id", "formula", "cost",
                                             "loss_val", "loss_test", "verified",
                                             "violation", "tree", "at")})
    if better(rec, prev, noise_rel):
        ent["record"] = rec
        info["new_record"] = True
    if extra:
        ent.setdefault("notes", {}).update(extra)
    state["iteration"] = max(int(state.get("iteration", 0)), iteration)
    state.setdefault("steps", []).append({
        "iteration": iteration, "run_id": run_id, "at": rec["at"],
        "bench": key, "new_record": info["new_record"],
        "cost": rec["cost"], "loss_val": rec["loss_val"], "verified": rec["verified"],
    })
    state["steps"] = state["steps"][-400:]
    return info


def summary(state: dict) -> List[dict]:
    """Site/rapor için: her benchmark'ın güncel rekoru."""
    out = []
    for key, ent in sorted((state.get("benchmarks") or {}).items()):
        rec = ent.get("record") or {}
        if not rec:
            continue
        steps = ent.get("steps") or []
        n_rec = sum(1 for a, b in zip(steps, steps[1:])
                    if _ranking(b) < _ranking(a))
        out.append({
            "key": key, "iteration": rec.get("iteration"), "run_id": rec.get("run_id"),
            "formula": rec.get("formula"), "cost": rec.get("cost"),
            "loss_val": rec.get("loss_val"), "loss_test": rec.get("loss_test"),
            "verified": rec.get("verified"), "violation": rec.get("violation"),
            "records_total": n_rec + 1, "steps_total": len(steps),
        })
    return out
