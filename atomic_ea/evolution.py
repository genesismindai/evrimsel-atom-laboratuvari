# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 genesismindai (Evrimsel Atom Laboratuvarı)
#
# Bu program özgür yazılımdır: GNU Genel Kamu Lisansı (GPL) sürüm 3 veya
# sonraki sürümleri koşulları altında yeniden dağıtabilir ve/veya
# değiştirebilirsiniz. Ayrıntılar için LICENSE dosyasına bakın.
"""
evolution.py — Filtre entegreli, ada-modeli evrimsel sembolik regresyon motoru.

Tasarım kararları
-----------------
* Arama uzayı: yalnızca whitelist operatörler (güvenli + ucuz katman).
* Uygunluk:  log10(eğitim hatası) + maliyet cezası
  -> "aynı hatayı daha ucuz veren formül kazanır" ilkesi matematikselleşir.
* Kaskad değerlendirme: ucuz filtreler popülasyonun tamamına,
  pahalı fizik filtreleri (Hellmann-Feynman, virial, gürbüzlük) yalnızca
  seçkinlere/arşive uygulanır. Bu, kare-hata maliyetini ~50x düşürür.
* Sayısal sabit ayarı: her bireyde Gauss-Newton (Levenberg benzeri) ile
  sabitler yeniden ayarlanır -> "n^a" tipi yasalar, tek bir tamsayı üs
  mutasyonuna bağlı kalmadan keşfedilebilir.
* Ada modeli + göç: 3 ada, 12 nesilde bir en iyiler göç eder (çeşitlilik).
* Arşiv: Pareto cephesi (hata, maliyet) — en ucuz-doğru formüller saklanır.
* Dürüstlük: hiçbir referans veri kapalı formdan gelmez; sınav kümesi
  yalnızca rapor aşamasında bir kez kullanılır.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
import copy
import itertools
import math
import time
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from . import expression as ex
from . import filters as fl


# ------------------------------------------------------------------ yapılandırma
@dataclass
class EAConfig:
    pop_size: int = 300
    generations: int = 90
    islands: int = 3
    migration_every: int = 12
    tournament: int = 5
    crossover_rate: float = 0.75
    elite_frac: float = 0.06
    max_depth: int = 5
    max_nodes: int = 22
    init_max_depth: int = 3
    cost_weight: float = 0.06
    lm_steps: int = 5
    simplify: bool = True
    seed: int = 0
    time_budget_us: float = 400.0
    null_factor: float = 5.0
    strict_interval: int = 4
    patience: int = 45
    physics_seeded: bool = True
    blind_mode: bool = False
    quantum_gate: bool = True          # aramada kuantum jürisi uygulanır mı
    quantum_kill: float = 30.0         # ihlal oranı bu eşiği aşarsa ∞ (ağır ihlal)
    quantum_weight: float = 0.15       # sınırda ihlal için kademeli ceza (log10 birimi)
    jury_refit: bool = True            # ihlal eden adayın sabitleri jüriye göre onarılır
    jury_steps: int = 4                # 1. aşama onarım adımı (hızlı)
    jury_weight: float = 2.0           # 1. aşama jüri ağırlığı
    jury_steps2: int = 14              # 2. aşama (yalnızca umut vadeden adaylarda)
    jury_weight2: float = 4.0          # 2. aşama jüri ağırlığı


# ------------------------------------------------------------------ başlangıç ağaçları
def _c(v: float) -> ex.Tree:
    return ("c", float(v))


def _x(i: int) -> ex.Tree:
    return ("x", int(i))


def _f(op: str, *a: ex.Tree) -> ex.Tree:
    return ("f", op) + tuple(a)


def _rational_seeds(R: ex.Tree) -> List[ex.Tree]:
    """Rasyonel (Padé) aileler — bağ eğrileri için genel yapısal kalıplar."""
    a, b, c, d, e = (_c(1.0), _c(1.0), _c(1.0), _c(1.0), _c(1.0))
    r2 = _f("sq", R)
    return [
        # (a + bR)/(1 + cR + dR²)²      → türevi kuvvet yapısıyla aynı aile
        _f("div", _f("add", a, _f("mul", b, R)),
             _f("sq", _f("add", _c(1.0), _f("add", _f("mul", c, R), _f("mul", d, r2))))),
        # (a + bR + cR²)/(1 + dR + eR²)
        _f("div", _f("add", _f("add", a, _f("mul", b, R)), _f("mul", c, r2)),
             _f("add", _c(1.0), _f("add", _f("mul", d, R), _f("mul", e, r2)))),
        # a + b/R + c/R² + d/R³
        _f("add", _f("add", a, _f("div", b, R)),
             _f("add", _f("div", c, r2), _f("div", d, _f("mul", r2, R)))),
        # a/R⁶ − b/R⁴ + c/R² + d
        _f("add", _f("add", _f("div", a, _f("mul", r2, _f("sq", r2))),
                     _f("neg", _f("div", b, _f("sq", r2)))),
             _f("add", _f("div", _c(-1.0), r2), d)),
    ]


def physics_seeds(var_names: List[str], key: str = "") -> List[ex.Tree]:
    """
    ŞEFFAF ön bilgi arketipleri (rapora aynen yazılır).

    Bunlar "cevap" değildir; güç yasası oranı / polinom gibi genel yapısal
    kalıplardır ve yalnızca sabitleri ayarlanacak boş şablonlardır. Keşfin
    geçerliliği bu şablondan değil, bağımsız filtrelerden gelir.
    """
    n = len(var_names)
    seeds: List[ex.Tree] = []
    generic2 = n == 2 and tuple(var_names) not in (("Z", "N"), ("R", "Z"))
    if generic2:
        x0, x1 = _x(0), _x(1)
        seeds += [
            _f("mul", _c(-1.0), _f("div", _f("sq", x0), _f("sq", x1))),          # A·x0²/x1²
            _f("div", _f("sq", x0), _f("mul", _f("sq", x1), _c(2.0))),
            _f("mul", _c(-1.0), _f("div", x0, _f("sq", x1))),                     # A·x0/x1²
            _f("div", x0, _f("mul", _f("sq", x1), _c(2.0))),
            _f("sub", _c(1.0), _f("mul", _c(1.0), _f("div", x0, _f("sq", x1)))),  # 1 - A·x0/x1²
            _f("add", _f("div", _f("neg", _f("sq", x0)), _c(2.0)),
                 _f("mul", _c(1.0), x0)),                                          # -x0²/2 + A·x0
            _f("mul", _c(-0.5), _f("div", _f("sq", x0), _f("add", _f("sq", x1), _c(0.0)))),
            _f("div", _f("sq", x0), _f("mul", _f("sq", x1), _f("add", _c(1.0), _c(0.0)))),
        ]
    if n == 3:  # (r, l, Z)
        r, l, Z = _x(0), _x(1), _x(2)
        seeds += [
            _f("add", _f("div", _f("neg", Z), r),
                 _f("div", _f("mul", l, _f("add", l, _c(1.0))), _f("mul", _c(2.0), _f("sq", r)))),
            _f("add", _f("mul", _c(-1.0), _f("div", Z, r)),
                 _f("div", _f("sq", l), _f("mul", _c(2.0), _f("sq", r)))),
            _f("div", _f("neg", Z), r),     # yalnız Coulomb (l=0 alt durumu)
            _f("add", _f("div", _f("neg", Z), r), _f("div", l, _f("sq", r))),
        ]
    if n == 1:                                   # bağ / kuvvet eğrileri: rasyonel aileler
        seeds += _rational_seeds(_x(0))
    if n == 2 and tuple(var_names) == ("Z", "N"):    # çok elektronlu atom serisi
        Z, N = _x(0), _x(1)
        N2 = _f("sq", N)
        Z2 = _f("sq", Z)
        seeds += [
            # −Z² + (a + bN)·Z + c·N     (izoelektronik 1/Z açılımı kalıbı)
            _f("add", _f("neg", Z2),
                 _f("add", _f("mul", _f("add", _c(1.0), _f("mul", _c(1.0), N)), Z),
                      _f("mul", _c(1.0), N))),
            # −(N/2)Z² + (a + bN)·Z + c·N + d·N²
            _f("add", _f("neg", _f("mul", _f("div", N, _c(2.0)), Z2)),
                 _f("add", _f("mul", _f("add", _c(1.0), _f("mul", _c(1.0), N)), Z),
                      _f("add", _f("mul", _c(1.0), N), _f("mul", _c(1.0), N2)))),
            # (a + b/N)Z² + cZ + d      (perdeleme düzeltmesi)
            _f("add", _f("mul", _f("add", _c(1.0), _f("div", _c(1.0), N)), Z2),
                 _f("add", _f("mul", _c(1.0), Z), _c(1.0))),
            # a·Z² + b·ZN + c·N²
            _f("add", _f("add", _f("mul", _c(1.0), Z2), _f("mul", _c(1.0), _f("mul", Z, N))),
                 _f("mul", _c(1.0), N2)),
            # İZOELEKTRONİK (1/Z) AÇILIM KALIBI — genel yapı, katsayılar serbest:
            #   −(a + b·N)·Z² + (c + d·N)·Z + e·N + f
            # (bu kalıp zorunlu değil; yalnızca yapısal bir şablon. Sabitler veriye
            #  ayarlanır ve kuantum jürisi (F5/F10/F11) geçilmeden doğrulanmaz.)
            _f("add", _f("neg", _f("mul", _f("add", _c(1.0), _f("mul", _c(1.0), N)), Z2)),
                 _f("add", _f("mul", _f("add", _c(1.0), _f("mul", _c(1.0), N)), Z),
                      _f("add", _f("mul", _c(1.0), N), _c(1.0)))),
        ]
    if n == 2 and tuple(var_names) == ("R", "Z"):     # ölçek yasası: ε = Z²·f(ZR)
        R, Z = _x(0), _x(1)
        Z2 = _f("sq", Z)
        u = _f("mul", Z, R)                            # ölçek değişkeni u = Z·R
        u2 = _f("sq", u)
        seeds += [
            # Z²·(a + b·u)/(1 + c·u)      → u→∞ limiti sabit (fiziksel)
            _f("mul", Z2, _f("div", _f("add", _c(1.0), _f("mul", _c(1.0), u)),
                              _f("add", _c(1.0), _f("mul", _c(1.0), u)))),
            # Z²·(a + b·u + c·u²)/(1 + d·u + e·u²)
            _f("mul", Z2, _f("div", _f("add", _f("add", _c(1.0), _f("mul", _c(1.0), u)),
                                        _f("mul", _c(1.0), u2)),
                              _f("add", _f("add", _c(1.0), _f("mul", _c(1.0), u)),
                                   _f("mul", _c(1.0), u2)))),
            # −Z²/2 + a·Z/R   (hidrojenik limit + Coulomb kuyruğu)
            _f("add", _f("neg", _f("mul", _f("div", _c(1.0), _c(2.0)), Z2)),
                 _f("mul", _c(1.0), _f("div", Z, R))),
            # a·Z² + b·Z/R + c
            _f("add", _f("add", _f("mul", _c(-1.0), Z2), _f("mul", _c(1.0), _f("div", Z, R))),
                 _c(1.0)),
        ]
    if n == 2 and var_names and var_names[1] in ("α", "a", "alpha"):
        Z, A = _x(0), _x(1)
        seeds += [
            _f("add", _f("div", _f("neg", _f("sq", Z)), _c(2.0)), _f("mul", _c(1.0), _f("mul", Z, A))),
            _f("add", _f("add", _f("div", _f("neg", _f("sq", Z)), _c(2.0)), _f("mul", _c(1.0), _f("mul", Z, A))),
                 _f("mul", _c(-1.0), _f("sq", A))),
            _f("mul", _c(-0.5), _f("add", _f("sq", Z), _f("mul", _c(2.0), _f("mul", Z, A)))),
        ]
    # yinelenenleri at
    uniq, seen = [], set()
    for s in seeds:
        k = repr(s)
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    return uniq


def random_tree(nvars: int, cfg: EAConfig, rng: np.random.Generator,
                max_depth: Optional[int] = None, cheap: bool = True) -> ex.Tree:
    """Ramped half-and-half: dengeli büyüme karışımı."""
    max_depth = max_depth or cfg.init_max_depth
    unary = ex.CHEAP_UNARY if cheap else ex.UNARY_SAFE
    binary = ex.CHEAP_BINARY if cheap else ex.BINARY_SAFE
    include_const_root = rng.random() < 0.35

    def build(depth: int) -> ex.Tree:
        leaf_p = 0.35 if depth > 1 else 0.0
        if depth >= max_depth or rng.random() < leaf_p:
            if rng.random() < 0.35:
                return _c(float(np.round(rng.normal(0, 2.0), 3)))
            return _x(int(rng.integers(nvars)))
        r = rng.random()
        if r < 0.45:
            a, b = build(depth + 1), build(depth + 1)
            if b[0] == "x" and rng.random() < 0.4:
                b = copy.deepcopy(a)      # kareleme olasılığını artır
            return _f(str(rng.choice(binary)), a, b)
        if r < 0.9:
            return _f(str(rng.choice(unary)), build(depth + 1))
        return build(depth + 1)

    t = build(1)
    if include_const_root:
        t = _f("mul", _c(float(np.round(rng.normal(-0.5, 1.5), 3))), t)
    return t


# ------------------------------------------------------------------ mutasyon / çaprazlama
def _nodes(t: ex.Tree) -> List[Tuple[Tuple, ...]]:
    """Tüm alt ağaçları (yol ile) listeler. Yol: (child_index, ...)"""
    out: List[Tuple[Tuple, ...]] = []
    def rec(node: ex.Tree, path: Tuple[int, ...]):
        out.append((path, node))
        if node[0] == "f":
            for i, ch in enumerate(node[2:]):
                rec(ch, path + (i,))
    rec(t, ())
    return out


def get_subtree(t: ex.Tree, path: Tuple[int, ...]) -> ex.Tree:
    node = t
    for i in path:
        node = node[2 + i]
    return node


def set_subtree(t: ex.Tree, path: Tuple[int, ...], sub: ex.Tree) -> ex.Tree:
    if not path:
        return sub
    i = path[0]
    children = list(t[2:])
    children[i] = set_subtree(children[i], path[1:], sub)
    return ("f", t[1]) + tuple(children)


def mutate(t: ex.Tree, nvars: int, cfg: EAConfig, rng: np.random.Generator,
           cheap: bool, max_depth: int) -> ex.Tree:
    nodes = _nodes(t)
    paths = [p for p, _ in nodes]
    ops = [n[1] for _, n in nodes if n[0] == "f"]

    which = rng.random()
    if which < 0.30 and paths:          # nokta mutasyonu (operatör değişimi)
        path = paths[int(rng.integers(len(paths)))]
        sub = get_subtree(t, path)
        if sub[0] == "f" and rng.random() < 0.5:
            pool = ex.CHEAP_UNARY if sub[1] == "neg" else ex.CHEAP_BINARY
            pool = ex.CHEAP_BINARY if len(sub) > 3 else ex.CHEAP_UNARY
            # arite korunmalı
            newop = str(rng.choice(ex.CHEAP_BINARY if len(sub) > 3 else ex.CHEAP_UNARY))
            t2 = ("f", newop) + tuple(sub[2:])
            return set_subtree(t, path, t2)
        if sub[0] == "c":
            return set_subtree(t, path, _c(float(np.round(sub[1] + rng.normal(0, 1.0), 3))))
        if sub[0] == "x":
            return set_subtree(t, path, _x(int(rng.integers(nvars))))
        return t
    if which < 0.55 and paths:          # alt-ağaç değişimi
        path = paths[int(rng.integers(len(paths)))]
        return set_subtree(t, path, random_tree(nvars, cfg, rng, max_depth=min(2, max_depth), cheap=cheap))
    if which < 0.70 and ops:            # sarma (unary kök)
        path = paths[int(rng.integers(len(paths)))]
        op = str(rng.choice(ex.CHEAP_UNARY if cheap else ex.UNARY_SAFE))
        return set_subtree(t, path, _f(op, get_subtree(t, path)))
    if which < 0.85 and paths:          # budama (düğümü çocuğuyla değiştir)
        cand = [p for p in paths if get_subtree(t, p)[0] == "f"]
        if cand:
            path = cand[int(rng.integers(len(cand)))]
            sub = get_subtree(t, path)
            return set_subtree(t, path, sub[2 + int(rng.integers(len(sub) - 2))])
        return t
    # sabit pertürbasyonu
    return set_subtree(t, paths[0], _f("mul", _c(float(np.round(rng.normal(1.0, 0.4), 3))), t))


def crossover(a: ex.Tree, b: ex.Tree, nvars: int, rng: np.random.Generator) -> ex.Tree:
    pa, _ = _nodes(a)[int(rng.integers(len(_nodes(a))))]
    pb, _ = _nodes(b)[int(rng.integers(len(_nodes(b))))]
    return set_subtree(a, pa, get_subtree(b, pb))


# ------------------------------------------------------------------ basitleştirme
def simplify(t: ex.Tree, probe: List[np.ndarray]) -> ex.Tree:
    """
    Cebirsel sadeleştirme + sabit katlama + sıfır/birim eleme.
    Ucuzluk hedefinin doğrudan uygulayıcısı: "aynı fonksiyonu daha az işlemle".
    """
    finite_probe = None

    def probe_ok(node: ex.Tree) -> bool:
        y = ex.evaluate(node, probe)
        return bool(np.all(np.isfinite(y)))

    def val_ok(node: ex.Tree) -> bool:
        y = ex.evaluate(node, probe)
        return bool(np.all(np.isfinite(y)))

    def rec(node: ex.Tree) -> ex.Tree:
        if node[0] in ("x", "c"):
            return node
        op = node[1]
        kids = [rec(k) for k in node[2:]]

        # sabit katlama
        if kids and all(k[0] == "c" for k in kids):
            try:
                tmp = ("f", op) + tuple(kids)
                y = ex.evaluate(tmp, [np.array([1.0])])
                if np.all(np.isfinite(y)):
                    v = float(np.ravel(y)[0])
                    if abs(v) <= 1e4:
                        return _c(0.0 if abs(v) < 1e-12 else v)
            except Exception:
                pass

        if len(kids) == 1:
            a = kids[0]
            if op == "neg" and a[0] == "f" and a[1] == "neg":
                return a[2]
            if op in ("sq", "pow2") and a[0] == "f" and a[1] in ("sq", "pow2"):
                return _f("pow4", a[2]) if False else _f("sq", _f("sq", a[2]))
            if op == "abs" and a[0] == "f" and a[1] == "abs":
                return a
            if op == "sqrt" and a[0] == "f" and a[1] in ("sq", "pow2"):
                return _f("abs", a[2])
            if op == "log" and a[0] == "f" and a[1] == "exp":
                return a[2]
            return _f(op, a)

        a, b = kids[0], kids[1]
        if op == "add":
            if b[0] == "c" and b[1] == 0.0:
                return a
            if a[0] == "c" and a[1] == 0.0:
                return b
        if op == "sub":
            if b[0] == "c" and b[1] == 0.0:
                return a
            if repr(a) == repr(b) and probe_ok(a):
                return _c(0.0)
        if op == "mul":
            if (b[0] == "c" and b[1] == 0.0) or (a[0] == "c" and a[1] == 0.0):
                if probe_ok(a) and probe_ok(b):
                    return _c(0.0)
            if b[0] == "c" and b[1] == 1.0:
                return a
            if a[0] == "c" and a[1] == 1.0:
                return b
            if a[0] == "c" and b[0] == "f" and b[1] == "mul":
                if b[2][0] == "c":
                    return _f("mul", _c(a[1] * b[2][1]), b[3])      # c1·(c2·x) -> (c1c2)·x
                if b[3][0] == "c":
                    return _f("mul", _c(a[1] * b[3][1]), b[2])      # c1·(x·c2) -> (c1c2)·x
            if a[0] == "c" and b[0] == "f" and b[1] == "div" and b[3][0] == "c" and b[3][1] != 0:
                return _f("mul", _c(a[1] / b[3][1]), b[2])          # c1·(x/c2) -> (c1/c2)·x
            if a[0] == "c" and b[0] == "c":
                return _c(a[1] * b[1])
            if a[0] == "f" and a[1] == "mul" and a[3][0] == "c" and b[0] == "c":
                return _f("mul", _c(a[3][1] * b[1]), a[2])          # (x·c1)·c2
            if b[0] == "f" and b[1] == "neg":
                if a[0] == "f" and a[1] == "neg":
                    return _f("mul", a[2], b[2])                     # (−a)(−b) -> ab
                return _f("neg", _f("mul", a, b[2]))
            if a[0] == "f" and a[1] == "neg":
                return _f("neg", _f("mul", a[2], b))
            if repr(a) == repr(b) and probe_ok(a):
                return _f("sq", a)
            if a[0] == "c" and b[0] == "c":
                return _c(a[1] * b[1])
        if op == "div":
            if a[0] == "f" and a[1] == "mul" and a[3][0] == "c" and b[0] == "c" and b[1] != 0:
                return _f("mul", _c(a[3][1] / b[1]), a[2])          # (x·c1)/c2 -> (c1/c2)·x
            if a[0] == "f" and a[1] == "mul" and a[2][0] == "c" and b[0] == "c" and b[1] != 0:
                return _f("mul", _c(a[2][1] / b[1]), a[3])          # (c1·x)/c2
            if a[0] == "c" and b[0] == "f" and b[1] == "mul" and b[2][0] == "c" and b[2][1] != 0:
                return _f("div", _c(a[1] / b[2][1]), b[3])          # c1/(c2·x) -> (c1/c2)/x
            if a[0] == "c" and b[0] == "f" and b[1] == "div" and b[2][0] == "c" and b[2][1] != 0:
                return _f("mul", _c(a[1] / b[2][1]), b[3])          # c1/(c2/x) -> (c1/c2)·x
            if a[0] == "f" and a[1] == "neg":
                return _f("neg", _f("div", a[2], b))
            if b[0] == "f" and b[1] == "neg":
                return _f("neg", _f("div", a, b[2]))
            if b[0] == "c" and b[1] == 1.0:
                return a
            if repr(a) == repr(b) and probe_ok(a):
                y = ex.evaluate(a, probe)
                if np.all(np.abs(y) > 1e-9):
                    return _c(1.0)
            if b[0] == "c" and b[1] != 0.0:
                return _f("mul", _c(1.0 / b[1]), a) if abs(1.0 / b[1]) <= 1e4 else _f("div", a, b)
        return _f(op, a, b)

    out = rec(t)
    # kareleme yerine sq kullanımı zaten var; maliyet yeniden hesabı
    return out


# ------------------------------------------------------------------ MDL sabit yuvarlama
# Basitlik sırasına göre aday sabitler (0 en basit, sonra ±1, ±1/2, ±2, …).
# Küme GENELdir (cevaba özel değil); kabul ölçütü verinin hassasiyet tabanıdır.
SNAP_CANDIDATES = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.25, -0.25, 3.0, -3.0,
                   1.5, -1.5, 4.0, -4.0, 1.0 / 3.0, -1.0 / 3.0, 1.0 / 6.0, -1.0 / 6.0,
                   2.0 / 3.0, -2.0 / 3.0, 6.0, -6.0, 8.0, -8.0, 12.0, -12.0]


def snap_constants(t: ex.Tree, X, y, noise_rel: float, passes: int = 12) -> ex.Tree:
    """
    Ölçüm-belirsizliği temelli sabit yuvarlama (Minimum Description Length).

    Kural: bir sabiti daha basit bir değere yuvarlamak, eğitim hatasını
    verinin hassasiyet tabanının (noise_rel) belirgin biçimde üstüne
    çıkarmıyorsa, sadeleştirmeyi kabul et. Böylece sayısal referansın
    çözünürlüğünün altında kalan "uydurma basamaklar" (numeroloji) elenir;
    formül, verinin gerçekten taşıdığı bilgiye indirgenir.
    """
    def loss(tt):
        return fl.rel_rmse(ex.evaluate(tt, X), y)

    cur = t
    cur_loss = loss(cur)
    simple_rank = {v: i for i, v in enumerate(SNAP_CANDIDATES)}

    for _ in range(passes):
        vals = ex.consts(cur)
        if not vals:
            break
        thr = max(noise_rel, cur_loss * 1.6)

        # --- (1) ORTAK (koordineli) yuvarlama: tüm sabitler birden
        # Tek tek yuvarlama, iki sabitin çarpımından oluşan katsayıyı
        # (ör. 0.995/(−1.99) ≈ −0.5) hedefine ulaştıramaz: ara adım vadide
        # kalır. MDL ölçütü: "veriye uygunluk kısıtı altında en kısa tanım".
        if 2 <= len(vals) <= 5:
            order = []
            for cv in vals:
                ranked = sorted(range(len(SNAP_CANDIDATES)),
                                key=lambda k: (abs(SNAP_CANDIDATES[k] - cv), k))
                order.append([SNAP_CANDIDATES[k] for k in ranked[:4]])
            combos = sorted(itertools.product(*order),
                            key=lambda cb: sum(simple_rank.get(v, 99) for v in cb))
            best = None
            for combo in combos[:160]:      # en basit atamalar önce denenir
                if sum(simple_rank.get(v, 99) for v in combo) >= 40:
                    break
                l = loss(ex.set_consts(cur, combo))
                if np.isfinite(l) and l <= thr:
                    best = combo
                    break
            if best is not None:
                cur = ex.set_consts(cur, best)
                cur_loss = loss(cur)
                improved_any = True
                continue

        # --- (2) tek sabit yuvarlama (yedek yol)
        improved = False
        for i in np.argsort(-np.abs(vals)):
            for v in SNAP_CANDIDATES:
                vals_i = ex.consts(cur)
                if abs(v - vals_i[int(i)]) < 1e-12:
                    continue
                trial = ex.set_consts(cur, [v if j == int(i) else vals_i[j]
                                            for j in range(len(vals_i))])
                l = loss(trial)
                if l <= max(noise_rel, cur_loss * 1.6):
                    cur, cur_loss = trial, l
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return cur


# ------------------------------------------------------------------ sabit ayarı (Gauss-Newton)
def tune_constants(t: ex.Tree, X: List[np.ndarray], y: np.ndarray,
                   steps: int = 5, lam: float = 1e-3) -> ex.Tree:
    """Sabitleri eğitim verisine göre Gauss-Newton ile ayarlar (LM benzeri)."""
    cs = ex.consts(t)
    if not cs:
        return t
    c = np.array(cs, dtype=float)
    if c.size > 6:
        return t
    best = copy.deepcopy(t)
    best_sse = float(np.sum((ex.evaluate(t, X) - y) ** 2))

    for _ in range(steps):
        p = c + 1e-6
        J = np.zeros((len(y), c.size))
        for i in range(c.size):
            h = 1e-5 * max(1.0, abs(c[i]))
            cp = c.copy(); cp[i] += h
            cm = c.copy(); cm[i] -= h
            yp = ex.evaluate(ex.set_consts(t, cp), X)
            ym = ex.evaluate(ex.set_consts(t, cm), X)
            if not (np.all(np.isfinite(yp)) and np.all(np.isfinite(ym))):
                return best
            J[:, i] = (yp - ym) / (2 * h)
        r = ex.evaluate(ex.set_consts(t, c), X) - y
        if not np.all(np.isfinite(r)):
            return best
        try:
            A = J.T @ J + lam * np.eye(c.size)
            delta = np.linalg.solve(A, -J.T @ r)
        except np.linalg.LinAlgError:
            return best
        # adım sınırı ve değer sınırı
        delta = np.clip(delta, -5.0, 5.0)
        cand = np.clip(c + delta, -1e6, 1e6)
        yc = ex.evaluate(ex.set_consts(t, cand), X)
        if not np.all(np.isfinite(yc)):
            lam *= 10
            continue
        sse = float(np.sum((yc - y) ** 2))
        if sse < best_sse:
            best_sse = sse
            best = ex.set_consts(t, cand)
            c = cand
        else:
            lam *= 10
        if best_sse < 1e-30:
            break
    return best



# --------------------------------------------------- JÜRİ-FARKINDA SABİT AYARI
def _res_vec(t: ex.Tree, Xrows, ops, target: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Ölçekli artık vektörü: (Σ_r coef·f(X_r) − t_k) / scale_k."""
    v = ex.evaluate(t, Xrows)
    out = np.empty(len(target), dtype=float)
    for k, op in enumerate(ops):
        acc = 0.0
        for (r, coef) in op:
            acc += coef * v[r]
        out[k] = (acc - target[k]) / scale[k]
    return out


def jury_fit_blocks(t: ex.Tree, bench, steps: int = 3, lam: float = 1e-3,
                    jury_weight: float = 1.0, scale_y: Optional[float] = None) -> ex.Tree:
    """
    Hem VERİYE hem JÜRİYE göre sabit ayarı (Gauss-Newton).

      artık = [ (f(X_eğitim) − y)/s_y ,  jw·(D_j f(X_j) − t_j)/(tol_j·|t_j|) ]

    Kuantum yasasını sağlamayan katsayılar cezalandırılır; böylece evrim
    "yasayı bilen" bir arayış yapar. Yalnızca eğitim+doğrulama noktaları.
    """
    cs = ex.consts(t)
    if not cs or len(cs) > 6:
        return t
    blocks = fl.jury_blocks(bench)
    if not blocks:
        return t

    X_d = bench.train.X
    y_d = np.asarray(bench.train.y, float)
    if scale_y is None:
        scale_y = float(np.sqrt(np.mean(y_d ** 2))) or 1.0
    Xrows = [np.asarray(x, float) for x in X_d]
    ops = [[(k, 1.0)] for k in range(len(y_d))]
    target = list(y_d)
    scale = [scale_y] * len(y_d)

    for b in blocks:
        off = len(Xrows[0]) if Xrows else 0
        Xrows = [np.concatenate([Xrows[i], np.asarray(b["X"][i], float)])
                 for i in range(len(Xrows))]
        for op in b["rows"]:
            ops.append([(r + off, c) for (r, c) in op])
        target.extend(list(b["target"]))
        scale.extend(list(np.maximum(b["scale"] / max(jury_weight, 1e-12), 1e-300)))

    target = np.asarray(target, float); scale = np.asarray(scale, float)
    c = np.array(cs, dtype=float)
    best = copy.deepcopy(t)
    r0 = _res_vec(best, Xrows, ops, target, scale)
    if not np.all(np.isfinite(r0)):
        return t
    best_sse = float(np.sum(r0 ** 2))

    for _ in range(max(1, steps)):
        J = np.zeros((len(target), c.size))
        r = _res_vec(ex.set_consts(t, c), Xrows, ops, target, scale)
        if not np.all(np.isfinite(r)):
            return best
        for i in range(c.size):
            h = 1e-5 * max(1.0, abs(c[i]))
            cp = c.copy(); cp[i] += h
            cm = c.copy(); cm[i] -= h
            rp = _res_vec(ex.set_consts(t, cp), Xrows, ops, target, scale)
            rm = _res_vec(ex.set_consts(t, cm), Xrows, ops, target, scale)
            if not (np.all(np.isfinite(rp)) and np.all(np.isfinite(rm))):
                return best
            J[:, i] = (rp - rm) / (2 * h)
        try:
            A = J.T @ J + lam * np.eye(c.size)
            delta = np.linalg.solve(A, -J.T @ r)
        except np.linalg.LinAlgError:
            return best
        delta = np.clip(delta, -5.0, 5.0)
        cand = np.clip(c + delta, -1e6, 1e6)
        rc = _res_vec(ex.set_consts(t, cand), Xrows, ops, target, scale)
        if not np.all(np.isfinite(rc)):
            lam *= 10
            continue
        sse = float(np.sum(rc ** 2))
        if sse < best_sse:
            best_sse = sse
            best = ex.set_consts(t, cand)
            c = cand
        else:
            lam *= 10
    return best


# ------------------------------------------------------------------ popülasyon
class Individual:
    __slots__ = ("tree", "ev", "score", "gen")

    def __init__(self, tree: ex.Tree, gen: int = 0):
        self.tree = tree
        self.gen = gen
        self.ev: Optional[fl.Evaluation] = None
        self.score = math.inf


def _score_ind(ind: Individual, bench, cfg: EAConfig) -> float:
    ev = ind.ev
    if ev is None or not ev.ok:
        return math.inf
    penalty = cfg.cost_weight * (ev.cost / max(bench.cost_budget, 1e-9))
    return math.log10(max(ev.loss_train, 1e-16)) + float(penalty)


def _mk_ind(tree: ex.Tree, bench, cfg: EAConfig, gen: int, do_lm: bool = True) -> Individual:
    if do_lm and ex.n_consts(tree) > 0:
        tree = tune_constants(tree, bench.train.X, bench.train.y, steps=cfg.lm_steps)
        tree = snap_constants(tree, bench.train.X, bench.train.y,
                              float(getattr(bench, "noise_rel", 1e-6)))
    if cfg.simplify:
        tree = simplify(tree, bench.probe_X)
        tree = snap_constants(tree, bench.train.X, bench.train.y,
                              float(getattr(bench, "noise_rel", 1e-6)))
    ind = Individual(tree, gen)
    ind.ev = fl.quick_eval(tree, bench, tier=bench.tier)
    ind.score = _score_ind(ind, bench, cfg)
    # --- KUANTUM KAPISI: ELEME grubundaki kuantum jürisi düşerse aday ölür (∞ ceza)
    if getattr(cfg, "quantum_gate", True) and np.isfinite(ind.score):
        qres = fl.quantum_gate(tree, bench)
        v = fl.quantum_violation([f for f in qres if f.group == "ELEME"])
        # --- ONARIM: yasayı ihlal eden ama umut vadeden adayın sabitleri
        #     hem veriye hem jüriye göre yeniden ayarlanır (EA yasayı bilir).
        if (getattr(cfg, "jury_refit", True) and v > 1.0
                and v <= cfg.quantum_kill * 2.0 and ex.n_consts(tree) > 0):
            def _repair(t0, steps, w):
                """Onarım denemesi; yalnızca ihlali AZALTAN sonuç kabul edilir."""
                nonlocal tree, ind, qres, v
                t2 = jury_fit_blocks(t0, bench, steps=steps, jury_weight=w)
                q2 = fl.quantum_gate(t2, bench)
                v2 = fl.quantum_violation([f for f in q2 if f.group == "ELEME"])
                if v2 < v:
                    tree = t2
                    ind.tree = t2
                    ind.ev = fl.quick_eval(t2, bench, tier=bench.tier)
                    ind.score = _score_ind(ind, bench, cfg)
                    qres, v = q2, v2
                return v2

            v1 = _repair(tree, cfg.jury_steps, cfg.jury_weight)
            # İKİNCİ AŞAMA: veriyi de açıklayan (bariyere yakın) adaylarda uzun onarım.
            # Ucuz/açıklayıcı olmayan adaylar buraya girmez → arama maliyeti düşük kalır.
            nb = getattr(bench, "null_barrier", None)
            promising = (nb is None) or (ind.ev.loss_val <= 3.0 * float(nb)) or (v1 <= 3.0)
            if promising and v > 1.0 and cfg.jury_steps2 > cfg.jury_steps:
                _repair(tree, cfg.jury_steps2, cfg.jury_weight2)
        if qres:
            ind.ev.filters = list(ind.ev.filters) + qres
            ind.ev.notes["quantum_violation"] = v
            if v > cfg.quantum_kill:
                # AĞIR İHLAL: kuantum yasasını açıkça çiğneyen aday anında elenir (∞)
                ind.ev.ok = False
                ind.ev.notes["quantum_gate_blocked"] = 1.0
                ind.score = math.inf
            elif v > 1.0:
                # SINIRDA İHLAL: kademeli ceza (evrim ihlali azaltmaya yönlendirilir)
                ind.score += cfg.quantum_weight * (v - 1.0)
    return ind


# ------------------------------------------------------------------ ana döngü
def run(bench, cfg: Optional[EAConfig] = None,
        progress: Optional[Callable[[dict], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None) -> dict:
    cfg = cfg or EAConfig()
    rng = np.random.default_rng(cfg.seed)
    t_start = time.time()
    nvars = len(bench.var_names)

    seeds: List[ex.Tree] = []
    if cfg.physics_seeded and not cfg.blind_mode:
        seeds = physics_seeds(bench.var_names, key=bench.key)

    islands: List[List[Individual]] = []
    for isl in range(cfg.islands):
        pop: List[Individual] = []
        for i in range(cfg.pop_size // cfg.islands):
            if i < len(seeds) and isl == 0:
                t = seeds[i]
            else:
                t = random_tree(nvars, cfg, rng)
            pop.append(_mk_ind(t, bench, cfg, 0))
        islands.append(pop)

    def top_of(pop: List[Individual]) -> Individual:
        return min(pop, key=lambda z: z.score)

    archive: Dict[str, dict] = {}
    history: List[dict] = []
    best_ever: Optional[fl.Evaluation] = None
    best_ever_score = math.inf
    stall = 0

    def consider_archive(ind: Individual, gen: int, strict: bool = False) -> None:
        nonlocal best_ever, best_ever_score
        if not np.isfinite(ind.score):
            return
        if strict:
            ind.ev = fl.full_eval(ind.ev, bench, tier=bench.tier,
                                  time_budget_us=cfg.time_budget_us,
                                  null_factor=cfg.null_factor)
        key = f"{round(ind.ev.cost,1)}|{round(math.log10(max(ind.ev.loss_train,1e-16)),4)}"
        rec = {
            "gen": gen, "tree": ind.tree, "score": ind.score,
            "cost": ind.ev.cost, "nodes": ind.ev.nodes, "depth": ind.ev.depth,
            "loss_train": ind.ev.loss_train, "loss_val": ind.ev.loss_val,
            "verified": ind.ev.verified, "strict": strict,
            "pretty": ex.to_pretty(ind.tree, bench.var_names),
            "python": ex.to_python(ind.tree, bench.var_names),
        }
        old = archive.get(key)
        if old is None or (not old["strict"] and strict):
            archive[key] = rec
        if ind.score < best_ever_score:
            best_ever_score = ind.score
            best_ever = ind.ev

    for gen in range(1, cfg.generations + 1):
        if should_stop and should_stop():
            history.append({"gen": gen, "stopped": True})
            break

        strict = (gen % cfg.strict_interval == 0) or gen == 1
        for isl in islands:
            isl.sort(key=lambda z: z.score)
            consider_archive(isl[0], gen, strict=strict)
            if strict:
                consider_archive(isl[min(2, len(isl) - 1)], gen, strict=False)

        # --- istatistik
        flat = [ind for isl in islands for ind in isl]
        best = min(flat, key=lambda z: z.score)
        if best_ever is None or best.score < best_ever_score - 1e-6:
            stall = 0
        else:
            stall += 1
        hist = {
            "gen": gen, "gen_": gen * (cfg.pop_size),
            "best_score": best.score if np.isfinite(best.score) else None,
            "best_train_loss": best.ev.loss_train if best.ev else None,
            "best_val_loss": best.ev.loss_val if best.ev else None,
            "cost": best.ev.cost if best.ev else None,
            "nodes": best.ev.nodes if best.ev else None,
            "formula": ex.to_pretty(best.tree, bench.var_names) if best.ev else None,
            "archive": len(archive),
        }
        history.append(hist)
        if progress and (gen % 2 == 0 or gen == 1):
            progress({"type": "gen", **hist, "elapsed": round(time.time() - t_start, 1)})

        if stall >= cfg.patience:
            if progress:
                progress({"type": "log", "msg": f"yakınsama: {cfg.patience} nesil iyileşme yok, durduruldu"})
            break

        # --- göç
        if gen % cfg.migration_every == 0 and len(islands) > 1:
            migrants = [top_of(isl) for isl in islands]
            for i, isl in enumerate(islands):
                donor = migrants[(i + 1) % len(islands)]
                isl[-1] = _mk_ind(copy.deepcopy(donor.tree), bench, cfg, gen)

        # --- yeni nesil
        n_elite = max(2, int(cfg.pop_size // cfg.islands * cfg.elite_frac))
        for isl in islands:
            isl.sort(key=lambda z: z.score)
            new_pop: List[Individual] = [Individual(i.tree, gen) or i for i in isl[:n_elite]]
            for i in range(n_elite):
                new_pop[i].ev = isl[i].ev
                new_pop[i].score = isl[i].score
            while len(new_pop) < len(isl):
                def tourn() -> Individual:
                    k = rng.integers(len(isl), size=cfg.tournament)
                    cands = [isl[int(j)] for j in k]
                    return min(cands, key=lambda z: z.score)
                p1 = tourn()
                if rng.random() < cfg.crossover_rate:
                    p2 = tourn()
                    child = crossover(p1.tree, p2.tree, nvars, rng)
                else:
                    child = copy.deepcopy(p1.tree)
                for _ in range(int(rng.integers(1, 3))):
                    child = mutate(child, nvars, cfg, rng, cheap=(bench.tier == "cheap"),
                                   max_depth=cfg.max_depth)
                if ex.n_nodes(child) > cfg.max_nodes or ex.depth(child) > cfg.max_depth + 1:
                    child = copy.deepcopy(p1.tree)
                    child = mutate(child, nvars, cfg, rng, cheap=(bench.tier == "cheap"),
                                   max_depth=cfg.max_depth)
                new_pop.append(_mk_ind(child, bench, cfg, gen))
            isl[:] = new_pop

    # --- kapanış: arşivi ve seçkinleri tam denetimden geçir
    candidates: List[ex.Tree] = [rec["tree"] for rec in archive.values()]
    for isl in islands:
        isl.sort(key=lambda z: z.score)
        candidates.extend([ind.tree for ind in isl[:3]])
    seen, unique = set(), []
    for t in candidates:
        if repr(t) not in seen:
            seen.add(repr(t))
            unique.append(t)

    hall: List[dict] = []
    seen_forms: set = set()
    for t in unique[:60]:
        ind = _mk_ind(t, bench, cfg, gen=cfg.generations, do_lm=False)
        form = ex.to_pretty(ind.tree, bench.var_names)
        if form in seen_forms:
            continue
        seen_forms.add(form)
        if not np.isfinite(ind.score):
            continue
        ev = fl.full_eval(ind.ev, bench, tier=bench.tier, time_budget_us=cfg.time_budget_us,
                          null_factor=cfg.null_factor)
        # sınav kümesi: YALNIZCA burada, bir kez ölçülür
        y_test = ex.evaluate(t, bench.test.X)
        ev.loss_test = fl.rel_rmse(y_test, bench.test.y)
        ev.notes["max_rel_err_test"] = fl.rel_maxerr(y_test, bench.test.y)
        rec = serialize_eval(ev, bench)
        rec["score_key"] = fl.score(ev, bench, cfg.cost_weight)
        # jüri ihlal oranı (ELEME grubu): sıralamada İKİNCİ ölçüt
        v = 0.0
        for f in rec["filters"]:
            if f.get("group") == "ELEME" and not f["passed"] and f.get("value") and f.get("limit"):
                try:
                    v = max(v, float(f["value"]) / float(f["limit"]))
                except Exception:
                    pass
        rec["violation"] = v
        hall.append(rec)
    # Sıralama: (1) kuantum jürisinden geçenler, (2) en AZ ihlal, (3) en iyi veri,
    #           (4) en ucuz. Böylece doğrulanmış aday yoksa bile en az ihlal eden
    #           şampiyon olur; "ucuz çöp" asla başa geçemez.
    hall.sort(key=lambda r: (not r["verified"], r.get("violation", 0.0),
                             math.log10(max(min(r["loss_val"], r["loss_train"]), 1e-16)),
                             r["cost"]))

    champ = hall[0] if hall else None
    trees_by_formula = {ex.to_pretty(t, bench.var_names): t for t in unique}
    return {
        "bench_key": bench.key,
        "bench_title": bench.title,
        "var_names": bench.var_names,
        "config": asdict(cfg),
        "seed": cfg.seed,
        "generations_run": gen,
        "evaluations": gen * cfg.pop_size,
        "wall_seconds": round(time.time() - t_start, 2),
        "history": history,
        "hall_of_fame": hall,
        "champion": champ,
        "champion_tree": (champ and trees_by_formula.get(champ["formula"])) or None,
        "null_barrier": bench.null_barrier,
        "seeds_used": [ex.to_pretty(s, bench.var_names) for s in seeds] if seeds else [],
        "physics_seeded": cfg.physics_seeded and not cfg.blind_mode,
    }


# ------------------------------------------------------------------ serileştirme
def serialize_eval(ev: fl.Evaluation, bench) -> dict:
    return {
        "formula": ex.to_pretty(ev.tree, bench.var_names),
        "python": ex.to_python(ev.tree, bench.var_names),
        "tree": repr(ev.tree),
        "cost": round(ev.cost, 2),
        "nodes": ev.nodes,
        "depth": ev.depth,
        "eval_us": round(ev.eval_us, 1),
        "loss_train": ev.loss_train,
        "loss_val": ev.loss_val,
        "loss_test": ev.loss_test,
        "max_rel_err_test": ev.notes.get("max_rel_err_test"),
        "ok": ev.ok,
        "verified": ev.verified,
        "filters": [
            {"id": f.id, "name": f.name, "group": f.group, "passed": f.passed,
             "value": None if f.value is None or not np.isfinite(f.value) else float(f.value),
             "limit": f.limit, "detail": f.detail}
            for f in ev.filters
        ],
        "failures": ev.failures,
    }


# ------------------------------------------------------------------ null kalibrasyonu
def calibrate_null_barrier(bench, cfg: Optional[EAConfig] = None,
                           progress: Optional[Callable[[dict], None]] = None) -> float:
    """
    Numeroloji bariyeri: etiketleri karıştırılmış veride aynı mimarinin
    en iyi hatası. Aynı popülasyon büyüklüğü, aynı operatör ailesi, kısa bütçe.
    """
    from . import benchmarks as bm
    cfg = cfg or EAConfig()
    ncfg = copy.copy(cfg)
    ncfg.seed = cfg.seed + 977
    ncfg.generations = max(10, cfg.generations // 4)
    ncfg.physics_seeded = False
    ncfg.patience = 10_000
    null_bench = bm.shuffle_targets(bench, seed=cfg.seed + 13)
    if progress:
        progress({"type": "log", "msg": f"null kalibrasyonu ({null_bench.key}) başladı…"})
    res = run(null_bench, ncfg, progress=None)
    best = min((h["loss_val"] for h in res["hall_of_fame"]), default=float("inf"))
    return float(best)
