"""
expression.py — Sembolik ifade ağaçları, güvenli derleme, maliyet (FLOP) modeli.

Ağaç gösterimi (değişmez, hash'lenebilir, kopyalanabilir demetler):

    ('x', i)              i. değişken
    ('c', value)          sabit
    ('f', op, a, b, ...)  operatör düğümü

Maliyet modeli iki katmanlıdır ve "donanıma zarar vermeme" hedefini somutlaştırır:

  * CHEAP_OPS   : + - * / ve küçük tamsayı kuvvetleri -> ucuz, önbellek dostu
  * EXPENSIVE_OPS: exp, log, sin, cos, tanh, gerçel kuvvet -> transandantal
    birim çağrıları; modern CPU'da ~10-50x daha pahalı ve güç yoğunluğu yüksek

Evrimsel algoritma maliyeti amaç fonksiyonuna dahil eder (Pareto: hata-maliyet)
ve "ucuz katman" filtresi transandantal operatörleri son üründen tamamen eler.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple
import numpy as np

Tree = tuple

# ------------------------------------------------------------------ maliyet modeli
# Maliyet birimi = 1 "temel işlem" (yaklaşık 1 doygun FP add/mul).
COST_ADD = 1.0
COST_SUB = 1.0
COST_MUL = 1.0
COST_DIV = 3.0        # bölme: bölücü birimi, ~3x çarpma
COST_SQ = 1.0         # x*x
COST_SQRT = 4.0       # FP karekök
COST_ABS = 1.0
COST_NEG = 0.0        # işaret biti
COST_EXP = 14.0       # exp/log/sin/cos: transandantal birim, yüksek güç
COST_LOG = 14.0
COST_TRIG = 16.0
COST_TANH = 18.0
COST_POW_INT = {2: 1.0, 3: 2.0, 4: 3.0}
COST_POW_REAL = 26.0

EXPENSIVE_OPS = {"exp", "log", "sin", "cos", "tanh", "powr"}

_OPS_META = {
    "add":  (2, COST_ADD),
    "sub":  (2, COST_SUB),
    "mul":  (2, COST_MUL),
    "div":  (2, COST_DIV),
    "neg":  (1, COST_NEG),
    "abs":  (1, COST_ABS),
    "sq":   (1, COST_SQ),
    "sqrt": (1, COST_SQRT),
    "exp":  (1, COST_EXP),
    "log":  (1, COST_LOG),
    "sin":  (1, COST_TRIG),
    "cos":  (1, COST_TRIG),
    "tanh": (1, COST_TANH),
    "pow2": (1, COST_POW_INT[2]),
    "pow3": (1, COST_POW_INT[3]),
    "powr": (2, COST_POW_REAL),
}

UNARY_SAFE = ["neg", "abs", "sq", "sqrt", "exp", "log", "sin", "cos", "tanh", "pow2", "pow3"]
BINARY_SAFE = ["add", "sub", "mul", "div", "powr"]
CHEAP_UNARY = ["neg", "abs", "sq", "sqrt", "pow2", "pow3"]
CHEAP_BINARY = ["add", "sub", "mul", "div"]

FUNCS = {
    "neg":  lambda x: -x,
    "abs":  np.abs,
    "sq":   lambda x: x * x,
    "sqrt": np.sqrt,
    "exp":  np.exp,
    "log":  np.log,
    "sin":  np.sin,
    "cos":  np.cos,
    "tanh": np.tanh,
    "pow2": lambda x: x * x,
    "pow3": lambda x: x * x * x,
    "add":  lambda a, b: a + b,
    "sub":  lambda a, b: a - b,
    "mul":  lambda a, b: a * b,
    "div":  lambda a, b: a / b,
    "powr": lambda a, b: np.power(a, b),
}

_PREC = {"add": 1, "sub": 1, "mul": 2, "div": 2, "neg": 3, "powr": 4}
_MAX_PREC = 5


# ------------------------------------------------------------------ yardımcılar
def variable(i: int) -> Tree:
    return ("x", int(i))


def const(v: float) -> Tree:
    return ("c", float(v))


def is_leaf(t: Tree) -> bool:
    return t[0] in ("x", "c")


def n_nodes(t: Tree) -> int:
    if is_leaf(t):
        return 1
    return 1 + sum(n_nodes(c) for c in t[2:])


def depth(t: Tree) -> int:
    if is_leaf(t):
        return 1
    return 1 + max(depth(c) for c in t[2:])


def cost(t: Tree) -> float:
    """İfadeyi bir kez değerlendirmenin yaklaşık temel-işlem maliyeti."""
    if t[0] == "c":
        return 0.0
    if t[0] == "x":
        return 0.0
    op = t[1]
    base = _OPS_META[op][1]
    return base + sum(cost(c) for c in t[2:])


def ops_used(t: Tree, acc: set | None = None) -> set:
    if acc is None:
        acc = set()
    if not is_leaf(t):
        acc.add(t[1])
        for c in t[2:]:
            ops_used(c, acc)
    return acc


def has_expensive_op(t: Tree) -> bool:
    return bool(ops_used(t) & EXPENSIVE_OPS)


def consts(t: Tree) -> List[float]:
    if t[0] == "c":
        return [t[1]]
    if t[0] == "x":
        return []
    out: List[float] = []
    for c in t[2:]:
        out.extend(consts(c))
    return out


def n_consts(t: Tree) -> int:
    return len(consts(t))


def n_vars_used(t: Tree) -> int:
    if t[0] == "x":
        return 1
    if t[0] == "c":
        return 0
    s: set[int] = set()
    _collect_vars(t, s)
    return len(s)


def _collect_vars(t: Tree, s: set) -> None:
    if t[0] == "x":
        s.add(t[1])
    elif t[0] == "f":
        for c in t[2:]:
            _collect_vars(c, s)


def set_consts(t: Tree, values: Sequence[float]) -> Tree:
    it = iter(values)

    def rec(node: Tree) -> Tree:
        if node[0] == "c":
            return ("c", float(next(it)))
        if node[0] == "x":
            return node
        return ("f", node[1]) + tuple(rec(c) for c in node[2:])

    return rec(t)


def max_abs_const(t: Tree) -> float:
    cs = consts(t)
    return max((abs(c) for c in cs), default=0.0)


def signatures(t: Tree) -> str:
    return repr(t)


# Basit rasyonel değerler ve "yazılış uzunluğu" (sıralama: en kısa tanım önce).
# MDL (Minimum Description Length) ilkesinin sabitler üzerindeki uygulamasıdır;
# seçimde yalnızca EŞİT DOĞRULUKTAKİ formlar arasında ayırt edici olarak kullanılır.
_SIMPLE_FRACTIONS = {
    0.0: 0, 1.0: 1, -1.0: 1, 0.5: 2, -0.5: 2, 2.0: 2, -2.0: 2,
    0.25: 3, -0.25: 3, 3.0: 3, -3.0: 3, 1.5: 3, -1.5: 3, 4.0: 3, -4.0: 3,
    1 / 3: 4, -1 / 3: 4, 1 / 6: 4, -1 / 6: 4, 2 / 3: 4, -2 / 3: 4,
    6.0: 4, -6.0: 4, 8.0: 4, -8.0: 4, 12.0: 4, -12.0: 4,
}


def const_dl(t: Tree) -> float:
    """Sabitlerin toplam tanım uzunluğu (basit rasyonel = kısa, uydurma basamak = uzun)."""
    tot = 0.0
    for v in consts(t):
        hit = None
        for k, r in _SIMPLE_FRACTIONS.items():
            if abs(v - k) <= 1e-9 * max(1.0, abs(k)):
                hit = r
                break
        if hit is None:
            s = f"{v:.8g}".replace("-", "").replace(".", "").lstrip("0")
            tot += 12.0 + len(s)
        else:
            tot += hit
    return tot


# ------------------------------------------------------------------ derleme (hızlı yol)
@lru_cache(maxsize=4096)
def _compile_cached(t: Tree, nvars: int):
    """Ağacı numpy vektörleştirilmiş bir Python lambda'ya derler."""
    from .expression import FUNCS as _F  # kendi modülü
    ns: Dict[str, object] = {k: v for k, v in _F.items()}
    ns["np"] = np

    def build(node: Tree) -> str:
        if node[0] == "x":
            return f"X[{node[1]}]"
        if node[0] == "c":
            return repr(node[1])
        op = node[1]
        args = [build(c) for c in node[2:]]
        return f"{op}(" + ",".join(args) + ")"

    src = build(t)
    code = compile(f"lambda X: {src}", "<expr>", "eval")
    # DİKKAT: lambda gövdesi çalışma anında GLOBALS sözlüğüne bakar;
    # isimleri locals'a koymak sessizce NameError'a yol açar.
    g: Dict[str, object] = {"__builtins__": {}}
    g.update(ns)
    return eval(code, g)


def evaluate(t: Tree, X: Sequence[np.ndarray]) -> np.ndarray:
    """X: değişken dizileri listesi. NaN/inf üretirse np.nan'lı dizi döner."""
    fn = _compile_cached(t, len(X))
    with np.errstate(all="ignore"):
        try:
            y = fn(X)
        except Exception:
            return np.full(np.asarray(X[0]).shape, np.nan)
    y = np.asarray(y, dtype=float)
    if y.shape == ():
        y = np.full(np.asarray(X[0]).shape, float(y))
    return y


# ------------------------------------------------------------------ güzel yazım
_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _fmt_const(v: float) -> str:
    if v == int(v) and abs(v) < 1e6:
        return str(int(v))
    return f"{v:.6g}"


def to_pretty(t: Tree, varnames: Sequence[str]) -> str:
    """İnsan okunur, operatör önceliğine saygılı gösterim (Unicode)."""
    def rec(node: Tree) -> Tuple[str, int]:
        if node[0] == "x":
            return varnames[node[1]], _MAX_PREC
        if node[0] == "c":
            return _fmt_const(node[1]), _MAX_PREC
        op = node[1]
        if op == "neg":
            s, p = rec(node[2])
            body = f"−{s}" if p >= 4 else f"−({s})"
            return body, 3
        if op in ("sq", "pow2"):
            s, p = rec(node[2])
            body = f"{s}²" if p >= _MAX_PREC else f"({s})²"
            return body, 3
        if op == "pow3":
            s, p = rec(node[2])
            body = f"{s}³" if p >= _MAX_PREC else f"({s})³"
            return body, 3
        if op in ("abs", "sqrt", "exp", "log", "sin", "cos", "tanh"):
            s, _ = rec(node[2])
            nice = {"abs": f"|{s}|", "sqrt": f"√{s}", "exp": f"e^({s})",
                    "log": f"ln({s})", "sin": f"sin({s})", "cos": f"cos({s})",
                    "tanh": f"tanh({s})"}[op]
            return nice, _MAX_PREC
        a, pa = rec(node[2])
        b, pb = rec(node[3])
        if op == "add":
            return f"{a} + {b}", 1
        if op == "sub":
            return f"{a} − {b}", 1
        if op == "mul":
            left = a if pa >= 2 else f"({a})"
            right = b if pb >= 2 else f"({b})"
            return f"{left}·{right}", 2
        if op == "div":
            left = a if pa >= 2 else f"({a})"
            right = b if pb >= 3 else f"({b})"
            return f"{left}/{right}", 2
        if op == "powr":
            return f"({a})^({b})", 4
        return f"{op}({a},{b})", 0

    return rec(t)[0]


def to_python(t: Tree, varnames: Sequence[str]) -> str:
    """Yeniden üretilebilir Python/numpy kaynak kodu."""
    if t[0] == "x":
        return varnames[t[1]]
    if t[0] == "c":
        return repr(t[1])
    op = t[1]
    args = [to_python(c, varnames) for c in t[2:]]
    if op == "sq":
        return f"({args[0]}**2)"
    if op == "pow2":
        return f"({args[0]}**2)"
    if op == "pow3":
        return f"({args[0]}**3)"
    if op == "neg":
        return f"(-{args[0]})"
    if op == "powr":
        return f"np.power({args[0]}, {args[1]})"
    if op == "abs":
        return f"np.abs({args[0]})"
    fn = {"sqrt": "np.sqrt", "exp": "np.exp", "log": "np.log",
          "sin": "np.sin", "cos": "np.cos", "tanh": "np.tanh"}.get(op)
    if fn:
        return f"{fn}({args[0]})"
    sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op]
    return f"({args[0]} {sym} {args[1]})"
