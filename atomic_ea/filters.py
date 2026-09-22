"""
filters.py — "Numeroloji savar" sert filtre katmanı.

Her aday formül yayımlanabilmek için bir filtre zincirinden geçmek zorundadır.
Filtreler iki sınıfa ayrılır:

  (A) ELEME filtreleri  -> başarısızlık = aday ölür (fitness = inf)
      F1  Sözdizimi / ucuz-katman (transandantal operatör yasağı)
      F2  Ölçek simetrisi (boyut analizi: E ~ Z^p, ölçek yasası veriden kurulur)
      F3  Gürbüzlük ve donanım güvenliği (NaN/inf, taşma riski, süre bütçesi)
      F4  Numeroloji bariyeri (etiket permütasyonu null testi - gerçek fizik
          gürültüye uymaz; uydurma formül ise uyar)

  (B) DOĞRULAMA filtreleri -> başarısızlık = "doğrulanmış" statüsü verilmez
      F5  Hellmann-Feynman:  dE/dZ = -<1/r>   (sayısal referansa karşı)
      F6  Virial / eylem durağanlığı:  E = -(Z/2)<1/r>  ve  2<T>+<V> ~ 0
      F7  Ekstrapolasyon: eğitim bölgesinin DIŞINDA da tolerans içinde

Sınav (test) kümesi hiçbir filtrede kullanılmaz; yalnızca son raporda bir kez
ölçülür. Böylece "test kümesine göre ayarlama" (sızıntı) engellenir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import math
import time
import numpy as np

from . import expression as ex


# ----------------------------------------------------------------- veri yapıları
@dataclass
class FilterResult:
    id: str
    name: str
    passed: bool
    value: Optional[float] = None
    limit: Optional[float] = None
    detail: str = ""
    group: str = "ELEME"          # ELEME | DOĞRULAMA


@dataclass
class Evaluation:
    tree: ex.Tree
    ok: bool = False                                  # tüm ELEME filtreleri geçti mi
    verified: bool = False                            # tüm filtreler geçti mi
    loss_train: float = math.inf
    loss_val: float = math.inf
    loss_test: float = math.inf
    cost: float = math.inf
    nodes: int = 0
    depth: int = 0
    eval_us: float = math.inf                          # 1000 noktada ölçülen süre
    filters: List[FilterResult] = field(default_factory=list)
    notes: Dict[str, float] = field(default_factory=dict)

    def fmap(self) -> Dict[str, FilterResult]:
        return {f.id: f for f in self.filters}

    @property
    def failures(self) -> List[str]:
        return [f.id for f in self.filters if not f.passed]


# ----------------------------------------------------------------- yardımcılar
def rel_rmse(pred: np.ndarray, y: np.ndarray) -> float:
    """Ölçek-duyarsız hata: RMSE / RMS(y).  (birimsiz, karşılaştırılabilir)"""
    if pred is None or not np.all(np.isfinite(pred)):
        return math.inf
    d = pred - y
    denom = float(np.sqrt(np.mean(y**2))) or 1.0
    return float(np.sqrt(np.mean(d**2)) / denom)


def rel_maxerr(pred: np.ndarray, y: np.ndarray) -> float:
    if pred is None or not np.all(np.isfinite(pred)):
        return math.inf
    denom = float(np.max(np.abs(y))) or 1.0
    return float(np.max(np.abs(pred - y)) / denom)


def _eval_sets(tree: ex.Tree, bench, sets: Sequence[str]) -> Tuple[Dict[str, np.ndarray], float]:
    preds: Dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    for s in sets:
        ds = getattr(bench, s)
        preds[s] = ex.evaluate(tree, ds.X)
    t1 = time.perf_counter()
    return preds, (t1 - t0) * 1e6       # mikrosaniye


# ----------------------------------------------------------------- (A) ELEME
def F1_syntax(tree: ex.Tree, bench, tier: str = "cheap") -> FilterResult:
    ops = ex.ops_used(tree)
    if tier == "cheap":
        bad = sorted(ops & ex.EXPENSIVE_OPS)
        return FilterResult(
            "F1", "ucuz katman (transandantal operatör yasağı)", not bad,
            value=float(len(bad)),
            detail=("yasaklı operatör yok" if not bad
                    else "yasaklı: " + ", ".join(bad) + " (exp/log/sin/cos/powr)"),
        )
    return FilterResult("F1", "operatör sözdizimi", True, detail=f"operatörler: {sorted(ops)}")


def F2_scaling(tree: ex.Tree, bench) -> FilterResult:
    """
    Boyut analizi filtresi (veriden kurulan ölçek yasası).

    Sayısal çözücüden, E(Z,n) için ölçek yasası E(lam Z, n) = lam^2 E(Z,n)
    olduğu bağımsız olarak doğrulanmıştır. Aday formül de aynı log-log eğimi
    vermek zorundadır; aksi halde birim/boyut tutarsızdır (numeroloji kokusu).
    """
    if not bench.scaling_specs:
        return FilterResult("F2", "ölçek simetrisi", True, detail="bu problemde ölçek yasası yok/atlandı")
    worst = 0.0
    detail = []
    for spec in bench.scaling_specs:
        vi, p = spec["var"], spec["power"]
        lam = spec["lambdas"]
        X0 = list(spec["X0"])
        y0 = ex.evaluate(tree, X0)
        for L in lam:
            XL = list(X0)
            XL[vi] = np.asarray(XL[vi]) * L
            yL = ex.evaluate(tree, XL)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = yL / (L**p * y0)
            r = np.abs(ratio - 1.0)
            r = r[np.isfinite(r)]
            worst = max(worst, float(np.max(r)) if r.size else math.inf)
        detail.append(f"Δlog/Δlog({bench.var_names[vi]})≈{p}: sapma {worst:.2e}")
    return FilterResult("F2", "ölçek simetrisi (boyut analizi)", worst <= 5e-2,
                        worst, 5e-2, "; ".join(detail))


def F3_robustness(tree: ex.Tree, bench, eval_us: float, time_budget_us: float) -> FilterResult:
    """
    Donanım güvenliği: statik + dinamik taşma analizi.

    * statik: |sabit| > 1e6, derinlik > 14, powr üssü büyük, maliyet > bütçe
    * dinamik: genişletilmiş bölgede NaN/inf ve |değer| > 1e12 yasak
    * süre: 1000 nokta değerlendirmesi bütçeyi aşamaz (donanımı yormayan sınır)
    """
    issues: List[str] = []
    if ex.max_abs_const(tree) > 1e6:
        issues.append(f"|sabit|={ex.max_abs_const(tree):.1e}>1e6")
    if ex.depth(tree) > 14:
        issues.append(f"derinlik {ex.depth(tree)}>14")
    if ex.cost(tree) > bench.cost_budget:
        issues.append(f"maliyet {ex.cost(tree):.1f}>{bench.cost_budget}")
    if "powr" in ex.ops_used(tree):
        issues.append("gerçel kuvvet (taşma riski + pahalı)")

    # dinamik: genişletilmiş bölge
    Xs = bench.probe_X
    y = ex.evaluate(tree, Xs)
    if not np.all(np.isfinite(y)):
        issues.append("NaN/inf üretiyor")
    else:
        m = float(np.max(np.abs(y)))
        if m > 1e12:
            issues.append(f"|değer|max={m:.1e} taşma riski")
    if eval_us > time_budget_us:
        issues.append(f"süre {eval_us:.1f}us>{time_budget_us:.0f}us/1000nokta")

    return FilterResult("F3", "gürbüzlük + donanım güvenliği", not issues,
                        eval_us, time_budget_us,
                        "güvenli" if not issues else "; ".join(issues))


def F4_null_barrier(loss_val: float, bench, factor: float = 5.0) -> FilterResult:
    """
    Numeroloji bariyeri.

    Aynı evrimsel mimari, ETİKETLERİ KARIŞTIRILMIŞ (fiziksel anlamı olmayan)
    veri üzerinde çalıştırılır; elde edilen en iyi hata `null_barrier`'dır.
    Gerçek bir yasa bu bariyerin `factor` katı kadar altına inmelidir.
    Uydurma/numerolojik bir ifade bariyeri geçemez çünkü gürültüye ancak
    gürültü kadar uyulur.
    """
    barrier = bench.null_barrier
    if barrier is None or not np.isfinite(barrier):
        return FilterResult("F4", "numeroloji bariyeri (null test)", True,
                            detail="kalibrasyon yok, filtre atlandı (rapor edilir)")
    ok = loss_val <= barrier / factor
    return FilterResult("F4", "numeroloji bariyeri (null test)", ok,
                        loss_val, barrier / factor,
                        f"aday {loss_val:.3e} vs bariyer {barrier/factor:.3e} "
                        f"(karışık etiket en iyisi {barrier:.3e} / {factor:g})")


# ----------------------------------------------------------------- (B) DOĞRULAMA
def F5_hellmann_feynman(tree: ex.Tree, bench) -> FilterResult:
    """
    Hellmann-Feynman teoremi:  dE/dZ = <dH/dZ> = -<1/r>.

    Aday E(Z,n) formülünün Z'ye göre sayısal türevi, BAĞIMSIZ sayısal çözücüden
    gelen <1/r> referansıyla karşılaştırılır. Bu, iki ayrı fiziksel büyüklüğü
    birbirine bağlayan bir kısıttır; veri uydurarak sağlanması mümkün değildir.
    """
    spec = bench.physics_checks.get("hellmann_feynman")
    if spec is None:
        return FilterResult("F5", "Hellmann-Feynman (dE/dZ = -<1/r>)", True,
                            group="DOĞRULAMA", detail="uygulanamaz")
    dZ = 1e-3
    Xp = list(spec["X"])
    Xp[0] = np.asarray(Xp[0]) + dZ
    Xm = list(spec["X"])
    Xm[0] = np.asarray(Xm[0]) - dZ
    yp = ex.evaluate(tree, Xp)
    ym = ex.evaluate(tree, Xm)
    dEdZ = (yp - ym) / (2 * dZ)
    ref = -spec["inv_r"]                     # bağımsız sayısal <1/r> referansı
    err = rel_rmse(dEdZ, ref)
    return FilterResult("F5", "Hellmann-Feynman (dE/dZ = -<1/r>)", err <= 1e-2,
                        err, 1e-2, f"göreli hata {err:.3e}", group="DOĞRULAMA")


def F6_virial_action(tree: ex.Tree, bench) -> FilterResult:
    """
    Virial + eylem durağanlığı:  E = -(Z/2)<1/r>.

    Kanıt: eylem ilkesi (Hamilton) uyarınca dalga fonksiyonunun ölçek
    dönüşümü altında  A(lam)=<H>(lam)=lam^2<T>+lam<V>  durağan olmalıdır;
    dA/dlam|_1 = 0 => 2<T>+<V>=0, virial teoremi 2<T> = -<V>, ve
    Hellmann-Feynman ile birleşince  E = -(Z/2)<1/r>  elde edilir.
    Referans çözümün eylem artığı da burada raporlanır.
    """
    spec = bench.physics_checks.get("virial")
    if spec is None:
        return FilterResult("F6", "virial / eylem durağanlığı", True,
                            group="DOĞRULAMA", detail="uygulanamaz")
    y = ex.evaluate(tree, spec["X"])
    ref = -0.5 * np.asarray(spec["X"][0]) * spec["inv_r"]
    err = rel_rmse(y, ref)
    return FilterResult("F6", "virial / eylem durağanlığı E=-(Z/2)<1/r>", err <= 1e-2,
                        err, 1e-2, f"göreli hata {err:.3e}", group="DOĞRULAMA")


def F7_extrapolation(loss_val: float, tol: float, factor: float = 4.0) -> FilterResult:
    return FilterResult("F7", "ekstrapolasyon (eğitim dışı rejim)", loss_val <= tol * factor,
                        loss_val, tol * factor,
                        f"doğrulama hatası {loss_val:.3e} ≤ {tol*factor:.1e}",
                        group="DOĞRULAMA")


def F8_reference_action(bench) -> FilterResult:
    """Referans çözücünün eylem artığı: veri kaynağının güvenilirliği."""
    v = float(bench.notes.get("action_residual", float("nan")))
    ok = np.isfinite(v) and abs(v) < 1e-2
    return FilterResult("F8", "referans eylem artığı |2<T>+<V>|/|E|", ok, abs(v), 1e-2,
                        f"sayısal referans göreli artığı {v:.2e}", group="DOĞRULAMA")


# ----------------------------------------------------------------- kaskad değerlendirme
def quick_eval(tree: ex.Tree, bench, tier: str = "cheap") -> Evaluation:
    """Ucuz aşama: yalnızca sözdizimi + hata. Popülasyonun tamamı için çağrılır."""
    ev = Evaluation(tree=tree, cost=ex.cost(tree), nodes=ex.n_nodes(tree), depth=ex.depth(tree))
    f1 = F1_syntax(tree, bench, tier=tier)
    ev.filters = [f1]
    if not f1.passed:
        return ev
    preds, us = _eval_sets(tree, bench, ("train", "val"))
    ev.eval_us = us
    ev.loss_train = rel_rmse(preds["train"], bench.train.y)
    ev.loss_val = rel_rmse(preds["val"], bench.val.y)
    ev.ok = np.isfinite(ev.loss_train) and np.isfinite(ev.loss_val)
    return ev


def full_eval(ev: Evaluation, bench, tier: str = "cheap",
              time_budget_us: float = 400.0, null_factor: float = 5.0) -> Evaluation:
    """Pahalı aşama: yalnızca ön elemeden geçen az sayıda aday için çağrılır."""
    fs: List[FilterResult] = list(ev.filters)
    fs.append(F2_scaling(ev.tree, bench))
    fs.append(F3_robustness(ev.tree, bench, ev.eval_us, time_budget_us))
    fs.append(F4_null_barrier(ev.loss_val, bench, factor=null_factor))
    fs.append(F5_hellmann_feynman(ev.tree, bench))
    fs.append(F6_virial_action(ev.tree, bench))
    fs.append(F7_extrapolation(ev.loss_val, bench.tol_rel))
    fs.append(F8_reference_action(bench))
    ev.filters = fs
    ev.ok = all(f.passed for f in fs if f.group == "ELEME")
    ev.verified = ev.ok and all(f.passed for f in fs)
    return ev


def score(ev: Evaluation, bench, cost_weight: float = 0.05) -> float:
    """
    Evrimsel amaç:  hata + maliyet cezası (UYGUNLUK TABANI İLE).

    Kritik ayrıntı: hata, referans verinin ölçüm gürültüsü tabanının
    (bench.noise_rel) altına indiğinde artık bilgi taşımaz. Bu yüzden skorda
    log10(max(hata, gürültü_tabanı)) kullanılır. Sonuç: gürültü tabanına
    ulaştıktan sonra evrim yalnızca DAHA UCUZ formu arar.

    Bu, "en basit matematiksel formül" hedefinin keyfi bir estetik değil,
    verinin bilgi içeriğine dayanan (MDL/ölçüm belirsizliği) bir gerekçesi
    olmasını sağlar.
    """
    if not ev.ok:
        return math.inf
    if not np.isfinite(ev.loss_train):
        return math.inf
    penalty = cost_weight * (ev.cost / max(bench.cost_budget, 1e-9))
    floor = float(getattr(bench, "noise_rel", 1e-12) or 1e-12)
    loss = max(ev.loss_train, floor)
    # MDL terimi: eşit doğrulukta daha KISA TANIMLI (basit rasyonel sabitli) form kazanır.
    mdl = 0.0025 * ex.const_dl(ev.tree)
    return math.log10(loss) + penalty + mdl
