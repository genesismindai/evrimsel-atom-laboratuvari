# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 genesismindai (Evrimsel Atom Laboratuvarı)
#
# Bu program özgür yazılımdır: GNU Genel Kamu Lisansı (GPL) sürüm 3 veya
# sonraki sürümleri koşulları altında yeniden dağıtabilir ve/veya
# değiştirebilirsiniz. Ayrıntılar için LICENSE dosyasına bakın.
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
    Boyut/ölçek analizi filtresi. Üç tür denetim:

      * "exact": E(λx, ...) = λ^p·E(x, ...) — veriden bağımsız doğrulanan ölçek yasası
      * "also":  diğer değişkenlerin eşzamanlı θ→λ^q değişimi (ör. R→R/λ, Z→λZ)
      * "asymptotic": d log|y| / d log x üssünün verilen noktalarda hedefe yakınlığı
                     (ör. çok elektronlu atomlarda Z→∞ için üs → 2)
    """
    if not bench.scaling_specs:
        return FilterResult("F2", "ölçek simetrisi", True, detail="bu problemde ölçek yasası yok/atlandı")
    worst = 0.0
    details: List[str] = []
    for spec in bench.scaling_specs:
        kind = spec.get("kind", "exact")
        if kind == "asymptotic":
            vi = spec["var"]; tgt = float(spec["target_exponent"])
            d = float(spec.get("delta", 0.2)); tol = float(spec.get("tolerance", 0.05))
            worst_local = 0.0
            for pt in spec["points"]:
                X0 = list(spec.get("X0", [None, None]))
                X0[vi] = np.array([float(pt[0])])
                if len(pt) > 1:
                    X0[1 if vi == 0 else 0] = np.array([float(pt[1])])
                Xp = list(X0); Xm = list(X0)
                Xp[vi] = Xp[vi] * (1.0 + d); Xm[vi] = Xm[vi] * (1.0 - d)
                yp = float(np.ravel(ex.evaluate(tree, Xp))[0])
                ym = float(np.ravel(ex.evaluate(tree, Xm))[0])
                y0 = float(np.ravel(ex.evaluate(tree, X0))[0])
                if not all(np.isfinite([yp, ym, y0])) or min(abs(y0), abs(yp), abs(ym)) < 1e-12:
                    worst_local = math.inf; break
                p_eff = (math.log(abs(yp)) - math.log(abs(ym))) / (math.log(Xp[vi][0]) - math.log(Xm[vi][0]))
                worst_local = max(worst_local, abs(p_eff - tgt))
            worst = max(worst, worst_local / max(tol, 1e-9))
            details.append(f"asimptotik üs → {tgt}: sapma {worst_local:.2e}")
            continue
        vi, pw = spec["var"], spec["power"]
        also = spec.get("also", [])
        tol = float(spec.get("tolerance", 5e-2))
        X0 = list(spec["X0"])
        y0 = ex.evaluate(tree, X0)
        worst_local = 0.0
        for L in spec["lambdas"]:
            XL = list(X0)
            XL[vi] = np.asarray(XL[vi]) * L
            for (j, q) in also:
                XL[j] = np.asarray(XL[j]) * (L ** q)
            yL = ex.evaluate(tree, XL)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = yL / (L ** pw * y0)
            r = np.abs(ratio - 1.0)
            r = r[np.isfinite(r)]
            worst_local = max(worst_local, float(np.max(r)) if r.size else math.inf)
        worst = max(worst, worst_local / max(tol, 1e-9))
        tag = f"eşzamanlı({bench.var_names[vi]}×λ" + "".join(
            f", {bench.var_names[j]}×λ^{q}" for j, q in also) + f")" if also else bench.var_names[vi]
        details.append(f"ölçek {tag}: en kötü sapma {worst_local:.2e} (tolerans {tol:.0e})")
    return FilterResult("F2", "ölçek simetrisi (boyut analizi)", worst <= 1.0,
                        worst, 1.0, "; ".join(details) + " [normalize edilmiş artık]")


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
    idx = int(spec.get("idx", 0))            # türev alınan değişken (Z)
    dZ = 1e-3
    Xp = list(spec["X"]); Xm = list(spec["X"])
    Xp[idx] = np.asarray(Xp[idx], float) + dZ
    Xm[idx] = np.asarray(Xm[idx], float) - dZ
    dEdZ = (ex.evaluate(tree, Xp) - ex.evaluate(tree, Xm)) / (2 * dZ)
    ref = -np.asarray(spec["inv_r"], float)   # BAĞIMSIZ <Σ1/r_i> referansı
    err = rel_rmse(dEdZ, ref)
    tol = float(spec.get("tolerance", 1e-2))
    return FilterResult("F5", f"Hellmann-Feynman (dE/dZ = -<1/r>) [{bench.key}]",
                        err <= tol, err, tol,
                        f"∂E/∂Z ile bağımsız <Σ1/r> referansı: göreli hata {err:.3e}",
                        group=("ELEME" if spec.get("eliminating") else "DOĞRULAMA"))


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

    if "T" in spec:          # MOLEKÜL: genelleştirilmiş virial  T = -E - R·dE/dR
        R = np.asarray(spec["X"][0], float)
        h = 1e-3 * np.maximum(np.abs(R), 1e-3)
        Xp = [R + h] + [np.asarray(x) for x in spec["X"][1:]]
        Xm = [R - h] + [np.asarray(x) for x in spec["X"][1:]]
        dEdR = (ex.evaluate(tree, Xp) - ex.evaluate(tree, Xm)) / (2 * h)
        T_impl = -ex.evaluate(tree, [R] + [np.asarray(x) for x in spec["X"][1:]]) - R * dEdR
        err = rel_rmse(T_impl, np.asarray(spec["T"], float))
        tol = float(spec.get("tolerance", 5e-2))
        return FilterResult("F6", f"genelleştirilmiş virial (T = -E - R·dE/dR) [{bench.key}]",
                            err <= tol, err, tol,
                            f"formülün örtük kinetik enerjisi bağımsız çözücünün T'siyle "
                            f"uyuşuyor mu: göreli hata {err:.3e}",
                            group=("ELEME" if spec.get("eliminating") else "DOĞRULAMA"))

    y = ex.evaluate(tree, spec["X"])
    ref = -0.5 * np.asarray(spec["X"][0]) * spec["inv_r"]
    err = rel_rmse(y, ref)
    return FilterResult("F6", "virial / eylem durağanlığı E=-(Z/2)<1/r>", err <= 1e-2,
                        err, 1e-2, f"göreli hata {err:.3e}", group="DOĞRULAMA")


def F10_electron_repulsion(tree: ex.Tree, bench) -> FilterResult:
    """
    Elektron–elektron itmesi jürisi (çok elektronlu atomlar).

    Virial (T = -E) + Hellmann-Feynman (V_ne = Z·dE/dZ) + E = T + V_ne + V_ee
    üçlüsünden, formülün ÖRTÜK olarak öngördüğü itme enerjisi çıkar:

        V_ee(örtük) = 2·E(Z,N) - Z·∂E/∂Z

    Bu, bağımsız HF çözücünün hesapladığı V_ee ile karşılaştırılır. Elektronları
    "yok eden" (N'siz, itmesiz) ucuz formüller burada ölür: örtük V_ee ∈ [pozitif,
    doğru mertebe] olmalıdır. Kuantum fiziğinin en zor terimi böylece ZORUNLU olur.
    """
    spec = bench.physics_checks.get("electron_repulsion")
    if spec is None:
        return FilterResult("F10", "elektron-elektron itmesi (V_ee)", True,
                            group="DOĞRULAMA", detail="uygulanamaz")
    Z = np.asarray(spec["X"][0], float)
    N = np.asarray(spec["X"][1], float)
    dZ = 1e-3 * np.maximum(np.abs(Z), 1e-3)
    dEdZ = (ex.evaluate(tree, [Z + dZ, N]) - ex.evaluate(tree, [Z - dZ, N])) / (2 * dZ)
    E = ex.evaluate(tree, [Z, N])
    V_ee_impl = 2.0 * E - Z * dEdZ
    ref = np.asarray(spec["V_ee"], float)
    err = rel_rmse(V_ee_impl, ref)
    tol = float(spec.get("tolerance", 5e-2))
    neg = float(np.min(V_ee_impl))
    ok = (err <= tol) and (neg > 0.0)
    detail = (f"örtük V_ee = 2E - Z·∂E/∂Z vs bağımsız çözücü: göreli hata {err:.3e}; "
              f"en küçük örtük V_ee = {neg:.3e} (>0 olmalı)")
    return FilterResult("F10", f"elektron-elektron itmesi V_ee [{bench.key}]", ok, err, tol,
                        detail, group=("ELEME" if spec.get("eliminating") else "DOĞRULAMA"))


def F11_electron_count(tree: ex.Tree, bench) -> FilterResult:
    """
    Elektron sayısı tepkisi (çok elektronlu atomlar).

    İzoelektronik çiftlerde (aynı Z, N=2 vs N=4) formülün öngördüğü fark,
    bağımsız çözücünün verdiği farkla uyuşmalı:
        f(Z,4) - f(Z,2)  ≈  E_ref(Z,4) - E_ref(Z,2)
    Elektron sayısını formülden tamamen silen adaylar burada elenir.
    """
    spec = bench.physics_checks.get("electron_count")
    if spec is None:
        return FilterResult("F11", "elektron sayısı tepkisi (∂E/∂N)", True,
                            group="DOĞRULAMA", detail="uygulanamaz")
    Z = np.asarray(spec["X"][0], float)
    N2 = np.asarray(spec["X"][1], float)          # 2
    N4 = np.asarray(spec["X"][2], float)          # 4
    d_ref = np.asarray(spec["dE"], float)
    d_pred = ex.evaluate(tree, [Z, N4]) - ex.evaluate(tree, [Z, N2])
    err = rel_rmse(d_pred, d_ref)
    tol = float(spec.get("tolerance", 5e-2))
    return FilterResult("F11", f"elektron sayısı tepkisi (ΔE, N=2→4) [{bench.key}]",
                        err <= tol, err, tol,
                        f"formülün N-farkı ile bağımsız çözücünün N-farkı: göreli hata {err:.3e}",
                        group=("ELEME" if spec.get("eliminating") else "DOĞRULAMA"))



# ----------------------------------------------------------------- jüri blokları
def jury_blocks(bench) -> List[dict]:
    """
    Jüri kısıtlarının DOĞRUSAL-ARTIK gösterimi (sabit ayarı için).

    Her kısıt, formülün sabit noktalardaki tahminleri üzerinde doğrusal bir
    operatördür (sonlu fark / fark).  Blok:

        artık_k = Σ_r coef·f(X_rows[r]) − t_k ,   ölçek_k = tol·|t_k|

    Yalnızca aramada zorunlu ('eliminating' ve 'watch') kısıtlar ve yalnızca
    eğitim+doğrulama noktaları kullanılır; SINAV kümesi kullanılmaz.
    """
    blocks: List[dict] = []

    def _cat(*vs):
        return [np.concatenate([np.asarray(v[i], float) for v in vs]) for i in range(len(vs[0]))]

    # --- F5: Hellmann-Feynman  dE/dZ = -<1/r>
    sp = bench.physics_checks.get("hellmann_feynman")
    if sp and sp.get("eliminating") and sp.get("watch"):
        X = [np.asarray(x, float) for x in sp["X"]]
        idx = int(sp.get("idx", 0)); dZ = 1e-3
        Xp, Xm = [x.copy() for x in X], [x.copy() for x in X]
        Xp[idx] = X[idx] + dZ; Xm[idx] = X[idx] - dZ
        n = len(X[idx]); rows = []
        for k in range(n):
            rows.append([(k, 1.0 / (2 * dZ)), (n + k, -1.0 / (2 * dZ))])
        target = -np.asarray(sp["inv_r"], float)
        blocks.append({"X": _cat(Xp, Xm), "rows": rows, "target": target,
                       "scale": float(sp.get("tolerance", 1e-2)) * np.abs(target), "tag": "F5"})

    # --- F6: moleküler genelleştirilmiş virial  T = -E - R·dE/dR
    sp = bench.physics_checks.get("virial")
    if sp and sp.get("eliminating") and sp.get("watch") and ("T" in sp):
        X = [np.asarray(x, float) for x in sp["X"]]
        R = X[0].copy(); h = 1e-3 * np.maximum(np.abs(R), 1e-3)
        Xp, Xm = [x.copy() for x in X], [x.copy() for x in X]
        Xp[0] = R + h; Xm[0] = R - h
        n = len(R); rows = []
        for k in range(n):
            rows.append([(k, -1.0), (n + k, -R[k] / (2 * h[k])), (2 * n + k, +R[k] / (2 * h[k]))])
        target = np.asarray(sp["T"], float)
        blocks.append({"X": _cat(X, Xp, Xm), "rows": rows, "target": target,
                       "scale": float(sp.get("tolerance", 2e-2)) * np.abs(target), "tag": "F6"})

    # --- F9: kuvvet tutarlılığı  dE/dR = -F
    sp = bench.physics_checks.get("force_consistency")
    if sp and sp.get("eliminating") and sp.get("watch") and ("F" in sp):
        X = [np.asarray(x, float) for x in sp["X"]]
        R = X[0].copy(); h = 1e-3 * np.maximum(np.abs(R), 1e-3)
        Xp, Xm = [x.copy() for x in X], [x.copy() for x in X]
        Xp[0] = R + h; Xm[0] = R - h
        n = len(R); rows = []
        for k in range(n):
            rows.append([(n + k, -1.0 / (2 * h[k])), (2 * n + k, +1.0 / (2 * h[k]))])
        target = np.asarray(sp["F"], float)
        blocks.append({"X": _cat(X, Xp, Xm), "rows": rows, "target": target,
                       "scale": float(sp.get("tolerance", 3e-2)) * np.abs(target), "tag": "F9"})

    # --- F10: örtük elektron-elektron itmesi  2E - Z·∂E/∂Z = V_ee
    sp = bench.physics_checks.get("electron_repulsion")
    if sp and sp.get("eliminating") and sp.get("watch"):
        X = [np.asarray(x, float) for x in sp["X"]]
        Z = X[0].copy(); N = X[1].copy()
        dZ = 1e-3 * np.maximum(np.abs(Z), 1e-3)
        Xp, Xm = [x.copy() for x in X], [x.copy() for x in X]
        Xp[0] = Z + dZ; Xm[0] = Z - dZ
        n = len(Z); rows = []
        for k in range(n):
            rows.append([(k, 2.0), (n + k, -Z[k] / (2 * dZ[k])), (2 * n + k, +Z[k] / (2 * dZ[k]))])
        target = np.asarray(sp["V_ee"], float)
        blocks.append({"X": _cat(X, Xp, Xm), "rows": rows, "target": target,
                       "scale": float(sp.get("tolerance", 5e-2)) * np.abs(target), "tag": "F10"})

    # --- F11: elektron sayısı tepkisi  f(Z,4) - f(Z,2) = ΔE
    sp = bench.physics_checks.get("electron_count")
    if sp and sp.get("eliminating") and sp.get("watch"):
        X = [np.asarray(x, float) for x in sp["X"]]
        n = len(X[0]); rows = []
        for k in range(n):
            rows.append([(k, 1.0), (n + k, -1.0)])
        target = np.asarray(sp["dE"], float)
        blocks.append({"X": [np.concatenate([x, x]) for x in X], "rows": rows, "target": target,
                       "scale": float(sp.get("tolerance", 5e-2)) * np.abs(target), "tag": "F11"})

    return blocks


def quantum_violation(results: List[FilterResult]) -> float:
    """
    İhlal oranı:  v = max( hata / tolerans )  — jüri kısıtları üzerinden.
    v ≤ 1  → jüri geçildi.  v büyüdükçe fizik ihlali ağırlaşır.
    """
    v = 0.0
    for f in results:
        if f.value is None or f.limit in (None, 0):
            continue
        try:
            v = max(v, float(f.value) / float(f.limit))
        except Exception:
            continue
    return v


def quantum_gate(tree: ex.Tree, bench) -> List[FilterResult]:
    """
    ARAMA DÜZEYİNDE kuantum jürisi: her bireye uygulanan hafif kuantum kısıtları.
    ELEME grubundaki bir kısıt düşerse aday ∞ ceza alır (evrim onu asla seçmez).
    """
    out: List[FilterResult] = []
    for fn in (F5_hellmann_feynman, F6_virial_action, F9_force_consistency,
               F10_electron_repulsion, F11_electron_count):
        spec = bench.physics_checks.get(
            {"F5_hellmann_feynman": "hellmann_feynman", "F6_virial_action": "virial",
             "F9_force_consistency": "force_consistency",
             "F10_electron_repulsion": "electron_repulsion",
             "F11_electron_count": "electron_count"}[fn.__name__]) or {}
        if not spec.get("watch"):      # yalnızca 'watch' işaretli kısıtlar arama kapısında
            continue
        try:
            out.append(fn(tree, bench))
        except Exception:
            continue
    return out


def F9_force_consistency(tree: ex.Tree, bench) -> FilterResult:
    """
    Mekanik Hellmann–Feynman / türev tutarlılığı ( iki yönlü ).

      (a) Aday bir ENERJİ eğrisi ise:  dE/dR ≈ F_ref  (bağımsız sonlu fark kuvveti)
      (b) Aday bir KUVVET ise:  dE_es/dR ≈ −F_aday  — burada E_es, BAŞKA bir
          benchmark'ta bağımsız keşfedilen enerji formülüdür. İki ayrı keşif,
          birbirine yalnızca fizik yasasıyla bağlıdır; eğri uydurarak sağlanamaz.
    """
    spec = bench.physics_checks.get("force_consistency")
    cross = getattr(bench, "cross_checks", None) or {}
    if spec is None and not cross.get("energy_tree"):
        return FilterResult("F9", "türev tutarlılığı (kuvvet = −dE/dR)", True,
                            group="DOĞRULAMA", detail="uygulanamaz")

    # (b) çapraz kontrol: aday kuvvet, eş keşfedilen enerjinin türeviyle karşılaştırılır
    if spec is None and cross.get("energy_tree") is not None:
        X = list(bench.val.X)
        R = np.asarray(X[0], float)
        h = 1e-3 * np.maximum(np.abs(R), 1e-3)
        Xp = [R + h] + list(X[1:]); Xm = [R - h] + list(X[1:])
        yp = ex.evaluate(cross["energy_tree"], Xp)
        ym = ex.evaluate(cross["energy_tree"], Xm)
        dEdR = (yp - ym) / (2 * h)
        yF = ex.evaluate(tree, X)
        err = rel_rmse(-dEdR, yF)
        return FilterResult("F9", "türev tutarlılığı (eş keşif: −dE/dR vs F)", err <= 3e-2,
                            err, 3e-2, f"enerji keşfi ile kuvvet keşfi uyumu: {err:.2e}",
                            group=("ELEME" if spec.get("eliminating") else "DOĞRULAMA")
                            if spec else "DOĞRULAMA")

    # (a) referans kuvvete karşı
    X = list(spec["X"])
    R = np.asarray(X[0], float)
    h = 1e-3 * np.maximum(np.abs(R), 1e-3)
    Xp = [R + h] + list(X[1:]); Xm = [R - h] + list(X[1:])
    yp = ex.evaluate(tree, Xp); ym = ex.evaluate(tree, Xm)
    dEdR = (yp - ym) / (2 * h)
    err = rel_rmse(-dEdR, np.asarray(spec["F"], float))
    tol = float(spec.get("tolerance", 3e-2))
    return FilterResult("F9", f"türev tutarlılığı (dE/dR = −F_ref) [{bench.key}]", err <= tol, err, tol,
                        f"bağımsız kuvvet referansına karşı: {err:.2e}",
                        group=("ELEME" if spec.get("eliminating") else "DOĞRULAMA"))


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
    fs.append(F9_force_consistency(ev.tree, bench))
    fs.append(F10_electron_repulsion(ev.tree, bench))
    fs.append(F11_electron_count(ev.tree, bench))
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
