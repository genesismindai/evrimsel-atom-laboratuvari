"""
benchmarks.py — Referans veritabanı ve problem tanımları.

Tüm referans veriler SAYISAL ÇÖZÜCÜDEN (physics.py) gelir ve diske özet
(checksum) ile birlikte önbelleklenir. Veri üretimi kapalı form kullanmaz;
kapalı formlar yalnızca *keşif sonrası karşılaştırma* için raporlarda geçer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from typing import Dict, List, Optional, Sequence
import numpy as np

from . import physics as ph

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@dataclass
class Dataset:
    X: List[np.ndarray]
    y: np.ndarray
    name: str = ""
    extra: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class Benchmark:
    key: str
    title: str
    purpose: str
    var_names: List[str]
    train: Dataset
    val: Dataset
    test: Dataset
    tol_rel: float = 1e-3
    noise_rel: float = 1e-6      # referans verinin ölçüm/hassasiyet tabanı
    cost_budget: float = 12.0
    tier: str = "cheap"
    scaling_specs: List[dict] = field(default_factory=list)
    physics_checks: Dict[str, dict] = field(default_factory=dict)
    null_barrier: Optional[float] = None
    notes: Dict[str, float] = field(default_factory=dict)
    reference_note: str = ""
    probe_X: List[np.ndarray] = field(default_factory=list)


# ------------------------------------------------------------------ veri üretimi
def _grid(rows: List[Sequence[float]], cols: List[Sequence[float]]):
    """(rows x cols) çapraz çarpımı -> X0, X1 düzleştirilmiş diziler."""
    a = np.array([[r for c in cols] for r in rows], dtype=float)
    b = np.array([[c for c in cols] for r in rows], dtype=float)
    return a.ravel(), b.ravel()


def build_reference(h: float = 0.005, r_max: float = 200.0, verbose: bool = True) -> dict:
    """
    Sayısal referans veritabanını kurar ve diske yazar (npz + sha256).

    İçerik:
      * E(Z, n)      hidrojenik enerji (l=0 çözümünden, tüm l için dejenerasyon
                     ayrıca l=1,2 ile sayısal olarak doğrulanır)
      * inv_r(Z, n)  <1/r> beklenti değeri
      * r_exp(Z, n)  <r>
      * yukawa(Z,a)  perdeli Coulomb taban durumu enerjisi (kapalı formu YOK)
      * doğrulama ölçütleri (norm, eylem artığı, virial, ızgara yakınsaması)
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "reference.npz")
    Zs = list(range(1, 9))
    Ns = list(range(1, 9))
    YZ = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    YA = [round(x, 3) for x in np.linspace(0.0, 0.60, 25)]

    E = np.zeros((len(Zs), len(Ns)))
    IR = np.zeros_like(E)
    RE = np.zeros_like(E)
    act = np.zeros_like(E)
    vir = np.zeros_like(E)
    deg_check = []

    for i, Z in enumerate(Zs):
        sol = ph.solve_hydrogenic(Z=Z, l=0, n_states=8, h=h, r_max=r_max)
        for j, n in enumerate(Ns):
            lv = sol.level(n)
            E[i, j] = lv["E"]
            IR[i, j] = lv["inv_r"]
            RE[i, j] = lv["r"]
            act[i, j] = lv["action_residual"]
            vir[i, j] = lv["virial_ratio"]
        # l-dejenerasyonunun SAYISAL doğrulaması (en düşük 3 l için)
        if Z in (1.0, 2.0):
            for l in (1, 2):
                sl = ph.solve_hydrogenic(Z=Z, l=l, n_states=3, h=h, r_max=r_max)
                for k, n in enumerate([l + 1 + q for q in range(3)]):
                    deg_check.append(abs(sl.E[k] - E[i, Ns.index(n)]))

    # Yukawa (perdeli Coulomb): taban durumu
    Y = np.zeros((len(YZ), len(YA)))
    for i, Z in enumerate(YZ):
        for j, a in enumerate(YA):
            s = ph.solve_yukawa(Z=Z, alpha=a, l=0, n_states=1, h=0.01, r_max=160.0)
            Y[i, j] = float(s.E[0])

    # Ölçek yasasının VERİDEN doğrulanması: E(2Z,n)/E(Z,n) = 4 ?
    scale_err = float(np.max(np.abs(E[1, :] / E[0, :] - 4.0)))
    ir_scale_err = float(np.max(np.abs(IR[1, :] / IR[0, :] - 2.0)))

    ric = ph.richardson_energy(1.0, 0, 0, h=0.02)

    np.savez_compressed(
        path, Zs=np.array(Zs), Ns=np.array(Ns), E=E, IR=IR, RE=RE,
        action=act, virial=vir, YZ=np.array(YZ), YA=np.array(YA), Y=Y,
    )
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    meta = {
        "checksum_sha256": digest, "h": h, "r_max": r_max,
        "max|E(2Z)/E(Z)-4|": scale_err, "max|<1/r> ölçek-2|": ir_scale_err,
        "max|l-dejenerasyon hatası|": float(max(deg_check)) if deg_check else None,
        "max|eylem artığı|": float(np.max(np.abs(act))),
        "max rel|eylem artığı|": float(np.max(np.abs(act) / np.abs(E))),
        "virial aralığı": [float(np.min(vir)), float(np.max(vir))],
        "richardson": ric,
        "E(1,1) sayısal": float(E[0, 0]),
        "E(1,1) kapalı form (yalnız karşılaştırma)": -0.5,
    }
    with open(os.path.join(DATA_DIR, "reference_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"[veri] referans kuruldu -> {path}")
        print(f"       sha256={digest[:16]}…  ölçek sapması {scale_err:.2e}, "
              f"|<1/r> ölçek| {ir_scale_err:.2e}")
        print(f"       l-dejenerasyon max hata {meta['max|l-dejenerasyon hatası|']:.2e}, "
              f"|eylem artığı| max {meta['max|eylem artığı|']:.2e}")
    return meta


def load_reference(verbose: bool = False, h: float = 0.005, r_max: float = 200.0) -> dict:
    path = os.path.join(DATA_DIR, "reference.npz")
    if not os.path.exists(path):
        if verbose:
            print("[veri] önbellek yok, sayısal çözücüden üretiliyor…")
        build_reference(h=h, r_max=r_max, verbose=verbose)
    d = np.load(path)
    return {k: d[k] for k in d.files}


# ------------------------------------------------------------------ benchmark'lar
def _energy_bench(ref: dict) -> Benchmark:
    """B1: E(Z,n) — hidrojenik enerji yasasının kör yeniden keşfi."""
    Zs, Ns, E, IR = ref["Zs"], ref["Ns"], ref["E"], ref["IR"]

    def mk(zlist, nlist):
        Z, N = _grid(zlist, nlist)
        idx_z = [list(Zs).index(z) for z in Z]
        idx_n = [list(Ns).index(n) for n in N]
        y = E[idx_z, idx_n]
        ir = IR[idx_z, idx_n]
        return Dataset(X=[Z, N], y=y, extra={"inv_r": ir, "Z": Z, "n": N})

    train = mk([1, 2, 3], [1, 2, 3])
    val = mk([1, 2, 3, 4], [1, 2, 3, 4, 5, 6])
    mask = ~np.isin(np.round(val.X[0], 6), [1, 2, 3]) | ~np.isin(val.X[1], [1, 2, 3])
    val = Dataset(X=[val.X[0][mask], val.X[1][mask]], y=val.y[mask],
                  extra={k: v[mask] for k, v in val.extra.items()})
    tz = np.concatenate([np.repeat(np.arange(5, 9), 8), np.repeat(np.arange(1, 5), 2)])
    tn = np.concatenate([np.tile(np.arange(1, 9), 4), np.tile([7, 8], 4)])
    test = mk(list(tz.astype(float)), list(tn.astype(float)))

    # ölçek simetrisi: E(λZ, n) = λ² E(Z,n)
    Z0, N0 = val.X[0], val.X[1]
    lam = np.array([0.5, 1.5, 2.0])
    specs = [{"var": 0, "power": 2.0, "X0": [Z0, N0], "lambdas": lam}]

    return Benchmark(
        key="enerji", title="Hidrojenik enerji yasası  E(Z, n)",
        purpose=("Sayısal Schrödinger çözücüsünün ürettiği enerji tablosundan, "
                 "sadece toplama/çarpma/bölme içeren en ucuz kapalı formu keşfetmek. "
                 "Eğitim bölgesi Z,n ≤ 3; sınav bölgesi tamamen dışarıda."),
        var_names=["Z", "n"], train=train, val=val, test=test,
        tol_rel=1e-3, noise_rel=2e-4, cost_budget=14.0, tier="cheap", scaling_specs=specs,
        physics_checks={
            "hellmann_feynman": {"X": [val.X[0], val.X[1]], "inv_r": val.extra["inv_r"]},
            "virial": {"X": [val.X[0], val.X[1]], "inv_r": val.extra["inv_r"]},
        },
        notes={},
        probe_X=[np.concatenate([val.X[0], test.X[0]]), np.concatenate([val.X[1], test.X[1]])],
        reference_note="Referans: sonlu farklar özdeğer çözümü, 1e-5 mertebesi yakınsama (Richarson doğrulamalı).",
    )


def _invr_bench(ref: dict) -> Benchmark:
    """B2: <1/r>(Z,n) — Hellmann-Feynman eşleşmesi için ikinci bağımsız yasa."""
    Zs, Ns, IR = ref["Zs"], ref["Ns"], ref["IR"]

    def mk(zlist, nlist):
        Z, N = _grid(zlist, nlist)
        idx_z = [list(Zs).index(z) for z in Z]
        idx_n = [list(Ns).index(n) for n in N]
        return Dataset(X=[Z, N], y=IR[idx_z, idx_n].copy(), extra={})

    train = mk([1, 2, 3], [1, 2, 3])
    val = mk([1, 2, 3, 4], [1, 2, 3, 4, 5, 6])
    mask = ~np.isin(np.round(val.X[0], 6), [1, 2, 3]) | ~np.isin(val.X[1], [1, 2, 3])
    val = Dataset(X=[val.X[0][mask], val.X[1][mask]], y=val.y[mask], extra={})
    test = mk([5, 6, 7, 8] + [1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7, 8] + [7, 8, 7, 8])

    Z0, N0 = val.X[0], val.X[1]
    return Benchmark(
        key="ters_yaricap", title="Beklenen ters yarıçap  ⟨1/r⟩(Z, n)",
        purpose=("İkinci bağımsız yasa. Enerji formülüyle birlikte "
                 "Hellmann-Feynman ve virial kısıtlarını oluşturur."),
        var_names=["Z", "n"], train=train, val=val, test=test,
        tol_rel=1e-3, noise_rel=5e-4, cost_budget=14.0, tier="cheap",
        scaling_specs=[{"var": 0, "power": 1.0, "X0": [Z0, N0], "lambdas": np.array([0.5, 1.5, 2.0])}],
        physics_checks={}, notes={},
        probe_X=[np.concatenate([val.X[0], test.X[0]]), np.concatenate([val.X[1], test.X[1]])],
        reference_note="Referans: sayısal özfonksiyondan Simpson integrali ile <1/r>.",
    )


def _pot_bench(ref: dict) -> Benchmark:
    """
    B3: Etkin potansiyel operatörünün sıkıştırılması.

    Sayısal çözücü, N=40001 elemanlı köşegen matrisi kurar; bu benchmark
    o köşegenin (yani pahalı ızgara operatörünün) 4-5 işlemlik bir kapalı
    formla değiştirilip değiştirilemeyeceğini ölçer. Eğitim örneklemi 240
    noktadır; sınav ise örneklenmeyen tüm ızgaradır (ızgara dışlama testi).
    """
    rng = np.random.default_rng(20260922)
    r_lo, r_hi = 0.05, 12.0
    Zv = np.array([1.0, 1.0, 2.0, 3.0, 6.0])
    lv = np.array([0, 1, 2, 3])
    Rs, Zs_, Ls = [], [], []
    for Z in Zv:
        for l in lv:
            rr = np.exp(rng.uniform(np.log(r_lo), np.log(r_hi), 12))
            Rs.append(rr); Zs_.append(np.full(12, Z)); Ls.append(np.full(12, l))
    R = np.concatenate(Rs); Zg = np.concatenate(Zs_); Lg = np.concatenate(Ls)
    y = -Zg / R + Lg * (Lg + 1.0) / (2.0 * R**2)

    r_test = np.linspace(r_lo, r_hi, 6000)
    Zt = np.tile(np.array([1.0, 2.0, 4.0]), len(r_test))
    Lt = np.tile(np.array([0, 1, 2]), len(r_test))
    Rt = np.repeat(r_test, 3)
    yt = -Zt / Rt + Lt * (Lt + 1.0) / (2.0 * Rt**2)

    tr = Dataset(X=[R, Lg, Zg], y=y)
    va = Dataset(X=[Rt[:6000], Lt[:6000], Zt[:6000]], y=yt[:6000])
    te = Dataset(X=[Rt, Lt, Zt], y=yt)
    return Benchmark(
        key="etkin_potansiyel",
        title="Etkin potansiyel  V_eff(r, l, Z) = −Z/r + l(l+1)/(2r²)",
        purpose=("Pahalı ızgara operatörünün (40001 köşegen elemanı) ucuz kapalı "
                 "forma sıkıştırılması. Örneklenmemiş ızgaranın tamamında sınanır."),
        var_names=["r", "l", "Z"], train=tr, val=va, test=te,
        tol_rel=2e-3, noise_rel=1e-9, cost_budget=12.0, tier="cheap",
        scaling_specs=[],
        physics_checks={}, notes={},
        probe_X=[np.linspace(0.05, 60.0, 800), np.tile([0, 1, 2], 267)[:800],
                 np.tile([1.0, 2.0, 3.0], 267)[:800]],
        reference_note="Referans: çözücünün kurduğu tam ızgara köşegeni (k=0: yalnız -Z/r).",
    )


def _yukawa_bench(ref: dict) -> Benchmark:
    """B4: Perdeli Coulomb (Yukawa) — KAPALI FORMU OLMAYAN problem için ucuz vekil."""
    YZ, YA, Y = ref["YZ"], ref["YA"], ref["Y"]

    def mk(zidx, aidx):
        Z, A = _grid([YZ[i] for i in zidx], [YA[j] for j in aidx])
        iz = [list(YZ).index(z) for z in Z]
        ja = [list(YA).index(a) for a in A]
        return Dataset(X=[Z, A], y=Y[iz, ja].copy(), extra={})

    train = mk([0, 1, 2, 3], range(0, 11))        # α ≤ 0.25
    val = mk([0, 1, 2, 3, 4, 5], range(0, 18))    # α ≤ 0.425
    m = (val.X[0] > 4.0) | (val.X[1] > 0.25)
    val = Dataset(X=[val.X[0][m], val.X[1][m]], y=val.y[m], extra={})
    test = mk(list(range(6)), range(10, 25))      # α ∈ [0.25, 0.60]: ekstrapolasyon
    Z0, A0 = val.X[0], val.X[1]
    return Benchmark(
        key="yukawa", title="Perdeli Coulomb (Yukawa) taban enerjisi için ucuz vekil",
        purpose=("Bu problemde KAPALI FORM YOKTUR. Amaç: sayısal çözücünün pahalı "
                 "sonucunu, beyan edilen geçerlik bölgesinde taklit eden en ucuz "
                 "formülü bulmak (keşif değil, sıkıştırma — dürüstçe etiketlenir)."),
        var_names=["Z", "α"], train=train, val=val, test=test,
        tol_rel=2e-2, noise_rel=1e-3, cost_budget=20.0, tier="cheap",
        scaling_specs=[],
        physics_checks={}, notes={},
        probe_X=[np.concatenate([val.X[0], test.X[0]]), np.concatenate([val.X[1], test.X[1]])],
        reference_note="Referans: Yukawa potansiyelinde sonlu farklar taban durumu (α=0 sınırında hidrojenik limite yakınsar).",
    )


def build_all(verbose: bool = True) -> List[Benchmark]:
    ref = load_reference(verbose=verbose)
    benches = [_energy_bench(ref), _invr_bench(ref), _pot_bench(ref), _yukawa_bench(ref)]
    meta_path = os.path.join(DATA_DIR, "reference_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    for b in benches:
        b.notes["action_residual"] = meta.get("max rel|eylem artığı|", float("nan"))
        b.notes["checksum"] = meta.get("checksum_sha256", "")
    return benches


# ------------------------------------------------------------------ etiket karıştırma (null test)
def shuffle_targets(bench: Benchmark, seed: int = 0) -> Benchmark:
    """Null testi için etiketleri karıştırılmış birebir kopya üretir."""
    def sh(ds: Dataset) -> Dataset:
        rng = np.random.default_rng(seed)
        y = ds.y.copy()
        rng.shuffle(y)
        return Dataset(X=[x.copy() for x in ds.X], y=y, name=ds.name + "_null")
    return Benchmark(
        key=bench.key + "_null", title=bench.title + " [etiketler karıştırıldı]",
        purpose="null test", var_names=bench.var_names,
        train=sh(bench.train), val=sh(bench.val), test=sh(bench.test),
        tol_rel=bench.tol_rel, cost_budget=bench.cost_budget, tier=bench.tier,
        scaling_specs=[], physics_checks={}, notes={},
        probe_X=[x.copy() for x in bench.probe_X],
    )
