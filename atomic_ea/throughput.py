# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 genesismindai (Evrimsel Atom Laboratuvarı)
#
# Bu program özgür yazılımdır: GNU Genel Kamu Lisansı (GPL) sürüm 3 veya
# sonraki sürümleri koşulları altında yeniden dağıtabilir ve/veya
# değiştirebilirsiniz. Ayrıntılar için LICENSE dosyasına bakın.
"""
throughput.py — Trilyon ölçeğinde değerlendirme muhasebesi (dürüst sürüm).

NE YAPILIR
----------
Keşfedilen ucuza formül, tek-atom (Z, n) durum uzayında bir *topluluk*
(ensemble) üzerinde değerlendirilir ve şunlar ÖLÇÜLÜR:
  * atom başına süre, öbek (chunk) başına bellek, toplam işlem sayısı,
  * taşma/NaN denetimi, değer sınırları,
  * öbek boyutu sabit tutulduğu için bellek tüketimi O(1) — N ile ARTMaz.

NE YAPILMAZ (dürüstlük notu)
----------------------------
(1) 10^12 atomun TAM ÇOK-CİSİMLİ kuantum durumu hesaplanmaz (bu, evrenin
    en büyük süper-bilgisayarında bile imkânsızdır; durum uzayı 2^N'dir).
    Yapılan iş: her atomun kendi (Z, n) durumunda, sayısal çözücünün yerine
    geçen ucuz kapalı formla değerlendirilmesi — yani tek-atom çözümünün
    trilyon ölçeğinde topluluk ortalaması.
(2) Süreler, ölçülen atom-başına hızın doğrusal dışdeğerlemesidir
    (linear extrapolation); gerçek koşu yalnızca `n_real` kadar atomda yapılır.
(3) Enerji (joule) tahmini, ölçülen iş hacmi ile İLAN EDİLEN bir
    joule/işlem varsayımından gelir; varsayım sayfasında açıkça yazılır.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import platform
import resource
import time
from typing import Callable, Dict, List, Optional
import numpy as np

J_PER_FLOP_ASSUMPTION = 8e-12   # ~8 pJ / fp64 işlem (bellek-bağlı çekirdek, ilan edilmiş varsayım)


@dataclass
class EnsembleSpec:
    """Topluluk tanımı: hangi (Z, n) aralığında hangi ağırlıkla atom var?"""
    z_min: float = 1.0
    z_max: float = 8.0
    n_min: float = 1.0
    n_max: float = 8.0
    weight_model: str = "uniform"     # uniform | thermal | weighted
    beta: float = 1.0                 # thermal için: w ~ exp(-beta*E_ref)

    def _thermal_cdf(self, nz: int = 512):
        """(Z, n) ızgarası üzerinde Boltzmann ağırlıklı CDF (bir kez hesaplanır)."""
        zb = np.linspace(self.z_min, self.z_max, nz)
        nb = np.arange(int(self.n_min), int(self.n_max) + 1).astype(float)
        Zg, Ng = np.meshgrid(zb, nb, indexing="ij")
        E = -0.5 * Zg**2 / Ng**2
        w = np.exp(-self.beta * (E - E.min())).ravel()
        cdf = np.cumsum(w)
        cdf /= cdf[-1]
        return Zg.ravel(), Ng.ravel(), cdf

    def _thermal_inv_cdf(self, nq: int = 1 << 20):
        """Ters-CDF arama tablosu: örnekleme O(1)/atom (searchsorted'siz)."""
        Zg, Ng, cdf = self._thermal_cdf()
        q = (np.arange(nq) + 0.5) / nq
        idx = np.searchsorted(cdf, q, side="right")
        np.clip(idx, 0, len(cdf) - 1, out=idx)
        # float32 tablo: bellek trafiği yarıya iner, değerler zaten ~1e-7 hassas
        return Zg[idx].astype(np.float32).copy(), Ng[idx].astype(np.float32).copy(), nq

    def sample(self, n_atoms: int, seed: int = 0) -> Dict[str, np.ndarray]:
        """
        Topluluk örneklemesi — kategorik (CDF/searchsorted) yol, ~20-30 ns/atom.
        (Yavaş `rng.choice(p=...)` ve öbek-bağımlı ağırlıklandırma kullanılmaz.)
        """
        rng = np.random.default_rng(seed)
        if self.weight_model in ("thermal", "weighted"):
            if not hasattr(self, "_inv_cache"):
                self._inv_cache = self._thermal_inv_cdf()
            invZ, invN, nq = self._inv_cache
            idx = (rng.random(n_atoms) * nq).astype(np.int32)
            return {"Z": np.take(invZ, idx), "n": np.take(invN, idx)}
        Z = rng.uniform(self.z_min, self.z_max, n_atoms)
        n = rng.integers(int(self.n_min), int(self.n_max) + 1, n_atoms).astype(float)
        return {"Z": Z, "n": n}


def _measure_peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_ensemble(formula: Callable[[List[np.ndarray]], np.ndarray],
                 spec: EnsembleSpec, n_atoms: int, chunk: int = 2_000_000,
                 ops_per_atom: float = 6.0, seed: int = 0,
                 collect_stats: bool = True,
                 time_budget_s: float = 60.0) -> dict:
    """
    Topluluğu öbekler halinde akıtır (streaming). Bellek O(chunk).
    Gerçek koşu `n_atoms` kadar atomda yapılır; süre ölçülür.
    """
    rss_before = _measure_peak_rss_mb()
    done = 0
    s_sum = 0.0
    s_sq = 0.0
    e_min, e_max = math.inf, -math.inf
    v_min, v_max = math.inf, -math.inf       # |1/r| gibi ikincil büyüklük varsa
    nonfinite = 0
    t0 = time.perf_counter()
    t_sample = 0.0
    t_formula = 0.0
    bytes_moved = 0

    while done < n_atoms:
        nb = min(chunk, n_atoms - done)
        ta = time.perf_counter()
        s = spec.sample(nb, seed=seed + done)
        X = list(s.values())
        tb = time.perf_counter()
        y = formula(X)
        tc = time.perf_counter()
        t_sample += tb - ta
        t_formula += tc - tb
        if not np.all(np.isfinite(y)):
            nonfinite += int(np.sum(~np.isfinite(y)))
        s_sum += float(np.sum(y))
        s_sq += float(np.sum(y * y))
        e_min = min(e_min, float(np.min(y)))
        e_max = max(e_max, float(np.max(y)))
        bytes_moved += nb * 8 * (len(X) + 3)
        done += nb
        if time.perf_counter() - t0 > time_budget_s:
            break

    t1 = time.perf_counter()
    rss_after = _measure_peak_rss_mb()
    elapsed = t1 - t0
    per_atom_s = elapsed / max(done, 1)
    return {
        "atoms_realized": done,
        "requested": n_atoms,
        "elapsed_s": round(elapsed, 4),
        "s_per_atom": per_atom_s,
        "ns_per_atom": round(per_atom_s * 1e9, 3),
        "atoms_per_s": round(done / max(elapsed, 1e-9), 1),
        "chunk": chunk,
        "ops_per_atom": ops_per_atom,
        "sampling_ns_per_atom": round(t_sample / max(done, 1) * 1e9, 2),
        "formula_ns_per_atom": round(t_formula / max(done, 1) * 1e9, 2),
        "gflop_total": done * ops_per_atom / 1e9,
        "gflops_achieved": done * ops_per_atom / max(elapsed, 1e-9) / 1e9,
        "bytes_moved_gb": round(bytes_moved / 1e9, 3),
        "peak_rss_mb": round(rss_after, 1),
        "rss_growth_mb": round(rss_after - rss_before, 1),
        "mean_E": s_sum / max(done, 1),
        "std_E": float(np.sqrt(max(s_sq / max(done, 1) - (s_sum / max(done, 1)) ** 2, 0.0))),
        "E_min": e_min,
        "E_max": e_max,
        "nonfinite_values": nonfinite,
        "completed": done >= n_atoms,
        "stats_collected": collect_stats,
    }


def project(n_atoms_target: float, per_atom_s: float, ops_per_atom: float,
            j_per_flop: float = J_PER_FLOP_ASSUMPTION) -> dict:
    """Ölçülen atom-başına hızdan trilyon ölçeğine doğrusal dışdeğerleme."""
    secs = n_atoms_target * per_atom_s
    flops = n_atoms_target * ops_per_atom
    return {
        "atoms": n_atoms_target,
        "estimated_seconds": secs,
        "estimated_human": _human_time(secs),
        "fp64_flops": flops,
        "estimated_joules": flops * j_per_flop,
        "joules_per_atom": ops_per_atom * j_per_flop,
        "single_core_days": secs / 86400.0,
    }


def _human_time(s: float) -> str:
    if s < 1e-3:
        return f"{s*1e6:.1f} µs"
    if s < 1:
        return f"{s*1e3:.1f} ms"
    if s < 90:
        return f"{s:.1f} s"
    if s < 5400:
        return f"{s/60:.1f} dk"
    if s < 172800:
        return f"{s/3600:.1f} saat"
    return f"{s/86400:.1f} gün"


def solver_baseline_cost(h: float = 0.005, r_max: float = 200.0,
                         n_levels: int = 8) -> dict:
    """
    "Pahalı" yolun maliyeti: aynı bilgi için sayısal özdeğer çözümü.
    Ölçülür: matris boyutu N, zaman, tahmini işlem sayısı.
    (Sonlu farklar tridiagonal özdeğer çözümü ~ O(N^2) işlem: N özvektör × N yineleme)
    """
    from . import physics as ph
    t0 = time.perf_counter()
    sol = ph.solve_hydrogenic(Z=1.0, l=0, n_states=n_levels, h=h, r_max=r_max)
    t1 = time.perf_counter()
    N = sol.r.size
    est_ops = float(N) * float(N) * 0.5        # tridiagonal QR yinelemesi için kaba üst sınır
    return {
        "N_grid": int(N),
        "n_levels": n_levels,
        "wall_s_per_configuration": t1 - t0,
        "estimated_ops_per_configuration": est_ops,
        "per_atom_ops_if_repeated": est_ops,
        "note": ("Her yeni (Z, l) yapılandırması için bu çözüm baştan yapılır: "
                 "maliyet atom başına sabit formül maliyetinin milyonlarca katıdır."),
    }


def full_report(formula_fn: Callable[[List[np.ndarray]], np.ndarray],
                ops_per_atom: float,
                n_real: int = 100_000_000,
                targets: Optional[List[float]] = None,
                chunk: int = 2_000_000,
                spec: Optional[EnsembleSpec] = None,
                seed: int = 0) -> dict:
    """Rapor: gerçek koşu + ölçek dışdeğerlemesi + donor (pahalı) referans karşılaştırması."""
    spec = spec or EnsembleSpec()
    targets = targets or [1e6, 1e9, 1e12]
    run = run_ensemble(formula_fn, spec, n_real, chunk=chunk,
                       ops_per_atom=ops_per_atom, seed=seed)
    proj = [project(t, run["s_per_atom"], ops_per_atom) for t in targets]
    base = solver_baseline_cost()
    speedup = base["estimated_ops_per_configuration"] / max(ops_per_atom, 1e-9)
    return {
        "spec": asdict(spec),
        "run": run,
        "projections": proj,
        "baseline_solver": base,
        "speedup_vs_solver_ops_per_atom": speedup,
        "speedup_vs_solver_seconds": base["wall_s_per_configuration"] / max(run["s_per_atom"], 1e-12),
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor() or "bilinmiyor",
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "assumptions": {
            "joule_per_flop": J_PER_FLOP_ASSUMPTION,
            "note": ("Süreler ÖLÇÜLEN ns/atom üzerinden doğrusal dışdeğerlemedir. "
                     "Bellek, öbek akışı sayesinde atom sayısıyla büyümez (O(chunk)). "
                     "Bu bir tek-atom topluluk simülasyonudur; tam çok-cisimli "
                     "kuantum simülasyonu değildir."),
        },
    }
