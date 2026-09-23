"""
runner.py — Koşu orkestrasyonu: kalibrasyon, evrim, ölçeklendirme, rapor, arşiv.

Her şey diske kaydedilir (kullanıcı şartı: "tüm ilerlemeler kaydedilsin"):
  data/index.json          koşu dizini (kalıcı kayıt defteri)
  data/runs/<id>.json      koşunun tam sonucu (tohum, yapılandırma, formüller)
  data/progress.json       CANLI ilerleme (web arayüzü bunu okur)
  data/null_barriers.json  numeroloji bariyerleri (etiket karıştırma kalibrasyonu)
  reports/<id>.md|.html    insan-okur rapor (yayın artefaktı)
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime as _dt
import json
import os
import threading
import time
from typing import Callable, Dict, List, Optional
import numpy as np

from atomic_ea import benchmarks as bm
from atomic_ea import evolution as ev
from atomic_ea import expression as ex
from atomic_ea import filters as fl
from atomic_ea import physics as ph
from atomic_ea import throughput as tp

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
RUNS = os.path.join(DATA, "runs")
REPORTS = os.path.join(ROOT, "reports")
for d in (DATA, RUNS, REPORTS):
    os.makedirs(d, exist_ok=True)

PRESETS = {
    "hizli": dict(pop_size=140, generations=35, islands=3, strict_interval=5),
    "standart": dict(pop_size=300, generations=80, islands=3, strict_interval=5),
    "derin": dict(pop_size=480, generations=160, islands=4, strict_interval=4),
}


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def compile_formula(src: str, var_names: List[str]) -> Callable[[List[np.ndarray]], np.ndarray]:
    """Üretilen Python kaynağını numpy-vektörleştirilmiş çağrılabilire çevirir."""
    env = {"np": np}
    def f(X):
        loc = {name: X[i] for i, name in enumerate(var_names)}
        return eval(src, env, loc)
    return f


class Engine:
    """İlerleme durumunu tutan, web sunucusunun okuduğu motor."""

    def __init__(self):
        self.lock = threading.RLock()
        self.state = "bekliyor"        # bekliyor | veri | kalibrasyon | evrim | olcek | rapor | tamam | durdu | hata
        self.run_id: Optional[str] = None
        self.preset: Optional[str] = None
        self.seed: Optional[int] = None
        self.started: Optional[str] = None
        self.finished: Optional[str] = None
        self.logs: List[str] = []
        self.results: Dict[str, dict] = {}
        self.bench_status: List[dict] = []
        self.subscribers: List["object"] = []
        self.stop_flag = False
        self.throughput: Optional[dict] = None
        self.self_test: Optional[dict] = None
        self.error: Optional[str] = None
        self.best_trees: Dict[str, object] = {}      # eş keşif çapraz kontrolü için (yalnız bellekte)

    # ---------- yayın
    def log(self, msg: str) -> None:
        line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
        with self.lock:
            self.logs.append(line)
            self.logs = self.logs[-500:]
        self._publish({"type": "log", "msg": line})

    def _publish(self, obj: dict) -> None:
        for q in list(self.subscribers):
            try:
                q.put_nowait(obj)
            except Exception:
                pass

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "state": self.state,
                "run_id": self.run_id,
                "preset": self.preset,
                "seed": self.seed,
                "started": self.started,
                "finished": self.finished,
                "logs": self.logs[-120:],
                "bench_status": self.bench_status,
                "results": {k: summarize_result(v) for k, v in self.results.items()},
                "throughput": self.throughput,
                "self_test": self.self_test,
                "error": self.error,
            }


# ------------------------------------------------------------------ yardımcılar
def summarize_result(res: dict) -> dict:
    """Tam sonuçtan arayüz için özet."""
    champ = res.get("champion") or {}
    return {
        "bench_key": res.get("bench_key"),
        "bench_title": res.get("bench_title"),
        "var_names": res.get("var_names"),
        "generations_run": res.get("generations_run"),
        "evaluations": res.get("evaluations"),
        "wall_seconds": res.get("wall_seconds"),
        "null_barrier": res.get("null_barrier"),
        "physics_seeded": res.get("physics_seeded"),
        "seeds_used": res.get("seeds_used", [])[:6],
        "history": res.get("history", [])[-400:],
        "champion": champ,
        "hall_of_fame": res.get("hall_of_fame", [])[:8],
        "known_limits": res.get("known_limits"),
    }


def _save_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def load_index() -> List[dict]:
    p = os.path.join(DATA, "index.json")
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return []
    return []


def load_run(run_id: str) -> Optional[dict]:
    p = os.path.join(RUNS, f"{run_id}.json")
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return None
    return None


# ------------------------------------------------------------------ null bariyerleri
def ensure_null_barriers(benches: List[bm.Benchmark], cfg: ev.EAConfig,
                         engine: Optional[Engine] = None, force: bool = False) -> Dict[str, float]:
    path = os.path.join(DATA, "null_barriers.json")
    cache = {}
    if os.path.exists(path) and not force:
        try:
            cache = json.load(open(path, encoding="utf-8"))
        except Exception:
            cache = {}
    out = {}
    for b in benches:
        key = f"{b.key}|pop{cfg.pop_size}|gen{max(10, cfg.generations//4)}|seed{cfg.seed}"
        if key in cache and not force:
            out[b.key] = float(cache[key])
            b.null_barrier = float(cache[key])
            if engine:
                engine.log(f"numeroloji bariyeri (önbellek): {b.key} -> {cache[key]:.4g}")
            continue
        if engine:
            engine.log(f"numeroloji bariyeri kalibrasyonu: {b.key} (etiketler karıştırılmış veri)")
        val = ev.calibrate_null_barrier(b, cfg, progress=None)
        cache[key] = val
        out[b.key] = val
        b.null_barrier = val
        _save_json(path, cache)
        if engine:
            engine.log(f"  bariyer {b.key} = {val:.4g}  (karışık veride ulaşılan en iyi doğrulama hatası)")
    return out


# ------------------------------------------------------------------ bilinen limitler
def known_limit_check(key: str, champ: dict, bench, n_probe: int = 400) -> Optional[dict]:
    """
    Keşfedilen formun BİLİNEN analitik limitlerle karşılaştırılması.

    Bu adım SEÇİMDE KULLANILMAZ; yalnızca teşhis amaçlıdır (keşif bittikten sonra
    bağımsız bir referansla karşılaştırma). Böylece "doğru bilinen sonuca yakınsadı mı?"
    sorusu, o sonuç aramada kullanılmadan yanıtlanır.
    """
    import numpy as _np
    if not champ:
        return None
    fn = compile_formula(champ["python"], bench.var_names)
    out = {"note": "Yalnızca keşif sonrası teşhis; arama/seçim bu bilgiyi kullanmaz."}

    if key == "enerji":
        rng = _np.random.default_rng(5)
        Z = rng.uniform(1, 9, n_probe); n = rng.integers(1, 9, n_probe).astype(float)
        y = fn([Z, n]); y_ref = -0.5 * Z**2 / n**2
        rel = _np.abs(y - y_ref) / _np.abs(y_ref)
        out.update({"limit": "E = −Z²/(2n²) Hartree (Rydberg)",
                    "max_rel_deviation": float(rel.max()),
                    "mean_rel_deviation": float(rel.mean()),
                    "champion_constants": champ["formula"]})
    elif key == "ters_yaricap":
        rng = _np.random.default_rng(6)
        Z = rng.uniform(1, 9, n_probe); n = rng.integers(1, 9, n_probe).astype(float)
        y = fn([Z, n]); y_ref = Z / n**2
        rel = _np.abs(y - y_ref) / y_ref
        out.update({"limit": "⟨1/r⟩ = Z/n² (a₀⁻¹)",
                    "max_rel_deviation": float(rel.max()),
                    "mean_rel_deviation": float(rel.mean())})
    elif key == "etkin_potansiyel":
        rng = _np.random.default_rng(7)
        r = _np.exp(rng.uniform(_np.log(0.05), _np.log(12.0), n_probe))
        l = rng.integers(0, 4, n_probe).astype(float); Z = rng.uniform(1, 6, n_probe)
        y = fn([r, l, Z]); y_ref = -Z / r + l * (l + 1) / (2 * r**2)
        rel = _np.abs(y - y_ref) / _np.maximum(_np.abs(y_ref), 1e-12)
        out.update({"limit": "V_eff = −Z/r + l(l+1)/(2r²) (tam ifade)",
                    "max_rel_deviation": float(rel.max()),
                    "mean_rel_deviation": float(rel.mean())})
    elif key == "yukawa":
        # Pertürbasyon serisi katsayıları: BAĞIMSIZ olarak yeniden hesaplanır
        alphas = _np.array([0.004, 0.008, 0.012, 0.016, 0.020])
        E_solver = []
        for a in alphas:
            sol = ph.solve_yukawa(Z=1.0, alpha=float(a), l=0, n_states=1, h=0.01, r_max=160.0)
            E_solver.append(float(sol.E[0]))
        E_solver = _np.array(E_solver)
        A = _np.vstack([_np.ones_like(alphas), alphas, alphas**2]).T
        coef_solver, *_ = _np.linalg.lstsq(A, E_solver, rcond=None)

        E_formula = fn([_np.ones_like(alphas), alphas])
        coef_formula, *_ = _np.linalg.lstsq(A, E_formula, rcond=None)
        names = ["E(α→0)", "dE/dα", "d²E/dα²"]
        table = []
        for nm, a_, b_ in zip(names, coef_solver, coef_formula):
            table.append({"term": nm, "solver": float(a_), "formula": float(b_),
                          "rel_diff": float(abs(a_ - b_) / max(abs(a_), 1e-12))})
        out.update({
            "limit": "küçük α için pertürbasyon serisi E(Z=1,α) = E₀ + c₁α + c₂α² (katsayılar bağımsız çözücüden)",
            "perturbation_table": table,
            "max_coefficient_rel_diff": max(t["rel_diff"] for t in table),
            "note": ("Yalnızca teşhis: katsayılar keşiften SONRA, bağımsız sayısal çözücüyle "
                     "yeniden hesaplanıp karşılaştırılmıştır; aramada kullanılmamıştır."),
        })
    return out


# ------------------------------------------------------------------ ana koşu
def run_all(engine: Engine, preset: str = "hizli", seed: int = 1,
            do_null: bool = True, do_throughput: bool = True,
            n_real_atoms: int = 100_000_000,
            targets: Optional[List[float]] = None) -> dict:
    engine.stop_flag = False
    cfg = ev.EAConfig(**PRESETS[preset], seed=seed)
    run_id = _dt.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{preset}-s{seed}"
    engine.run_id, engine.preset, engine.seed = run_id, preset, seed
    engine.started, engine.finished = now_iso(), None
    engine.results, engine.error, engine.throughput = {}, None, None
    t_start = time.time()
    record = {
        "run_id": run_id, "preset": preset, "seed": seed, "started": engine.started,
        "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
        "benchmarks": {}, "throughput": None, "self_test": None,
    }

    try:
        # ---- 0) çözücü öz testi (veri kaynağının güvenilirliği)
        engine.state = "veri"
        engine.log("sayısal referans çözücü öz testi başlıyor (bağımsız doğrulama)")
        st = ph.solver_self_test(verbose=False)
        meta_path = os.path.join(DATA, "reference_meta.json")
        meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
        st["meta"] = {k: meta.get(k) for k in
                      ("checksum_sha256", "max|E(2Z)/E(Z)-4|", "max|l-dejenerasyon hatası|",
                       "max rel|eylem artığı|", "richardson", "virial aralığı")}
        engine.self_test = st
        record["self_test"] = st
        engine.log(f"  |norm-1|={st['norm_residual']:.2e}  |eylem artığı|={st['action_residual']:.2e}  "
                   f"virial={st['virial_ratio']:.6f}  ızgara farkı={st['grid_delta']:.2e}")
        engine.log(f"  referans sha256={str(meta.get('checksum_sha256'))[:16]}…")

        benches = bm.build_all(verbose=False)
        engine.bench_status = [{"key": b.key, "title": b.title, "status": "bekliyor",
                                "purpose": b.purpose, "var_names": b.var_names}
                               for b in benches]
        engine._publish({"type": "bench_status", "data": engine.bench_status})

        # ---- 1) numeroloji bariyerleri
        engine.state = "kalibrasyon"
        if do_null:
            ensure_null_barriers(benches, cfg, engine)
        else:
            engine.log("numeroloji bariyeri kalibrasyonu atlandı (uyarı: F4 filtresi pasif)")

        # ---- 2) her benchmark için evrim
        for idx, b in enumerate(benches):
            if engine.stop_flag:
                engine.state = "durdu"
                break
            engine.state = "evrim"
            engine.bench_status[idx]["status"] = "çalışıyor"
            engine.log(f"evrim başlıyor: {b.key} — {b.title}")
            if b.null_barrier:
                engine.log(f"  numeroloji bariyeri: val hatası ≤ {b.null_barrier/cfg.null_factor:.3g} olmalı")

            def progress(d, idx=idx, b=b):
                if d.get("type") == "gen":
                    engine.bench_status[idx]["gen"] = d.get("gen")
                    engine.bench_status[idx]["best_score"] = d.get("best_score")
                    engine.bench_status[idx]["best_val"] = d.get("best_val_loss")
                    engine.bench_status[idx]["formula"] = d.get("formula")
                    engine.bench_status[idx]["cost"] = d.get("cost")
                    engine.bench_status[idx]["archive"] = d.get("archive")
                    engine._publish({"type": "gen", "bench": b.key, **d})

            # ---- eş keşif çapraz kontrolü: iki bağımsız keşif fizik yasasıyla bağlanır
            eng = engine.best_trees.get("bag_h2")
            if b.key == "kuvvet_h2" and eng is not None:
                b.cross_checks = {"energy_tree": eng}
                engine.log("  çapraz kontrol: kuvvet adayının türevi, bag_h2 şampiyonunun "
                           "enerjisiyle karşılaştırılacak (F9)")
            elif b.key == "bag_h2" and b.cross_checks.get("force_tree") is not None:
                pass
            t0 = time.time()
            res = ev.run(b, cfg, progress=progress, should_stop=lambda: engine.stop_flag)
            engine.best_trees[b.key] = res.pop("champion_tree", None)
            res["wall_before_report_s"] = round(time.time() - t0, 2)
            champ = res.get("champion")
            engine.results[b.key] = res
            record["benchmarks"][b.key] = res
            engine.bench_status[idx]["status"] = "tamamlandı"
            try:
                res["known_limits"] = known_limit_check(b.key, champ, b)
            except Exception as e:
                res["known_limits"] = {"error": f"{type(e).__name__}: {e}"}
            if champ:
                engine.log(f"  ŞAMPİYON [{b.key}]: {champ['formula']}")
                engine.log(f"    maliyet={champ['cost']} birim | train={champ['loss_train']:.3e} | "
                           f"val={champ['loss_val']:.3e} | SINAV(ekstrapolasyon)={champ['loss_test']:.3e} | "
                           f"doğrulandı={champ['verified']}")
                for f in champ["filters"]:
                    engine.log(f"    {f['id']} [{'GEÇTİ' if f['passed'] else 'KALDI'}] {f['name']}: {f['detail'][:80]}")
            _save_json(os.path.join(DATA, "progress.json"), engine.snapshot() | {"full": record})

        # ---- 3) trilyon ölçeği muhasebesi (şampiyon enerji formülü ile)
        if do_throughput and "enerji" in engine.results and engine.results["enerji"].get("champion"):
            engine.state = "olcek"
            champ = engine.results["enerji"]["champion"]
            engine.log(f"ölçeklendirme: {int(n_real_atoms):,} atom gerçek koşu + hedef projeksiyonlar")
            fn = compile_formula(champ["python"], engine.results["enerji"]["var_names"])
            ops = max(champ["cost"], 1.0)
            rep = tp.full_report(fn, ops, n_real=n_real_atoms, targets=targets,
                                 spec=tp.EnsembleSpec(z_min=1.0, z_max=8.0, n_min=1, n_max=8,
                                                      weight_model="thermal", beta=0.15),
                                 seed=seed)
            rep["formula"] = champ["formula"]
            rep["formula_python"] = champ["python"]
            rep["formula_cost"] = ops
            rep["formula_verified"] = champ["verified"]
            engine.throughput = rep
            record["throughput"] = rep
            engine.log(f"  topluluk: Z∈[{rep['spec']['z_min']}, {rep['spec']['z_max']}], n∈[{rep['spec']['n_min']}, {rep['spec']['n_max']}], "
                       f"ağırlık={rep['spec']['weight_model']} (β={rep['spec']['beta']})"
                       f"  E ort={rep['run']['mean_E']:.6g} ± {rep['run']['std_E']:.3g} Hartree")
            engine.log(f"  gerçek koşu: {rep['run']['atoms_realized']:,} atom, "
                       f"{rep['run']['elapsed_s']:.2f} s, {rep['run']['ns_per_atom']:.2f} ns/atom "
                       f"(formül {rep['run']['formula_ns_per_atom']:.2f} + örnekleme {rep['run']['sampling_ns_per_atom']:.2f}), "
                       f"tepe RSS {rep['run']['peak_rss_mb']:.0f} MB, NaN={rep['run']['nonfinite_values']}")
            for p in rep["projections"]:
                engine.log(f"  projeksiyon: {p['atoms']:.0e} atom -> {p['estimated_human']} "
                           f"({p['estimated_joules']:.3g} J tahmini)")

        engine.finished = now_iso()
        record["finished"] = engine.finished
        record["wall_seconds_total"] = round(time.time() - t_start, 2)

        # ---- 4) kayıt + rapor
        engine.state = "rapor"
        _save_json(os.path.join(RUNS, f"{run_id}.json"), record)
        idx = load_index()
        idx.append({
            "run_id": run_id, "preset": preset, "seed": seed,
            "started": engine.started, "finished": engine.finished,
            "wall_seconds_total": record["wall_seconds_total"],
            "champions": {k: (v.get("champion") or {}).get("formula") for k, v in record["benchmarks"].items()},
        })
        _save_index(idx)
        md, html = write_report(record)
        engine.log(f"koşu kaydedildi: data/runs/{run_id}.json  |  rapor: reports/{run_id}.md")
        # Bulut işlerinde (GitHub Actions / Render / HF Spaces) statik siteyi her koşuda tazele:
        # kalıcı yayın bu klasörden servis edilir, sunucu düşse bile site ayakta kalır.
        if os.environ.get("EA_AUTO_EXPORT", "1") == "1":
            try:
                import export_static
                man = export_static.build(verbose=False)
                engine.log(f"kalıcı statik yayın güncellendi: docs/index.html "
                           f"({man['index_html_bytes']//1024} KB, {man['runs']} koşu)")
            except Exception as e:
                engine.log(f"(statik dışa aktarma başarısız: {type(e).__name__}: {e})")
        _save_json(os.path.join(DATA, "progress.json"), engine.snapshot() | {"full": record})
        engine.state = "tamam"
        engine._publish({"type": "done", "run_id": run_id})
    except Exception as e:  # pragma: no cover
        import traceback
        engine.error = f"{type(e).__name__}: {e}"
        engine.log("HATA: " + engine.error)
        engine.log(traceback.format_exc()[-1500:])
        engine.state = "hata"
    return record


def _save_index(idx: List[dict]) -> None:
    _save_json(os.path.join(DATA, "index.json"), idx[-200:])


# ------------------------------------------------------------------ ölçüm yenileme
def refresh_throughput(engine: "Engine", run_id: Optional[str] = None,
                       n_real: int = 200_000_000,
                       targets: Optional[List[float]] = None) -> dict:
    """Son koşudaki enerji şampiyonunu diskten alıp ölçeklendirmeyi yeniden ölçer."""
    if run_id is None:
        idx = load_index()
        run_id = idx[-1]["run_id"] if idx else None
    rec = load_run(run_id) if run_id else None
    key = "enerji"
    champ = None
    var_names = None
    if rec and rec.get("benchmarks", {}).get(key):
        champ = rec["benchmarks"][key].get("champion")
        var_names = rec["benchmarks"][key].get("var_names")
    elif engine.results.get(key):
        champ = engine.results[key].get("champion")
        var_names = engine.results[key].get("var_names")
    if not champ:
        raise RuntimeError("enerji benchmark'ı için kayıtlı şampiyon bulunamadı")
    fn = compile_formula(champ["python"], var_names)
    ops = max(float(champ["cost"]), 1.0)
    engine.log(f"ölçüm yenileniyor (örnekleyici v2): {champ['formula']}")
    rep = tp.full_report(fn, ops, n_real=n_real, targets=targets,
                         spec=tp.EnsembleSpec(z_min=1.0, z_max=8.0, n_min=1, n_max=8,
                                              weight_model="thermal", beta=0.15),
                         seed=7)
    rep["formula"] = champ["formula"]
    rep["formula_python"] = champ["python"]
    rep["formula_cost"] = ops
    rep["formula_verified"] = champ["verified"]
    engine.throughput = rep
    r = rep["run"]
    engine.log(f"  {r['atoms_realized']:,} atom · {r['ns_per_atom']:.2f} ns/atom "
               f"(formül {r['formula_ns_per_atom']:.2f} + örnekleme {r['sampling_ns_per_atom']:.2f}) · "
               f"{r['gflops_achieved']:.2f} GFLOP/s · RSS {r['peak_rss_mb']:.0f} MB")
    if rec:
        rec["throughput"] = rep
        _save_json(os.path.join(RUNS, f"{rec['run_id']}.json"), rec)
        try:
            write_report(rec)      # rapor da yeni ölçümle yeniden üretilir
        except Exception as e:
            engine.log(f"  (rapor yeniden üretilemedi: {e})")
        engine.log(f"  kayıt ve rapor güncellendi: data/runs/{rec['run_id']}.json · reports/{rec['run_id']}.md")
    _save_json(os.path.join(DATA, "progress.json"), engine.snapshot() | {"full": rec or {}})
    return rep


# ------------------------------------------------------------------ rapor
def write_report(record: dict) -> tuple:
    rid = record["run_id"]
    L: List[str] = []
    A = L.append
    A(f"# Evrimsel Keşif Raporu — {rid}")
    A("")
    A(f"- **Başlangıç / bitiş:** {record['started']} → {record.get('finished')}")
    A(f"- **Ön ayar / tohum:** `{record['preset']}` / `{record['seed']}`")
    A(f"- **Toplam süre:** {record.get('wall_seconds_total')} s")
    A(f"- **Yapılandırma:** `{json.dumps(record['config'], ensure_ascii=False)}`")
    A("")
    A("## Bağımsız veri doğrulaması")
    st = record.get("self_test") or {}
    A("")
    A("| ölçüt | değer |")
    A("|---|---|")
    for k in ("norm_residual", "action_residual", "virial_ratio", "grid_delta"):
        if k in st:
            A(f"| {k} | {st[k]:.3e} |")
    meta = st.get("meta") or {}
    for k, v in meta.items():
        if isinstance(v, (int, float, str)):
            A(f"| {k} | {v} |")
    A("")
    A("## Benchmarklar ve keşfedilen formüller")
    for key, res in (record.get("benchmarks") or {}).items():
        c = res.get("champion") or {}
        A("")
        A(f"### {res.get('bench_title')}  (`{key}`)")
        A("")
        A(f"- Değişkenler: `{', '.join(res.get('var_names') or [])}`")
        A(f"- Nesil / değerlendirme: {res.get('generations_run')} / {res.get('evaluations')}")
        A(f"- Numeroloji bariyeri: {res.get('null_barrier')}")
        A(f"- **Keşfedilen en ucuz formül:** `{c.get('formula')}`")
        A(f"- Python: `{c.get('python')}`")
        A(f"- Maliyet: **{c.get('cost')} işlem birimi**, düğüm={c.get('nodes')}, derinlik={c.get('depth')}, "
          f"{c.get('eval_us')} µs/1000 nokta")
        A(f"- Hatalar (göreli RMSE): eğitim={_f(c.get('loss_train'))}, doğrulama={_f(c.get('loss_val'))}, "
          f"**sınav={_f(c.get('loss_test'))}**, en kötü bağıl sınav hatası={_f(c.get('max_rel_err_test'))}")
        A(f"- Doğrulandı mı: **{c.get('verified')}**")
        if c.get("generalization_ok") is False:
            A("")
            A("> ⚠️ **Genelleme uyarısı:** sınav (ekstrapolasyon) hatası toleransın 5 katından "
              "büyük. Formül eğitim/doğrulama rejiminde geçerli sayılır, **sınav rejiminde "
              "değil**; sınav hatası yukarıda dürüstçe raporlanmıştır.")
        A("")
        A("| filtre | grup | sonuç | ayrıntı |")
        A("|---|---|---|---|")
        for f in c.get("filters", []):
            A(f"| {f['id']} — {f['name']} | {f['group']} | {'✅' if f['passed'] else '❌'} | {f['detail']} |")
        A("")
        kl = res.get("known_limits")
        if kl and not kl.get("error"):
            A("**Bilinen analitik limit karşılaştırması (yalnızca teşhis — seçimde kullanılmadı):**")
            A("")
            A(f"- {kl.get('limit')}")
            if "max_rel_deviation" in kl:
                A(f"- en büyük bağıl sapma: **{_f(kl['max_rel_deviation'])}**, "
                  f"ortalama: {_f(kl['mean_rel_deviation'])}")
            if "perturbation_table" in kl:
                A("")
                A("| terim | bağımsız çözücü | keşfedilen formül | bağıl fark |")
                A("|---|---|---|---|")
                for t in kl["perturbation_table"]:
                    A(f"| {t['term']} | {t['solver']:.8f} | {t['formula']:.8f} | {_f(t['rel_diff'])} |")
            A("")
        hall = res.get("hall_of_fame") or []
        if hall:
            A("**Onur listesi (en iyiler):**")
            A("")
            A("| formül | maliyet | train | val | sınav | doğrulandı |")
            A("|---|---|---|---|---|---|")
            for h in hall[:6]:
                A(f"| `{h['formula']}` | {h['cost']} | {_f(h.get('loss_train'))} | {_f(h.get('loss_val'))} | "
                  f"{_f(h.get('loss_test'))} | {h['verified']} |")
            A("")
        if res.get("physics_seeded"):
            A(f"Şeffaflık — kullanılan fizik ön-bilgisi şablonları (bunlar cevap değil, boş kalıplardır): "
              f"`{'`, `'.join(res.get('seeds_used') or [])}`")
            A("")
    tp_rep = record.get("throughput")
    if tp_rep:
        A("## Trilyon ölçeği muhasebesi")
        A("")
        r = tp_rep["run"]
        A(f"- Formül: `{tp_rep['formula']}` (maliyet {tp_rep['formula_cost']} işlem/atom)")
        A(f"- Gerçek koşu: **{r['atoms_realized']:,} atom**, {r['elapsed_s']:.3f} s, "
          f"**{r['ns_per_atom']:.2f} ns/atom**, {r['gflops_achieved']:.2f} GFLOP/s")
        A(f"- Bellek: tepe RSS {r['peak_rss_mb']:.0f} MB (öbek={r['chunk']:,} → bellek O(öbek), atom sayısıyla büyümez)")
        A(f"- Sayısal güvenlik: NaN/inf sayısı = {r['nonfinite_values']}, "
          f"E ∈ [{r['E_min']:.6g}, {r['E_max']:.6g}] Hartree")
        A("")
        A("| atom | tahmini süre (tek çekirdek, dışdeğerleme) | işlem sayısı | tahmini enerji (J) |")
        A("|---|---|---|---|")
        for p in tp_rep["projections"]:
            A(f"| {p['atoms']:.0e} | {p['estimated_human']} | {p['fp64_flops']:.3e} | {p['estimated_joules']:.3g} |")
        A("")
        A(f"- Pahalı yol (sayısal çözücü) referansı: {tp_rep['baseline_solver']['N_grid']} ızgara noktası, "
          f"{tp_rep['baseline_solver']['wall_s_per_configuration']:.3f} s/yapılandırma → "
          f"işlem başına hızlanma ≈ **{tp_rep['speedup_vs_solver_ops_per_atom']:.3g}×**, "
          f"süre hızlanması ≈ **{tp_rep['speedup_vs_solver_seconds']:.3g}×**")
        A(f"- Varsayımlar: {tp_rep['assumptions']['note']}")
        A("")
        A("## Dürüstlük notları")
        A("")
        A("- Referans veri kapalı formdan DEĞİL, sonlu farklar özdeğer çözümünden gelir; "
          "ızgara yakınsaması ve varyasyonel eylem durağanlığı ile doğrulanmıştır.")
        A("- Sınav (test) kümesi eğitim bölgesinin dışındadır ve yalnızca rapor aşamasında bir kez ölçülür.")
        A("- Numeroloji bariyeri: aynı mimari etiketleri karıştırılmış veride çalıştırılmış, "
          "elde edilen en iyi hatanın 5 katı altına inmeyen aday elenmiştir.")
        A("- Trilyon atom ölçeği bir TOPLULUK (tek-atom durumlarının ortalaması) simülasyonudur; "
          "tam çok-cisimli kuantum durumu hesabı değildir.")
    md = "\n".join(L)
    md_path = os.path.join(REPORTS, f"{rid}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    html = md_to_html(md, rid)
    with open(os.path.join(REPORTS, f"{rid}.html"), "w", encoding="utf-8") as f:
        f.write(html)
    return md, html


def md_to_html(md: str, rid: str) -> str:
    """Küçük, bağımsız (external kaynaksız) Markdown→HTML dönüştürücü."""
    import html as _h
    out, in_tbl, in_code = [], False, False
    for line in md.split("\n"):
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            tag = "th" if not in_tbl else "td"
            if not in_tbl:
                out.append('<table>'); in_tbl = True
            out.append("<tr>" + "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_tbl:
            out.append("</table>"); in_tbl = False
        if line.startswith("```"):
            out.append("<pre>" if not in_code else "</pre>"); in_code = not in_code; continue
        t = _inline(line)
        if line.startswith("### "):
            out.append(f"<h3>{t[4:]}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{t[3:]}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{t[2:]}</h1>")
        elif line.startswith("- ") or line.startswith("* "):
            out.append(f"<li>{t[2:]}</li>")
        elif line.strip() == "":
            out.append("")
        else:
            out.append(f"<p>{t}</p>")
    if in_tbl:
        out.append("</table>")
    body = "\n".join(out)
    return f"""<!DOCTYPE html><html lang="tr"><head><meta charset="utf-8">
<title>Rapor {rid}</title><style>
body{{background:#0b0f14;color:#e6edf3;font:15px/1.6 ui-monospace,Menlo,Consolas,monospace;margin:0;padding:28px;max-width:1100px}}
h1,h2,h3{{color:#7ee787;font-weight:600}} h1{{border-bottom:1px solid #21262d;padding-bottom:8px}}
table{{border-collapse:collapse;width:100%;margin:10px 0}} td,th{{border:1px solid #21262d;padding:6px 9px;text-align:left;font-size:13px}}
th{{background:#161b22;color:#79c0ff}} code,pre{{background:#161b22;border-radius:4px;padding:1px 5px;color:#ffa657}}
pre{{padding:10px;overflow:auto}} li{{margin-left:18px}} p{{color:#c9d1d9}}
</style></head><body>{body}</body></html>"""


def _f(x) -> str:
    """Güvenli sayı biçimlendirme (None/NaN toleranslı)."""
    try:
        if x is None:
            return "—"
        return f"{float(x):.3e}"
    except Exception:
        return "—"


def _inline(s: str) -> str:
    import html as _h
    import re as _re
    s = _h.escape(s)
    s = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = _re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s
