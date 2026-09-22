#!/usr/bin/env python3
"""
export_static.py — Kalıcı, ücretsiz yayın için STATİK site üreticisi.

Neden: geliştirme sanal alanı (sandbox) oturumla birlikte kapanır; içindeki
sunucu yaşayamaz. Bu yüzden sonuçlar sunucudan bağımsız, TEK dosyada çalışan
(bütün verisi gömülü) statik bir siteye aktarılır:

    docs/index.html        canlı panelin statik eşdeğeri (veri gömülü, fetch yok)
    docs/methods.html      yöntem + filtre zinciri + dürüstlük kuralları
    docs/publish.html      kalıcılık ve yeniden üretim kaydı
    docs/runs/<id>.json    her koşunun tam kaydı (makine-okur kanıt)
    docs/reports/<id>.md   Markdown raporlar
    docs/reports/<id>.html HTML raporlar
    docs/data/*.json       dizin, bariyerler, çözücü doğrulama özeti, yayın kaydı

Bu klasör GitHub Pages / Netlify / Cloudflare Pages gibi ücretsiz statik
barındırmalarda sonsuza kadar yayınlanabilir; Python kodu ise GitHub Actions
(zamanlanmış iş) olarak koşup bu klasörü yeniden üretir.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from runner import DATA, REPORTS, RUNS, load_index, load_run  # noqa: E402

WEB = os.path.join(ROOT, "web")


def _read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def summarize_for_static(rec: dict, max_history: int = 60) -> dict:
    """Koşu kaydını statik sayfa için sadeleştirir (boyut/hız dengesi)."""
    benches_out = {}
    for key, res in (rec.get("benchmarks") or {}).items():
        hist = res.get("history") or []
        step = max(1, len(hist) // max_history)
        hist_small = [h for i, h in enumerate(hist) if i % step == 0 or i == len(hist) - 1]
        benches_out[key] = {
            "bench_title": res.get("bench_title"),
            "var_names": res.get("var_names"),
            "generations_run": res.get("generations_run"),
            "evaluations": res.get("evaluations"),
            "wall_seconds": res.get("wall_seconds"),
            "null_barrier": res.get("null_barrier"),
            "physics_seeded": res.get("physics_seeded"),
            "seeds_used": (res.get("seeds_used") or [])[:6],
            "champion": res.get("champion"),
            "hall_of_fame": (res.get("hall_of_fame") or [])[:6],
            "known_limits": res.get("known_limits"),
            "history": hist_small,
        }
    return {
        "started": rec.get("started"), "finished": rec.get("finished"),
        "wall_seconds_total": rec.get("wall_seconds_total"),
        "config": rec.get("config"),
        "self_test": rec.get("self_test"),
        "throughput": rec.get("throughput"),
        "benchmarks": benches_out,
    }


EMBED_LAST_RUNS = 25     # sayfaya gömülecek en yeni koşu sayısı (eskiler runs/ içinde kalır)


def build(out_dir: str = "docs", verbose: bool = True,
          embed_last: int = EMBED_LAST_RUNS) -> dict:
    out = out_dir if os.path.isabs(out_dir) else os.path.join(ROOT, out_dir)
    os.makedirs(os.path.join(out, "runs"), exist_ok=True)
    os.makedirs(os.path.join(out, "reports"), exist_ok=True)
    os.makedirs(os.path.join(out, "data"), exist_ok=True)

    idx = load_index()
    details = {}
    copied_runs, copied_reps = 0, 0
    embed_ids = {e.get("run_id") for e in idx[-embed_last:]} if embed_last else {e.get("run_id") for e in idx}
    for entry in idx:
        rid = entry.get("run_id")
        if not rid:
            continue
        rec = load_run(rid)
        if rec:
            if rid in embed_ids or len(details) < embed_last:
                details[rid] = summarize_for_static(rec)
            shutil.copyfile(os.path.join(RUNS, f"{rid}.json"),
                            os.path.join(out, "runs", f"{rid}.json"))
            copied_runs += 1
        for ext in (".md", ".html"):
            srcp = os.path.join(REPORTS, f"{rid}{ext}")
            if os.path.exists(srcp):
                shutil.copyfile(srcp, os.path.join(out, "reports", f"{rid}{ext}"))
                copied_reps += 1

    progress = _read_json(os.path.join(DATA, "progress.json"), {})
    if progress.get("run_id") and progress["run_id"] not in details and progress.get("full"):
        details[progress["run_id"]] = summarize_for_static(progress["full"])

    payload = {
        "generated_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "runs": idx,
        "details": details,
        "null_barriers": _read_json(os.path.join(DATA, "null_barriers.json"), {}),
        "reference_meta": _read_json(os.path.join(DATA, "reference_meta.json"), {}),
        "publish_records": _read_json(os.path.join(DATA, "publish_records.json"), []),
        "note": ("Statik yayın: veri sayfaya gömülüdür, ağ çağrısı yapılmaz; "
                 "sunucu/oturum kapansa da erişilebilir kalır."),
    }

    tmpl_path = os.path.join(WEB, "static_template.html")
    with open(tmpl_path, encoding="utf-8") as f:
        html = f.read()
    blended = html.replace("/*__DATA__*/ {\"runs\":[],\"details\":{}}", json.dumps(payload, ensure_ascii=False))
    if blended == html:
        raise RuntimeError("şablon yer tutucusu bulunamadı (/*__DATA__*/)")
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(blended)

    # yöntem + kalıcılık sayfaları
    shutil.copyfile(os.path.join(WEB, "methods.html"), os.path.join(out, "methods.html"))
    write_static_publish(out, payload)
    with open(os.path.join(out, ".nojekyll"), "w") as f:
        f.write("")     # GitHub Pages'in alt klasörleri olduğu gibi sunması için

    for name in ("index.json", "null_barriers.json", "reference_meta.json", "publish_records.json"):
        srcp = os.path.join(DATA, name)
        if os.path.exists(srcp):
            shutil.copyfile(srcp, os.path.join(out, "data", name))

    manifest = {
        "built_at": payload["generated_at"],
        "runs": len(idx),
        "run_files": copied_runs,
        "report_files": copied_reps,
        "index_html_bytes": os.path.getsize(os.path.join(out, "index.html")),
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"[statik] {out}/index.html üretildi ({manifest['index_html_bytes']//1024} KB, "
              f"{len(idx)} koşu, {copied_runs} kayıt, {copied_reps} rapor)")
    return manifest


def write_static_publish(out: str, payload: dict) -> None:
    """Kalıcılık kaydı sayfası (statik; geçici adres listesini de gösterir)."""
    records = payload.get("publish_records") or []
    origins = sorted({o for r in records for o in (r.get("observed_origins") or [])})
    rows = "".join(
        f"<tr><td><code>{r.get('run_id','—')}</code></td><td>{r.get('preset','—')} / {r.get('seed','—')}</td>"
        f"<td>{r.get('started','—')}</td><td>{r.get('finished','—')}</td>"
        f"<td class='num'>{r.get('wall_seconds_total','—')} s</td></tr>"
        for r in reversed(payload.get("runs") or [])
    ) or "<tr><td colspan='5'>henüz koşu yok</td></tr>"
    origins_html = "".join(f"<li><code>{o}</code></li>" for o in origins) or "<li>—</li>"
    html = f"""<!DOCTYPE html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kalıcılık ve yeniden üretim kaydı</title>
<style>body{{margin:0;background:#080b10;color:#e6edf3;font:14px/1.6 ui-monospace,Menlo,Consolas,monospace}}
.wrap{{max-width:1050px;margin:0 auto;padding:26px 22px 70px}}h1{{font-size:20px;color:#7ee787}}
h2{{font-size:14px;color:#39d0d8;text-transform:uppercase;letter-spacing:.1em;margin-top:26px}}
code{{background:#131b25;padding:1px 5px;border-radius:4px;color:#ffa657}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}}
td,th{{border:1px solid #1d2836;padding:6px 9px;text-align:left}}th{{background:#131b25;color:#79c0ff}}
a{{color:#58a6ff}}.ok{{border-left:3px solid #3fb950;padding-left:12px;color:#bfe3c6}}
li{{color:#c9d1d9}}</style></head><body><div class="wrap">
<h1>Kalıcılık ve yeniden üretim kaydı</h1>
<div class="ok">Bu site <b>statik</b> olarak yayınlanır: bütün veri <code>index.html</code> içine gömülüdür,
ağ çağrısı yapmaz. Sunucu ya da geliştirme oturumu kapansa da erişilebilir kalır. Sayfa üretimi:
<b>{payload.get('generated_at','—')}</b> · arşivdeki koşu sayısı: <b>{len(payload.get('runs') or [])}</b>.</div>
<h2>Yeniden üretilebilirlik</h2>
<ul>
<li>Her koşu: tohum, yapılandırma, referans veri sha256'sı ve tüm formüllerle <code>runs/&lt;id&gt;.json</code> olarak saklanır.</li>
<li>Raporlar: <code>reports/&lt;id&gt;.md</code> (Markdown) ve <code>reports/&lt;id&gt;.html</code> (bağımsız HTML).</li>
<li>Yeniden üretme: <code>python3 runner_cli.py --preset hizli --seed &lt;tohum&gt;</code> + <code>python3 export_static.py</code>.</li>
<li>Referans veri: sonlu farklar Schrödinger çözücüsü (kapalı form kullanılmaz); doğrulama ölçütleri <code>data/reference_meta.json</code>.</li>
</ul>
<h2>Gözlemlenen yayın adresleri (geçici olabilir)</h2>
<ul>{origins_html}</ul>
<p>Geçici adresler oturuma bağlıdır; kalıcı olan bu statik arşivdir.</p>
<h2>Koşu arşivi</h2>
<table><tr><th>koşu</th><th>ön ayar / tohum</th><th>başlangıç</th><th>bitiş</th><th>süre</th></tr>{rows}</table>
<p style="margin-top:16px"><a href="index.html">← panele dön</a> · <a href="methods.html">yöntem →</a></p>
</div></body></html>"""
    with open(os.path.join(out, "publish.html"), "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Kalıcı statik yayın üret")
    ap.add_argument("--out", default="docs", help="çıktı klasörü (varsayılan: docs)")
    a = ap.parse_args()
    build(a.out)
