#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 genesismindai (Evrimsel Atom Laboratuvarı)
#
# Bu program özgür yazılımdır: GNU Genel Kamu Lisansı (GPL) sürüm 3 veya
# sonraki sürümleri koşulları altında yeniden dağıtabilir ve/veya
# değiştirebilirsiniz. Ayrıntılar için LICENSE dosyasına bakın.
"""
runner_cli.py — Komut satırı koşusu, sürekli (kesintisiz) keşif modu ve arşiv yönetimi.

Tek seferlik koşu:
    python3 runner_cli.py --preset hizli --seed 7

Sürekli mod (bulut işleri/kalıcı yayın için): her turda yeni tohumla koşar,
her turdan sonra statik siteyi (docs/) tazeler. Durdurmak için:
  * Ctrl+C, veya
  * aynı klasörde `data/STOP` dosyası oluşturmak (örn. `touch data/STOP`)
    — bu, GitHub Actions gibi ortamlarda çalışırken temiz çıkış sağlar.

    python3 runner_cli.py --forever --preset hizli --interval 60
    python3 runner_cli.py --forever --max-runs 5 --atoms 5e8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from runner import Engine, run_all, PRESETS, DATA, REPORTS, load_index  # noqa: E402

STOP_FILE = os.path.join(DATA, "STOP")


def console_engine() -> Engine:
    eng = Engine()
    eng.log = lambda msg: print(msg, flush=True)     # konsol günlüğü
    return eng


def one_run(eng: Engine, preset: str, seed: int, do_null: bool,
            do_throughput: bool, atoms: float) -> dict:
    return run_all(eng, preset=preset, seed=seed, do_null=do_null,
                   do_throughput=do_throughput, n_real_atoms=int(atoms))


def main() -> None:
    ap = argparse.ArgumentParser(description="Filtreli evrimsel atom fiziği koşusu")
    ap.add_argument("--preset", choices=list(PRESETS), default="hizli")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-null", action="store_true", help="numeroloji bariyeri kalibrasyonunu atla")
    ap.add_argument("--no-throughput", action="store_true", help="trilyon ölçeği muhasebesini atla")
    ap.add_argument("--atoms", type=float, default=1e8, help="gerçek koşulacak atom sayısı (örn. 1e9)")
    ap.add_argument("--forever", action="store_true", help="kesintisiz mod: koşular aralıksız sürer")
    ap.add_argument("--interval", type=int, default=30, help="sürekli modda turlar arası bekleme (s)")
    ap.add_argument("--max-runs", type=int, default=0, help="sürekli modda en fazla koşu (0 = sınırsız)")
    ap.add_argument("--export", action="store_true", help="koşudan sonra statik siteyi üret (docs/)")
    a = ap.parse_args()

    eng = console_engine()
    if a.export:
        os.environ["EA_AUTO_EXPORT"] = "1"

    if not a.forever:
        rec = one_run(eng, a.preset, a.seed, not a.no_null, not a.no_throughput, a.atoms)
        print("\n=== ÖZET ===")
        for k, v in rec["benchmarks"].items():
            c = v.get("champion") or {}
            print(f"{k:16s} → {c.get('formula')}   (maliyet {c.get('cost')}, "
                  f"sınav hatası {c.get('loss_test')}, doğrulandı={c.get('verified')})")
        print(f"kayıt: data/runs/{rec['run_id']}.json · rapor: reports/{rec['run_id']}.md")
        if a.export or os.environ.get("EA_AUTO_EXPORT") == "1":
            import export_static
            export_static.build(verbose=True)
        return

    # ---------------- sürekli mod
    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)
    n = 0
    print(f"[sürekli mod] başlangıç tohumu {a.seed}, bekleme {a.interval} s, "
          f"maks koşu {a.max_runs or '∞'} · durdurmak için: touch {os.path.relpath(STOP_FILE)}")
    try:
        while True:
            seed = a.seed + n
            print(f"\n[sürekli mod] === koşu #{n+1} (tohum {seed}) ===")
            try:
                one_run(eng, a.preset, seed, not a.no_null, not a.no_throughput, a.atoms)
            except Exception as e:
                print(f"[sürekli mod] koşu hatası: {type(e).__name__}: {e} — devam ediliyor")
            n += 1
            idx = load_index()
            print(f"[sürekli mod] arşivde {len(idx)} koşu var · son: {idx[-1]['run_id'] if idx else '—'}")
            if a.max_runs and n >= a.max_runs:
                print(f"[sürekli mod] maks koşu sayısına ulaşıldı ({a.max_runs}), çıkılıyor")
                break
            if os.path.exists(STOP_FILE):
                print("[sürekli mod] STOP dosyası bulundu, temiz çıkış")
                os.remove(STOP_FILE)
                break
            time.sleep(max(1, a.interval))
    except KeyboardInterrupt:
        print("\n[sürekli mod] Ctrl+C — temiz çıkış")


if __name__ == "__main__":
    main()
