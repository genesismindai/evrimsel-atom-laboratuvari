#!/usr/bin/env python3
"""
server.py — Bağımlılıksız (saf stdlib) yayın sunucusu.

Neden stdlib? Sandbox/prod ortamında pip paketleri kalıcı değildir; bu sunucu
hiçbir dış bağımlılık olmadan çalışır ve yayın için gereken her şeyi yapar:

  GET  /                     canlı panel (web/index.html)
  GET  /methods              yöntem + filtreler + dürüstlük kuralları
  GET  /publish              yayın kaydı (geçici bağlantı → kalıcı kanıt)
  GET  /api/status           canlı durum (JSON)
  GET  /api/stream           sunucu-olay akışı (SSE, chunked)
  GET  /api/index            koşu dizini (kalıcı arşiv)
  GET  /api/run/<id>         koşunun tam kaydı
  GET  /api/progress         son yazılan ilerleme dosyası
  GET  /api/reference_meta   referans çözücü doğrulama ölçütleri
  GET  /api/publish_record   yayın kaydı (gözlemlenen genel adresler dâhil)
  POST /api/run              koşu başlat  {preset, seed, do_null}
  POST /api/stop             koşuyu durdur
  GET  /reports/<dosya>      üretilmiş raporlar (md/html)

ÖNEMLİ: Sunucu 0.0.0.0'a bağlanır (önizleme proxy'si ile uyumlu) ve tüm
içerik aynı kaynaktan (relative URL) sunulur.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from runner import (Engine, run_all, load_index, load_run, PRESETS,  # noqa: E402
                    DATA, REPORTS, now_iso, _save_json, refresh_throughput)

WEB = os.path.join(ROOT, "web")
PUBLISH_RECORD = os.path.join(DATA, "publish_records.json")

ENGINE = Engine()
RUN_THREAD: list = []
HOST_PORT = os.environ.get("PORT", "8000")
OBSERVED_HOSTS: set = set()


def record_publication(host_header: str | None) -> None:
    """Yayın kaydı: geçici adres(ler) ve zaman damgası diske işlenir."""
    if host_header and host_header not in OBSERVED_HOSTS:
        OBSERVED_HOSTS.add(host_header)
    recs = []
    if os.path.exists(PUBLISH_RECORD):
        try:
            recs = json.load(open(PUBLISH_RECORD, encoding="utf-8"))
        except Exception:
            recs = []
    entry = {
        "recorded_at": now_iso(),
        "bind": f"0.0.0.0:{HOST_PORT}",
        "observed_origins": sorted(OBSERVED_HOSTS),
        "note": ("Bu adresler geçici/oturumluk olabilir. Bilimsel kanıtlar kalıcıdır: "
                 "data/runs/*.json ve reports/*.md|.html"),
    }
    if recs and recs[-1].get("observed_origins") == entry["observed_origins"]:
        recs[-1] = entry
    else:
        recs.append(entry)
    _save_json(PUBLISH_RECORD, recs[-100:])


class Handler(BaseHTTPRequestHandler):
    server_version = "AtomEvolutionLab/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):        # gürültüyü azalt
        pass

    # ---------------- yardımcılar
    def _json(self, obj, code: int = 200):
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: str, ctype: str | None = None):
        if not os.path.exists(path) or not os.path.isfile(path):
            self._json({"error": "bulunamadı", "path": os.path.basename(path)}, 404)
            return
        ctype = ctype or {".html": "text/html; charset=utf-8",
                          ".md": "text/markdown; charset=utf-8",
                          ".json": "application/json; charset=utf-8",
                          ".js": "application/javascript; charset=utf-8",
                          ".css": "text/css; charset=utf-8"}.get(
            os.path.splitext(path)[1], "application/octet-stream")
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream(self):
        """SSE: chunked transfer ile canlı olay akışı."""
        q: queue.Queue = queue.Queue(maxsize=512)
        with ENGINE.lock:
            ENGINE.subscribers.append(q)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(obj) -> None:
            data = ("data: " + json.dumps(obj, ensure_ascii=False, default=str) + "\n\n").encode("utf-8")
            self.wfile.write(b"%X\r\n" % len(data) + data + b"\r\n")
            self.wfile.flush()

        try:
            send({"type": "snapshot", "state": ENGINE.state, "run_id": ENGINE.run_id})
            t_last = time.time()
            while True:
                try:
                    obj = q.get(timeout=3.0)
                    send(obj)
                except queue.Empty:
                    if time.time() - t_last > 12:
                        send({"type": "ping"})
                        t_last = time.time()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with ENGINE.lock:
                if q in ENGINE.subscribers:
                    ENGINE.subscribers.remove(q)

    # ---------------- yönlendirme
    def do_GET(self):                                     # noqa: N802
        u = urlparse(self.path)
        p = u.path
        record_publication(self.headers.get("Host"))
        if p in ("/", "/index.html"):
            return self._file(os.path.join(WEB, "index.html"))
        if p in ("/methods", "/methods.html"):
            return self._file(os.path.join(WEB, "methods.html"))
        if p in ("/publish", "/publish.html"):
            return self._file(os.path.join(WEB, "publish.html"))
        if p in ("/healthz", "/api/health"):
            return self._json({"ok": True, "state": ENGINE.state, "run_id": ENGINE.run_id,
                               "time": now_iso(), "uptime_hint": "keep-alive ping için uygun"})
        if p == "/api/export":
            try:
                import export_static
                man = export_static.build(verbose=False)
                return self._json({"ok": True, "manifest": man})
            except Exception as e:
                return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)
        if p == "/api/status":
            return self._json(ENGINE.snapshot())
        if p == "/api/stream":
            return self._stream()
        if p == "/api/index":
            return self._json(load_index())
        if p.startswith("/api/run/"):
            rid = p.split("/")[-1]
            rec = load_run(rid)
            return self._json(rec if rec else {"error": "koşu bulunamadı", "run_id": rid},
                              200 if rec else 404)
        if p == "/api/progress":
            fp = os.path.join(DATA, "progress.json")
            return self._file(fp) if os.path.exists(fp) else self._json({"state": ENGINE.state})
        if p == "/api/reference_meta":
            return self._file(os.path.join(DATA, "reference_meta.json"))
        if p == "/api/publish_record":
            recs = json.load(open(PUBLISH_RECORD, encoding="utf-8")) if os.path.exists(PUBLISH_RECORD) else []
            return self._json(recs)
        if p == "/api/benchmarks":
            from atomic_ea import benchmarks as bm
            return self._json([{"key": b.key, "title": b.title, "purpose": b.purpose,
                                "var_names": b.var_names, "noise_rel": b.noise_rel,
                                "tol_rel": b.tol_rel, "cost_budget": b.cost_budget}
                               for b in bm.build_all(verbose=False)])
        if p == "/static-site" or p.startswith("/static-site/"):
            # kanıp: kalıcı statik yayının (docs/) canlı sunucudan önizlemesi
            rel = p[len("/static-site"):].lstrip("/") or "index.html"
            target = os.path.normpath(os.path.join(ROOT, "docs", rel))
            docs_root = os.path.normpath(os.path.join(ROOT, "docs"))
            if not target.startswith(docs_root):
                return self._json({"error": "geçersiz yol"}, 400)
            if os.path.isdir(target):
                target = os.path.join(target, "index.html")
            return self._file(target)
        if p.startswith("/reports/"):
            name = os.path.basename(p)
            return self._file(os.path.join(REPORTS, name))
        if p == "/favicon.ico":
            self.send_response(204); self.end_headers(); return
        return self._json({"error": "yol yok", "path": p}, 404)

    def do_POST(self):                                    # noqa: N802
        p = urlparse(self.path).path
        record_publication(self.headers.get("Host"))
        length = int(self.headers.get("Content-Length") or 0)
        body = {}
        if length:
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception:
                body = {}
        if p == "/api/run":
            running = any(t.is_alive() for t in RUN_THREAD)
            if running:
                return self._json({"error": "zaten bir koşu sürüyor",
                                   "state": ENGINE.state, "run_id": ENGINE.run_id}, 409)
            preset = body.get("preset", "hizli")
            if preset not in PRESETS:
                return self._json({"error": f"geçersiz ön ayar: {preset}",
                                   "gçerçeveli": list(PRESETS)}, 400)
            seed = int(body.get("seed", 1))
            do_null = bool(body.get("do_null", True))
            n_real = int(body.get("n_real_atoms", 100_000_000))

            def job():
                run_all(ENGINE, preset=preset, seed=seed, do_null=do_null,
                        do_throughput=True, n_real_atoms=n_real)

            t = threading.Thread(target=job, name="ea-run", daemon=True)
            RUN_THREAD.append(t)
            t.start()
            return self._json({"started": True, "preset": preset, "seed": seed,
                               "do_null": do_null, "atoms_real": n_real})
        if p == "/api/throughput/refresh":
            n_real = int(body.get("n_real_atoms", 200_000_000))
            try:
                rep = refresh_throughput(ENGINE, run_id=body.get("run_id"), n_real=n_real)
                return self._json({"ok": True, "run": rep["run"],
                                   "projections": rep["projections"]})
            except Exception as e:
                return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 400)
        if p == "/api/stop":
            ENGINE.stop_flag = True
            ENGINE.log("durdurma isteği alındı — koşu güvenli noktada sonlanacak")
            return self._json({"stopping": True})
        return self._json({"error": "yol yok"}, 404)


def maybe_autostart() -> None:
    """
    Bulut barındırma için: AUTO_START=1 ise sunucu açılır açılmaz koşu başlar.
    AUTO_FOREVER=1 ise koşular aralıksız sürer (kesintisiz keşif modu).
    """
    if os.environ.get("AUTO_START", "0") != "1":
        return
    preset = os.environ.get("AUTO_PRESET", "hizli")
    n_real = int(float(os.environ.get("AUTO_ATOMS", "2e8")))
    forever = os.environ.get("AUTO_FOREVER", "0") == "1"
    interval = int(float(os.environ.get("AUTO_INTERVAL", "30")))
    do_null = os.environ.get("AUTO_NULL", "1") == "1"

    def job():
        i = 0
        while True:
            seed = int(os.environ.get("AUTO_SEED", "1")) + i
            run_all(ENGINE, preset=preset, seed=seed, do_null=do_null,
                    do_throughput=True, n_real_atoms=n_real)
            i += 1
            if not forever or ENGINE.stop_flag:
                break
            ENGINE.log(f"kesintisiz mod: {interval} s bekleniyor, sıradaki tohum {int(os.environ.get('AUTO_SEED','1'))+i}")
            for _ in range(interval):
                if ENGINE.stop_flag:
                    break
                time.sleep(1)
            ENGINE.stop_flag = False
        ENGINE.log("otomatik koşu dizisi tamamlandı")

    ENGINE.log(f"otomatik başlatma: ön ayar={preset}, sonsuz mod={forever}, atom={n_real:,}")
    threading.Thread(target=job, name="auto-run", daemon=True).start()


def main() -> None:
    port = int(HOST_PORT)
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.daemon_threads = True
    ENGINE.log(f"yayın sunucusu açıldı: 0.0.0.0:{port}  (tüm içerik kalıcı: data/, reports/, docs/)")
    record_publication(None)
    maybe_autostart()
    print(f"[yayın] http://0.0.0.0:{port} dinleniyor — panel: /  ·  yöntem: /methods  ·  kayıt: /publish")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[yayın] kapatılıyor…")
        srv.shutdown()


if __name__ == "__main__":
    main()
