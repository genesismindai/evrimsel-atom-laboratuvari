# Docker: 7/24 canlı Python sunucusu (Hugging Face Spaces / Render / Fly.io / VPS)
#
# Neden: kalıcı ve ücretsiz canlı yayın için tek komutla dağıtılabilir imaj.
# Statik kalıcı site (docs/) için Docker GEREKMEZ — bkz. DEPLOY.md, yol A.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    # --- bulut davranışı ---
    AUTO_START=1 \
    AUTO_FOREVER=1 \
    AUTO_PRESET=hizli \
    AUTO_ATOMS=2e8 \
    AUTO_INTERVAL=20 \
    EA_AUTO_EXPORT=1

WORKDIR /app

# yalnız bilimsel çekirdek: numpy + scipy (sunucu saf stdlib'dir, ek paket gerekmez)
RUN pip install --no-cache-dir numpy scipy

COPY . .

# referans veri önbelleğini imaj içinde hazırla (ilk açılış hızlı olsun)
RUN python -c "from atomic_ea import benchmarks as bm; bm.build_reference(verbose=True)" || true

EXPOSE 8000
HEALTHCHECK --interval=45s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=4).status==200 else 1)"

CMD ["python", "web/server.py"]
