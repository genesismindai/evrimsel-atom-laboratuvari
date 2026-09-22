#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# publish_artifacts.sh — Sonuçları git arşivine + kalıcı statik yayına işler.
#
# Kullanım:  ./publish_artifacts.sh "açıklama"
#
# Ne yapar:
#   1) docs/ (statik site) yeniden üretilir  -> sunucu olmadan da site ayakta
#   2) data/, reports/, docs/ git'e eklenir ve commit'lenir
#   3) varsa uzak depoya push edilir (GitHub Pages / Netlify / Cloudflare bunu yayınlar)
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

MSG="${1:-keşif koşusu kaydı $(date -u +%Y-%m-%dT%H:%MZ)}"

echo "[yayın] statik site üretiliyor…"
python3 export_static.py --out docs

if [ ! -d .git ]; then
  echo "[yayın] git deposu yok. Kurmak için:"
  echo "    git init && git add -A && git commit -m 'ilk kayıt'"
  echo "    git remote add origin git@github.com:<kullanıcı>/<depo>.git"
  echo "    git branch -M main && git push -u origin main"
  exit 1
fi

git add -A data reports docs

if git diff --staged --quiet; then
  echo "[yayın] değişiklik yok — commit gerekmiyor"
  exit 0
fi

git -c user.name="${GIT_AUTHOR_NAME:-evrimsel-lab}" \
    -c user.email="${GIT_AUTHOR_EMAIL:-lab@example.com}" \
    commit -m "$MSG"

if git remote get-url origin >/dev/null 2>&1; then
  echo "[yayın] uzak depoya gönderiliyor…"
  git pull --rebase --autostash || true
  git push
  echo "[yayın] tamam. GitHub Pages açıksa site birkaç saniye içinde güncellenir."
else
  echo "[yayın] uzak depo tanımlı değil; yerel commit yapıldı."
fi
