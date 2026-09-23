#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup_github.sh — A YOLU: tek komutla kalıcı GitHub yayını kurulumu
#
# Ne yapar (hepsi idempotent; tekrar çalıştırılabilir):
#   1) Depoyu GitHub'da oluşturur (yoksa)            [API]
#   2) Kodu main dalına gönderir                     [git push]
#   3) GitHub Pages'i "Actions" kaynağıyla etkinleştirir [API]
#   4) Actions iş akışı izinlerini "Read and write" yapar [API]
#   5) İlk keşif koşusunu elle tetikler               [API]
#   6) Pages adresini ve koşu durumunu bekleyip yazdırır
#
# KULLANIM (iki yol):
#
#   A) Token vererek (önerilen — kurulumu bu betik yapar):
#      export GITHUB_TOKEN=github_pat_xxx        # veya ghp_xxx
#      ./setup_github.sh <kullanici>/<depo> [public|private]
#
#   B) Token istemiyorsanız: depoyu tarayıcıda açıp şu komutları kullanın:
#      git remote add origin https://github.com/<kullanici>/<depo>.git
#      git push -u origin main
#      (sonra: Settings → Pages → Source: GitHub Actions;
#       Settings → Actions → Workflow permissions → Read and write)
#
# TOKEN İZNİ (fine-grained PAT): Repository permissions →
#   Contents: Read and write · Actions: Read and write · Pages: Read and write ·
#   Administration: Read and write (depo oluşturma/silme ve Pages ayarı için) ·
#   Metadata: Read (otomatik).  Public depo için ücret/limit yok.
#
# GÜVENLİK: Token yalnızca bu betiğin çalıştığı süreçte ortam değişkenidir;
# diske YAZILMAZ (.git/config'e düşmemesi için push URL'i geçici kullanılır).
# İşiniz bitince GitHub'da tokenı silin.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

REPO_SLUG="${1:-}"
VISIBILITY="${2:-public}"
: "${GITHUB_TOKEN:?HATA - once su komutu verin: export GITHUB_TOKEN=... (ayrinti DEPLOY.md)}"
: "${REPO_SLUG:?HATA - kullanim: ./setup_github.sh KULLANICI/DEPO [public|private]}"

OWNER="${REPO_SLUG%%/*}"
REPO="${REPO_SLUG##*/}"
API="https://api.github.com"
AUTH="Authorization: Bearer ${GITHUB_TOKEN}"
ACCEPT="Accept: application/vnd.github+json"

api() {  # api <METHOD> <path> [json-body]
  local method="$1" path="$2" body="${3:-}"
  if [ -n "$body" ]; then
    curl -sS -X "$method" -H "$AUTH" -H "$ACCEPT" -H "X-GitHub-Api-Version: 2022-11-28" \
         -d "$body" "$API$path"
  else
    curl -sS -X "$method" -H "$AUTH" -H "$ACCEPT" -H "X-GitHub-Api-Version: 2022-11-28" "$API$path"
  fi
}
code_of() { curl -s -o /dev/null -w "%{http_code}" -H "$AUTH" -H "$ACCEPT" "$API$1"; }

echo "== 1/6 Depo kontrolü: ${REPO_SLUG}"
if [ "$(code_of "/repos/${REPO_SLUG}")" = "200" ]; then
  echo "   depo zaten var"
else
  echo "   depo oluşturuluyor (${VISIBILITY})…"
  if [ "$OWNER" = "$(api GET /user | python3 -c 'import json,sys; print(json.load(sys.stdin).get("login",""))')" ]; then
    api POST /user/repos "{\"name\":\"${REPO}\",\"private\":$([ "$VISIBILITY" = public ] && echo false || echo true),\"auto_init\":false}" \
      | python3 -c 'import json,sys; d=json.load(sys.stdin); print("   oluşturuldu:", d.get("full_name") or d.get("message"))'
  else
    api POST "/orgs/${OWNER}/repos" "{\"name\":\"${REPO}\",\"private\":$([ "$VISIBILITY" = public ] && echo false || echo true),\"auto_init\":false}" \
      | python3 -c 'import json,sys; d=json.load(sys.stdin); print("   oluşturuldu:", d.get("full_name") or d.get("message"))'
  fi
fi

echo "== 2/6 Kod gönderiliyor (main)…"
if [ ! -d .git ]; then git init -b main -q; fi
git add -A
if ! git diff --staged --quiet; then
  git -c user.name="${GIT_AUTHOR_NAME:-evrimsel-lab}" -c user.email="${GIT_AUTHOR_EMAIL:-lab@example.com}" \
      commit -q -m "kalıcı yayın kurulumu"
fi
if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "https://github.com/${REPO_SLUG}.git"
else
  git remote set-url origin "https://github.com/${REPO_SLUG}.git"
fi
# token yalnızca bu tek push çağrısında kullanılır, .git/config'e yazılmaz
PUSH_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${REPO_SLUG}.git"
if ! git push "$PUSH_URL" main:main 2>/dev/null; then
  echo "   uzak depo dolu (örn. README ile açılmış) — geçmişler birleştiriliyor…"
  git fetch "$PUSH_URL" main
  git merge --allow-unrelated-histories -X ours -m "uzak depo ile birleştirme" FETCH_HEAD || true
  git push "$PUSH_URL" main:main
fi
echo "   push tamam"

echo "== 3/6 GitHub Pages etkinleştiriliyor (kaynak: Actions)…"
st=$(code_of "/repos/${REPO_SLUG}/pages")
if [ "$st" = "200" ]; then
  api PUT "/repos/${REPO_SLUG}/pages" '{"build_type":"workflow"}' >/dev/null && echo "   Pages ayarı güncellendi (workflow)"
elif [ "$st" = "404" ]; then
  api POST "/repos/${REPO_SLUG}/pages" '{"build_type":"workflow"}' \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print("   Pages açıldı:", d.get("html_url") or d.get("message"))'
else
  echo "   (Pages durumu: HTTP $st — panelden elle açmanız gerekebilir)"
fi

echo "== 4/6 Actions izinleri (sonuçları depoya işleyebilsin)…"
api PUT "/repos/${REPO_SLUG}/actions/permissions/workflow" \
    '{"default_workflow_permissions":"write","can_approve_pull_request_reviews":true}' >/dev/null \
  && echo "   izinler: read/write (tamam)" \
  || echo "   (izin ayarı başarısız — panelden: Settings → Actions → Workflow permissions → Read and write)"

echo "== 5/6 İlk keşif koşusu tetikleniyor…"
api POST "/repos/${REPO_SLUG}/actions/workflows/evolve.yml/dispatches" \
    '{"ref":"main","inputs":{"preset":"hizli","atoms":"1e8"}}' >/dev/null \
  && echo "   tetiklendi: evolve.yml" || echo "   (tetiklenemedi; iş akışı eklendikten sonra tekrar deneyin)"

echo "== 6/6 Pages adresi ve koşu durumu"
PAGES_URL="https://${OWNER}.github.io/${REPO}/"
echo "   ADRES: ${PAGES_URL}"
echo "   (Actions sayfası: https://github.com/${REPO_SLUG}/actions)"
echo -n "   ilk yayın bekleniyor"
for i in $(seq 1 40); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "$PAGES_URL" || true)
  if [ "$code" = "200" ]; then echo " → YAYINDA ✔"; break; fi
  echo -n "."
  sleep 15
done
echo
echo "Tamamlandı. Site: ${PAGES_URL}"
echo "Bundan sonra: her 6 saatte bir otomatik koşu + yayın (Actions)."
