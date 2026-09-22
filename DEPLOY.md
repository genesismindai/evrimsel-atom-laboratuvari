# KALICI YAYIN — kurulum kılavuzu (ücretsiz, sunucu düşse de ayakta)

## Sorun neydi?

Geliştirme ortamı (**sandbox**) oturum kapanınca durur. İçinde çalışan web sunucusu
yok olur; bu yüzden bağlantı düşer. **Sandbox içinde çalışan hiçbir süreç kalıcı olamaz.**
Kalıcılık, kodu bu kutu dışına, ücretsiz bir buluta taşımakla mümkündür.

Bu projede iki tamamlayıcı yol hazırdır:

| | **Yol A — GitHub (önerilen)** | **Yol B — Hugging Face / Render** |
|---|---|---|
| Ne çalışır | Python kodu **GitHub Actions**'te zamanlanır (her 6 saatte bir koşu) | Python sunucusu **7/24** ayakta (canlı panel) |
| Site | **GitHub Pages** (statik, kalıcı, ücretsiz `*.github.io`) | Aynı sunucunun sayfaları |
| Kalıcılık | **Sonsuz**: her koşu git geçmişine işlenir | Disk geçici; kalıcı arşiv için Yol A ile birlikte kullanın |
| Ücret | Ücretsiz (herkese açık depo) | Ücretsiz katman (uyku modu olabilir) |
| Kurulum | ~5 dakika | ~3 dakika |

En sağlam düzen: **ikisini birlikte** kullanmak → site hiç düşmez (Yol A), canlı panel de olur (Yol B).

---

## Yol A — GitHub Pages + Actions (kalıcı, sonsuz, ücretsiz)

Python kodu GitHub'ın ücretsiz koşucularında koşar; sonuçlar depoya işlenir ve
`docs/` klasörü GitHub Pages'te **kalıcı** olarak yayınlanır. Sunucu gerekmez.

1. **Depo oluşturun.** GitHub'da `evrimsel-atom-laboratuvari` adında **Public** bir depo açın.
2. **Kodu yükleyin.** Bu proje klasöründe:
   ```bash
   git init
   git add -A
   git commit -m "ilk kayıt"
   git branch -M main
   git remote add origin https://github.com/<KULLANICI>/<DEPO>.git
   git push -u origin main
   ```
3. **Pages'i açın.** Depo → *Settings → Pages* → **Source: GitHub Actions**.
4. **İş akışı hazır.** `.github/workflows/evolve.yml` dosyası şunları yapar:
   * her 6 saatte bir (veya elle) evrimsel keşif koşusu (Python + numpy/scipy),
   * sonuçları `data/`, `reports/` klasörlerine işler (git geçmişi = sonsuz arşiv),
   * `docs/` statik sitesini yeniden üretir ve **Pages'e yayınlar**.
5. **Elle çalıştırma:** *Actions → Keşif koşusu + kalıcı yayın → Run workflow*
   (ön ayar ve atom sayısı seçilebilir).
6. **Adresiniz:** `https://<KULLANICI>.github.io/<DEPO>/` — bu adres **sonsuza kadar**
   çalışır; sunucu, oturum, bilgisayar kapansa da.

Yerelde aynısını yapmak için:
```bash
python3 runner_cli.py --preset hizli --seed 42      # koşu (sonunda docs/ otomatik tazelenir)
python3 export_static.py --out docs                 # statik siteyi yeniden üret
./publish_artifacts.sh "keşif koşusu 42"            # commit + push (Pages güncellenir)
```

**Kesintisiz keşif** (kendi bilgisayarınızda veya bir bulut kabuğunda):
```bash
python3 runner_cli.py --forever --preset hizli --interval 60 --atoms 5e8
# durdurmak için: Ctrl+C  veya  touch data/STOP
```

---

## Yol B1 — Hugging Face Spaces (7/24 canlı Python sunucusu, ücretsiz)

1. <https://huggingface.co> hesabı açın → **New Space**.
2. **SDK: Docker** (boş şablon), isim: `evrimsel-atom-laboratuvari`.
3. Bu projedeki `deploy/Dockerfile` dosyasını kullanın; Space deposuna tüm projeyi yükleyin
   (veya Space'i Git ile klonlayıp `git push`).
4. `deploy/hf-space-README.md` içeriğini Space'in `README.md` dosyası olarak kullanın
   (başlıktaki `app_port: 8000` ayarı sunucunun portunu bildirir).
5. Space açılır açılmaz `AUTO_START=1` sayesinde koşular başlar; `AUTO_FOREVER=1` ile
   kesintisiz sürer. Sayfalar: `/` canlı panel, `/methods`, `/publish`, `/healthz`.

Not: Space diski geçicidir; kalıcı arşiv için Yol A'yı da kurun (Actions → Pages).

---

## Yol B2 — Render (ücretsiz web servisi + uyku)

1. <https://render.com> → **New → Blueprint** → depoyu bağlayın (`deploy/render.yaml` okunur).
2. Ücretsiz katman: hareketsizlikte uyur, ilk istekte 40-60 s'de uyanır.
3. **Uyanık kalsın:** <https://cron-job.org> (ücretsiz) ile her 10 dakikada bir
   `https://<servis>.onrender.com/healthz` adresini pingletin.

---

## Yol C — Sadece statik site (Python yok): Netlify / Cloudflare Pages

Statik sitemiz (`docs/`) sunucu gerektirmez; `docs/index.html` içine **tüm veri gömülüdür**
(ağ çağrısı yapmaz). Netlify/Cloudflare Pages'e `docs/` klasörünü sürükleyip bırakmanız
yeterlidir; Python koşuları için yine Yol A (Actions) kullanılır.

---

## Kalıcılık matrisi (ne nerede saklanır?)

| içerik | yol | kalıcı mı? |
|---|---|---|
| koşu kayıtları (tohum, formüller, filtreler) | `data/runs/*.json` | ✅ git geçmişinde |
| raporlar (md + html) | `reports/` | ✅ git geçmişinde |
| statik site | `docs/` | ✅ Pages'te yayında |
| referans veri + çözücü doğrulama özeti | `data/reference.npz`, `reference_meta.json` | ✅ |
| canlı durum (anlık) | `data/progress.json` | ⚠️ yalnızca çalışırken |
| sandbox sunucusu | `web/server.py` | ❌ oturumla kapanır (beklenen) |

## Sık sorunlar

* **“Site açılmıyor”** → Yol A kurulduysa sunucu gerekmiyor; adres `https://<kullanıcı>.github.io/<depo>/`.
  Pages kaynağını **GitHub Actions** yaptığınızdan emin olun.
* **Actions başarısız: push reddedildi** → depo ayarlarında *Workflow permissions*:
  **Read and write permissions** seçin.
* **İlk koşu uzun sürüyor** → `derin` ön ayar yerine `hizli` ile başlayın (≈2,5 dakika);
  `--atoms` değerini düşürün (örn. `1e7`).
* **Panelde satır içi önizleme boş** → `docs/index.html` tek dosyalıktır; doğrudan
  tarayıcıda açın ya da Pages'ten görüntüleyin.
