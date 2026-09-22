# Evrimsel Atom Fiziği Laboratuvarı

Filtreli evrimsel algoritma (ada modeli sembolik regresyon) ile **atom fiziğinin en ucuz
matematiksel formlarını**, *gerçek sayısal Schrödinger çözücüsünden* üretilen veriden yeniden
keşfeden ve keşfi **trilyon ölçeğinde atom değerlendirmesine** ölçekleyen açık sistem.

## Neden "hile yok, numeroloji yok"?

| Risk | Bu projedeki önlem |
|---|---|
| Veriye kapalı formu uydurma ("numeroloji") | Referans veri hiçbir kapalı formdan gelmez; radyal Schrödinger denklemi sonlu farklar ızgarasında köşegenleştirilir |
| Uydurmanın fark edilmemesi | **Etiket-karıştırma (permütasyon) bariyeri**: aynı mimari fiziksel anlamı olmayan karışık etiketlerde çalıştırılır; elde edilen en iyi hatanın belirgin altına inmeyen aday elenir |
| Sızıntı | Eğitim / doğrulama / **sınav** kümeleri rejimlere göre ayrıdır; sınav kümesi yalnızca nihai raporda bir kez ölçülür |
| Yalnızca eğri uydurma | **Hellmann–Feynman** (dE/dZ = −⟨1/r⟩) ve **virial/eylem durağanlığı** (E = −(Z/2)⟨1/r⟩, 2⟨T⟩+⟨V⟩→0) bağımsız sayısal referansa karşı sınanır |
| Aşırı uyum (gereksiz basamaklar) | Uygunluk fonksiyonu **ölçüm gürültüsü tabanı** ile sınırlanır; hata tabanın altına indiğinde seçim yalnızca **daha ucuz** formu arar (MDL) |
| Pahalı/güvenilmez formüller | F1 filtresi transandantal operatörleri (exp/log/trig) ürün formüllerden tamamen eler; maliyet modeli işlem birimleriyle ölçülür |
| Donanım hasarı / bellek patlaması | Değer sınırları, NaN/inf denetimi, |sabit|≤10⁶, süre bütçesi (1000 nokta ≤ 400 µs); ölçeklendirme **öbek akışıyla** yapılır, bellek O(öbek) kalır |

## Dürüstçe sınırlar (iddia DEĞİL)

* 10¹² atomun **tam çok-cisimli kuantum durumu** hesaplanmaz (durum uzayı 2^N — imkânsız).
  Yapılan: her atomun kendi (Z, n) durumunda, sayısal çözücünün yerine geçen ucuz kapalı formla
  değerlendirilmesi → **tek-atom çözümünün trilyon ölçeğinde topluluk ortalaması**.
* Trilyon ölçeği süreleri **doğrusal dışdeğerlemedir**; gerçek ölçüm beyan edilen atom sayısında yapılır.
* Joule tahminleri ilan edilen bir J/işlem varsayımına dayanır.
* Yukawa (perdeli Coulomb) benchmark'ında kapalı form **yoktur**; orada yapılan iş *keşif* değil,
  sayısal çözücünün ucuz **vekil (surrogate) sıkıştırmasıdır** ve böyle etiketlenir.

## Mimari

```
atomic_ea/
  physics.py      sayısal referans çözücü (sonlu farklar özdeğer) + öz-testler
  expression.py   sembolik ağaçlar, güvenli derleme, FLOP maliyet modeli
  filters.py      filtre zinciri F1–F8, kaskad değerlendirme, uygunluk (gürültü tabanlı)
  benchmarks.py   veri kümeleri (train/val/test) ve fizik denetimleri
  evolution.py    ada modeli EA, mutasyon/çaprazlama, Gauss–Newton sabit ayarı,
                  MDL sabit yuvarlama, arşiv, onur listesi, null kalibrasyonu
  throughput.py   trilyon ölçeği muhasebesi (ölçüm + projeksiyon + pahalı yol karşılaştırması)
runner.py         koşu orkestrasyonu, kalıcı kayıt, Markdown/HTML rapor
runner_cli.py     komut satırı arayüzü + kesintisiz mod (--forever)
export_static.py  kalıcı statik yayın üreticisi (docs/; veri gömülü, sunucu gerekmez)
publish_artifacts.sh  statik üret → commit → push yardımcısı
deploy/           Dockerfile (HF Spaces), render.yaml, HF Space README
.github/workflows/evolve.yml  zamanlanmış Python koşusu + GitHub Pages yayını
DEPLOY.md         kalıcı ve ücretsiz yayın kılavuzu (adım adım)
web/server.py     bağımlılıksız (stdlib) yayın sunucusu + SSE + API
web/*.html        panel, yöntem, yayın kaydı sayfaları
data/             referans veri, koşu kayıtları, canlı ilerleme, yayın kaydı
reports/          insan-okur raporlar (md + html)
```

## Kalıcı yayın (sunucu düşse de site ayakta)

Geliştirme ortamı (sandbox) oturumla kapanır; bu yüzden kalıcılık bulutta kurulur:

* **Yol A — GitHub Actions + Pages (önerilen, sonsuz ve ücretsiz):**
  Python kodu GitHub koşucularında zamanlanır, sonuçlar depoya işlenir, `docs/` statik
  sitesi GitHub Pages'te kalıcı olarak yayınlanır. Kurulum: **`DEPLOY.md`** ve hazır iş
  akışı: `.github/workflows/evolve.yml`.
* **Yol B — 7/24 canlı Python sunucusu (ücretsiz):** `deploy/Dockerfile` (Hugging Face
  Spaces) veya `deploy/render.yaml` (Render). `AUTO_START=1` + `AUTO_FOREVER=1` ile
  açılışta kesintisiz keşif başlar.
* **Statik site üretimi:** `python3 export_static.py --out docs` → tüm veri gömülü tek
  dosyalık site (ağ çağrısı yapmaz, sunucu gerektirmez).
* **Yayınlama yardımcısı:** `./publish_artifacts.sh "mesaj"` → statik siteyi üretir,
  sonuçları commit'leyip push eder.

## Koşular

```bash
python3 runner_cli.py --preset hizli --seed 7          # hızlı koşu (+ docs/ otomatik tazelenir)
python3 runner_cli.py --preset standart --seed 1       # standart
python3 runner_cli.py --forever --interval 60          # kesintisiz keşif modu (touch data/STOP ile durur)
python3 export_static.py --out docs                    # kalıcı statik yayın
python3 web/server.py                                  # canlı yayın paneli (port 8000)
```

Paneller: `/` canlı evrim · `/methods` yöntem ve dürüstlük kuralları · `/publish` kalıcı yayın kaydı

## Şu ana kadar keşfedilenler (örnek)

| benchmark | keşfedilen form | maliyet | sınav hatası (ekstrapolasyon) | filtreler |
|---|---|---|---|---|
| Hidrojenik enerji E(Z,n) | `−0.5·(Z/n)²` | 5 birim | 3.2e-04 | F1–F8 ✅ |
| ⟨1/r⟩(Z,n) | koşuda ölçülür | — | — | — |
| Etkin potansiyel V_eff | koşuda ölçülür | — | — | — |
| Yukawa vekili | koşuda ölçülür | — | — | — |

> Enerji yasası bilinen kapalı forma (Rydberg) karşılık gelir: `E = −Z²/(2n²) Hartree`.
> Bu, "keşfin doğru olduğunun" bağımsız göstergesidir — ama veri o formülden üretilmedi,
> evrim onu sayısal çözücü verisinden buldu ve filtreler doğruladı.
