# Evrimsel Atom Fiziği Laboratuvarı

**🔴 Canlı kalıcı yayın: <https://genesismindai.github.io/evrimsel-atom-laboratuvari/>**

![Lisansa bak](https://img.shields.io/badge/lisans-GPL--3.0--or--later-blue.svg)

**Lisans: [GPL-3.0-or-later](LICENSE)** — özgür yazılım: kullanabilir, inceleyebilir, değiştirebilir ve
dağıtabilirsiniz; türev çalışmaların kaynak kodu da aynı lisansla açık kalmalıdır.
(saatte bir otomatik koşu · gece standart koşu · her koşu arşive işlenir — sunucu gerekmez)

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
| Moleküler bağda "eğri uydurma" | **F9 türev tutarlılığı**: kuvvet ve enerji İKİ AYRI evrim koşusunda bağımsız keşfedilir; kuvvet adayının −dE/dR'si öteki keşfin (enerji şampiyonunun) türeviyle karşılaştırılır — yalnız mekanikten gelir, uydurmayla geçilemez |
| Çok elektronluda "N ve e–e itmesini silme" hilesi | **Kuantum jürisi aramada zorunlu**: F5 (Hellmann–Feynman), **F10** (örtük e–e itmesi `V_ee = 2E − Z·∂E/∂Z`, min V_ee > 0) ve **F11** (elektron sayısı tepkisi `ΔE(2→4)`) ihlalinde aday elenir; moleküler tarafta F6 (genelleştirilmiş virial) ve F9 (kuvvet tutarlılığı) aynı rejimde |
| Ölçek yasasını "tesadüfen" tutturmak | F2 filtresi eşzamanlı dönüşümü (R→R/λ, Z→λZ) ve asimptotik üssü (Z→∞ için 2) sınar; yasa **veriden bağımsız** olarak makine hassasiyetinde doğrulanır (7.3×10⁻¹⁵) |
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
  filters.py      filtre zinciri F1–F9, kaskad değerlendirme, uygunluk (gürültü tabanlı)
  benchmarks.py   veri kümeleri (train/val/test) ve fizik denetimleri
  gto.py          Gauss tabanlı HF çözücü (Boys F0, örtüşme/kinetik/nükleer/ERI, damped SCF,
                  dondurulmuş çekirdek Li-benzeri, H2 ve H2+ enerji eğrileri) + öz-testler
  hbond.py        çok elektronlu atom serisi, H2 bağ eğrisi + kuvvet alanı, H2+ ölçek yasası
                  referansları ve dört yeni benchmark (bkz. "Kapsam")
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

## Kapsam: tek elektronlu atom → çok elektronlu atom, bağ, kuvvet alanı

| benchmark | hedef | referans |
|---|---|---|
| `enerji`, `ters_yaricap`, `etkin_potansiyel`, `yukawa` | hidrojenik aile (kapalı form) | sonlu farklar Schrödinger çözücüsü |
| `cok_elektron` | E(Z,N) — kapalı kabuk 2e/4e serileri, Z≤10 | kendi GTO/HF çözücümüz (üsler varyasyonel optimize; 2e serisi analitik 1/Z açılımına karşı ≤5×10⁻⁴ @Z=10) |
| `bag_h2` | E_H2(R) kimyasal bağ eğrisi | iki merkezli GTO/HF |
| `kuvvet_h2` | F(R) = −dE/dR | bağımsız merkezi fark türevi |
| `h2plus_olcek` | ε(R,Z) elektronik enerji + **tam ölçek yasası** | ε(R,Z) = Z²·f(ZR); özdeşlik 7.3×10⁻¹⁵ doğrulukla sınanır |

Açıkça belirtilen sınırlar: H2 eğrisi R≤3 a₀ (RHF ayrışma kuyruğu yapay yükselir); Li-benzeri açık
kabuk (N=3) serisi taban sınırı nedeniyle benchmark'a alınmadı; He/Be literatür değerleri yalnız teşhis.

## Kuantum koruma (çok elektronlu atom + moleküler bağ)

Ölçülen başarısızlık: yalnız veriye bakan EA, çok elektronlu atomda elektron sayısını ve e–e itmesini
formülden **silerek** ucuz ama imkânsız formüller üretiyordu (47. koşu şampiyonu `(0.5 − Z)·Z`).
Kuantum teoremleri artık yalnız "doğrulama" değil, **arama düzeyinde yaptırım**:

| jüri | yasa | tolerans | benchmark |
|---|---|---|---|
| F5 | Hellmann–Feynman `∂E/∂Z = −⟨Σ1/r⟩` | 5×10⁻² | çok elektronlu |
| F10 | örtük e–e itmesi `V_ee = 2E − Z·∂E/∂Z` (ve `min V_ee > 0`) | 5×10⁻² | çok elektronlu |
| F11 | elektron sayısı tepkisi `f(Z,4) − f(Z,2)` | 5×10⁻² | çok elektronlu |
| F6 | genelleştirilmiş virial `T = −E − R·dE/dR` | 2×10⁻² | H₂ bağı |
| F9 | kuvvet tutarlılığı `dE/dR = −F_ref` | 3×10⁻² | bağ / kuvvet / ölçek |

Üç katmanlı yaptırım:

1. **Jüri-farkında sabit ayarı** — her adayın sabitleri hem veriye hem jüri kısıtlarına göre
   Gauss–Newton ile ayarlanır (kısıtlar formülün kendi tahminleri üzerinde doğrusal sonlu-fark
   operatörleri; referanslar bağımsız çözücüden, **yalnız eğitim+doğrulama** noktalarında).
   Yani EA kuantum yasasını *bilir*, sadece cezalandırılmaz.
2. **Kademeli ceza** — onarım sonrası ihlal sürerse `+0.15·(v−1)`, `v = max(hata/tolerans)`.
3. **Ağır ihlalde ∞** — `v > 30` ise aday anında ölür.

Doğrulama (hizli preset, seed 33, uçtan uca koşu): eski şampiyonlar **elendi**, yeni şampiyonlar
**DOĞRULANDI=True** —

| benchmark | yeni şampiyon | jüri | değer |
|---|---|---|---|
| çok elektronlu atom | izoelektronik aile `−(0.748+0.126N)Z² + (−0.358+0.492N)Z − 0.430N + 0.747` | F5 / F10 / F11 | 1.5×10⁻⁴ / 1.5×10⁻³ / 1.1×10⁻³ |
| H₂ bağı | rasyonel (Padé) aile | F6 / F9 | ✓ / 9.0×10⁻³ |
| H₂⁺ ölçek | `Z²·(−9.42 − 0.930·ZR − 0.311·(ZR)²)/(4.37 + 2.47·ZR + 0.548·(ZR)²)` | F9 (F5 rapor) | 2.9×10⁻² |

Ek düzeltme: F2’nin asimptotik ölçek kontrolü artık **Z→∞ limitini** (Z = 16…32) sınar; alt-başat
terimler sonlu Z’de üssü kaydırdığı için doğru izoelektronik aile haksız eleniyordu.

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

## Lisans

Bu proje **GNU Genel Kamu Lisansı sürüm 3 (veya sonraki sürümleri)** ile lisanslanmıştır —
tam metin: [`LICENSE`](LICENSE).

```
Copyright (C) 2026 genesismindai (Evrimsel Atom Laboratuvarı)

Bu program özgür yazılımdır: GNU Genel Kamu Lisansı (GPL) sürüm 3 veya sonraki
sürümleri koşulları altında yeniden dağıtabilir ve/veya değiştirebilirsiniz.
Ayrıntılar için LICENSE dosyasına bakın.
```

Ne anlama gelir (kısa):

* **Kullanma/inceleme/değiştirme/dağıtma** serbesttir — bilimsel şeffaflık bunu gerektirir:
  keşfedilen formüller, filtreler ve referans veri üretimi denetlenebilir kalır.
* **Copyleft:** kodu alıp değiştirerek dağıtırsanız (veya bir hizmette sunarsanız),
  türev çalışmanın kaynak kodu da **aynı GPL-3.0 koşullarıyla** açık olmalıdır.
* **Garanti yoktur:** GPL-3.0 §15–17 uyarınca yazılım "olduğu gibi" sunulur; bilimsel
  sonuçların doğruluğu için kendi doğrulamanızı yapın (proje bunun için `solver_self_test`
  ve bağımsız filtre zincirini içerir).
* Kaynak dosyalar başlarında **SPDX** tanımlayıcısı taşır: `SPDX-License-Identifier: GPL-3.0-or-later`.
