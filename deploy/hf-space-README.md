---
title: Evrimsel Atom Fiziği Laboratuvarı
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Filtreli evrimsel algoritma ile atom fiziğinin en ucuz formülleri
---

# Evrimsel Atom Fiziği Laboratuvarı (Hugging Face Space)

Bu Space, projenin **7/24 canlı Python sunucusudur**: filtreli evrimsel algoritma
(sembolik regresyon) gerçek sayısal Schrödinger çözücüsünden üretilen veriden atom
fiziğinin en ucuz matematiksel formlarını keşfeder; sonuçlar Hellmann–Feynman,
virial/eylem durağanlığı, ekstrapolasyon ve etiket-karıştırma (numeroloji bariyeri)
filtrelerinden geçmek zorundadır.

## Kullanım

Bu Space'i kendi hesabınıza kopyalayıp (Duplicate → Docker) çalıştırın.
Sunucu `0.0.0.0:8000` üzerinde dinler; `app_port: 8000` ayarı yukarıda tanımlıdır.

Varsayılan ortam değişkenleri Dockerfile içindedir:

| değişken | anlam | varsayılan |
|---|---|---|
| `AUTO_START` | açılışta koşuyu başlat | `1` |
| `AUTO_FOREVER` | koşular aralıksız sürsün | `1` |
| `AUTO_PRESET` | `hizli` / `standart` / `derin` | `hizli` |
| `AUTO_ATOMS` | gerçek koşulacak atom sayısı | `2e8` |
| `AUTO_INTERVAL` | turlar arası bekleme (s) | `20` |
| `EA_AUTO_EXPORT` | her koşudan sonra `docs/` statik yayınını tazele | `1` |

## Sayfalar

* `/` — canlı panel (evrim durumu, filtre sonuçları, trilyon ölçeği muhasebesi)
* `/methods` — yöntem, filtre zinciri, dürüstlük kuralları
* `/publish` — kalıcılık ve yeniden üretim kaydı
* `/healthz` — sağlık kontrolü (uyandırma pingleri için)
* `/api/status`, `/api/index`, `/api/run/<id>` — JSON API

## Notlar (dürüstlük)

* Referans veri kapalı formdan değil, sonlu farklar özdeğer çözümünden gelir ve
  ızgara yakınsaması + varyasyonel eylem durağanlığı ile doğrulanır.
* Trilyon atom ölçeği bir **topluluk** simülasyonudur (tek-atom çözümlerinin ortalaması);
  tam çok-cisimli kuantum durumu hesabı değildir. Süreler ölçülen ns/atom üzerinden
  doğrusal dışdeğerlemedir.
* Space'in diski geçicidir: kalıcı arşiv için GitHub deposundaki `data/` + `reports/` +
  `docs/` klasörleri (GitHub Actions iş akışı) kullanılmalıdır.
