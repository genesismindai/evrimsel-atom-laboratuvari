# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 genesismindai (Evrimsel Atom Laboratuvarı)
#
# Bu program özgür yazılımdır: GNU Genel Kamu Lisansı (GPL) sürüm 3 veya
# sonraki sürümleri koşulları altında yeniden dağıtabilir ve/veya
# değiştirebilirsiniz. Ayrıntılar için LICENSE dosyasına bakın.
"""
atomic_ea — Atom fiziği için filtreli evrimsel sembolik regresyon çekirdeği.

Amaç
----
Pahalı sayısal çözücülerin (Schrödinger özdeğer problemi, ızgara köşegenleştirme)
ürettiği *referans veriden*, en UCUZ matematiksel kapalı formu evrimsel algoritma
ile yeniden keşfetmek. Keşif, bir dizi "sert fizik filtresi"nden geçmek zorundadır;
böylece numerolojik tesadüfler (şanslı uydurma) elenir.

Bilimsel dürüstlük ilkeleri (kod içinde zorunlu kılınmıştır)
-----------------------------------------------------------
1. Referans veri, keşfedilecek formülden BAĞIMSIZ üretilir (sayısal PDE çözümü),
   ızgara yakınsaması ve varyasyonel durağanlık ile doğrulanır  -> physics.py
2. Eğitim / doğrulama / sınav (test) kümeleri rejimlere göre ayrılır; sınav kümesi
   sadece raporlamada bir kez kullanılır -> benchmarks.py
3. Filtreler: boyut/ölçek simetrisi, ekstrapolasyon, Hellmann-Feynman, virial
   teoremi, eylem durağanlığı, numeroloji bariyeri (permütasyon null testi),
   donanım güvenliği ve maliyet (FLOP) bütçesi -> filters.py
4. Her sonuç yeniden üretilebilir: tohum (seed) loglanır, "frozen" artefakt yazılır.
"""

__version__ = "1.0.0"
__license__ = "GPL-3.0-or-later"
__copyright__ = "Copyright (C) 2026 genesismindai (Evrimsel Atom Laboratuvarı)"
__homepage__ = "https://genesismindai.github.io/evrimsel-atom-laboratuvari/"
__all__ = ["expression", "physics", "filters", "evolution", "benchmarks", "throughput"]
