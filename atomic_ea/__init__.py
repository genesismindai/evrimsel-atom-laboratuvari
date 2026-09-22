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
__all__ = ["expression", "physics", "filters", "evolution", "benchmarks", "throughput"]
