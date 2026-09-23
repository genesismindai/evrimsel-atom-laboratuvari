# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 genesismindai (Evrimsel Atom Laboratuvarı)
#
# Bu program özgür yazılımdır: GNU Genel Kamu Lisansı (GPL) sürüm 3 veya
# sonraki sürümleri koşulları altında yeniden dağıtabilir ve/veya
# değiştirebilirsiniz. Ayrıntılar için LICENSE dosyasına bakın.
"""
physics.py — Referans veri üreteci: gerçek sayısal kuantum çözücü.

Bu modül, evrimsel algoritmanın "ezberlemesini" engelleyen temel taştır:
eğitim verisi hiçbir kapalı formülden gelmez; radyal Schrödinger denklemi
sonlu farklar ızgarasında köşegenleştirilerek üretilir ve şu bağımsız
yollarla doğrulanır:

  * Izgara yakınsaması (h ve h/2 -> Richardson ekstrapolasyonu)
  * Varyasyonel eylem durağanlığı  E_action(lambda) = <H>(lambda),
    <H>(lambda) = lambda^2 <T> + lambda <V>,  d<H>/dlambda|_lam=1 ~ 0
  * Virial teoremi  2<T> + <V> ~ 0

Atomik birimler (a0 = 1, hbar = 1, m_e = 1, 1 Hartree = 27.2114 eV) kullanılır.

Denklem (radyal, u(r) = r R(r)):

    [ -1/2 d^2/dr^2 + l(l+1)/(2 r^2) - Z/r ] u(r) = E u(r),   u(0)=u(rmax)=0
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh_tridiagonal

HARTREE_TO_EV = 27.211386245988
BOHR_TO_PM = 52.9177210903


@dataclass
class Solution:
    """Tek bir (Z, l) çözümünün sayısal sonuçları."""
    Z: float
    l: int
    h: float
    r: np.ndarray            # ızgara noktaları (a0)
    E: np.ndarray            # özdeğerler (Hartree)
    u: np.ndarray            # normalize radyal fonksiyonlar, u[:, k]
    rho: np.ndarray          # |u_k|^2 olasılık yoğunluğu
    T: np.ndarray            # <T> kinetik beklenti değeri
    V: np.ndarray            # <V> potansiyel beklenti değeri
    inv_r: np.ndarray        # <1/r>
    r_exp: np.ndarray        # <r>
    norm_residual: np.ndarray
    action_residual: np.ndarray   # d<H>/dlambda | lam=1 = 2<T> + <V>  (~0 olmalı)
    virial_ratio: np.ndarray      # -<V>/<T> (~2 olmalı)

    def level(self, n: int):
        """n = l + 1 + k  (radyal düğüm sayısı k = n-l-1)."""
        k = n - self.l - 1
        if k < 0 or k >= len(self.E):
            raise IndexError(f"n={n}, l={self.l} için çözüm yok (k={k})")
        return {
            "E": float(self.E[k]),
            "T": float(self.T[k]),
            "V": float(self.V[k]),
            "inv_r": float(self.inv_r[k]),
            "r": float(self.r_exp[k]),
            "action_residual": float(self.action_residual[k]),
            "virial_ratio": float(self.virial_ratio[k]),
            "norm_residual": float(self.norm_residual[k]),
        }


def _simpson_weights(n: int, h: float) -> np.ndarray:
    """Bileşik Simpson ağırlıkları (n tek sayıda nokta varsayımıyla)."""
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w * h / 3.0


def solve_hydrogenic(
    Z: float = 1.0,
    l: int = 0,
    n_states: int = 4,
    h: float = 0.01,
    r_max: float = 160.0,
) -> Solution:
    """
    Hidrojenik atomun radyal denklemini sonlu farklar ile çözer.

    Tridiagonal matris:  H = (1/h^2) I_tri  - (1/(2h^2)) offdiag + V(r)
    En düşük `n_states` özdeğer/özvektör seçilerek alınır.
    """
    N = int(round(r_max / h))
    if N % 2 == 0:           # Simpson için tek sayı iyi
        N += 1
    r = (np.arange(1, N + 1)) * h                      # r = h ... N*h
    V = -Z / r + l * (l + 1) / (2.0 * r**2)

    d = 1.0 / h**2 + V                                 # köşegen
    e = np.full(N - 1, -0.5 / h**2)                    # alt/üst köşegen
    k = min(n_states, max(2, N // 4))
    E, U = eigh_tridiagonal(d, e, select="i", select_range=(0, k - 1))

    w = _simpson_weights(N, h)

    # Normalizasyon: int u^2 dr = 1
    for j in range(U.shape[1]):
        nn = float(np.sum(w * U[:, j] ** 2))
        U[:, j] /= np.sqrt(abs(nn))
    rho = U**2

    Vexp = np.array([float(np.sum(w * rho[:, j] * V)) for j in range(U.shape[1])])
    # Kinetik enerji: ayrık özdeğer özdeşliği  <u|H|u> = E  =>  <T> = E - <V>.
    # (Izgara türeviyle hesaplamak O(h^2) hata getirir ve eylem artığını bozar;
    #  bu özdeşlik ayrık problemde TAMDIR, süreklilik limitinde 2<T>+<V> -> 0.)
    T = np.array([float(E[j]) - float(Vexp[j]) for j in range(U.shape[1])])
    du = np.gradient(U, h, axis=0)   # yalnızca teşhis amaçlı saklanır
    inv_r = np.array([float(np.sum(w * rho[:, j] / r)) for j in range(U.shape[1])])
    r_exp = np.array([float(np.sum(w * rho[:, j] * r)) for j in range(U.shape[1])])

    norm_res = np.array([float(np.sum(w * rho[:, j])) for j in range(U.shape[1])]) - 1.0
    action_res = 2.0 * T + Vexp                     # d<H>/dlam @ lam=1
    with np.errstate(divide="ignore", invalid="ignore"):
        virial = -Vexp / T

    return Solution(
        Z=float(Z), l=int(l), h=float(h), r=r, E=E, u=U, rho=rho,
        T=T, V=Vexp, inv_r=inv_r, r_exp=r_exp,
        norm_residual=norm_res, action_residual=action_res,
        virial_ratio=virial,
    )


def richardson_energy(Z: float, l: int, k: int, h: float = 0.02,
                      r_max: float = 160.0) -> dict:
    """
    Izgara yakınsama kanıtı: aynı özdeğeri h ve h/2 üzerinde hesaplayıp
    ikinci mertebe Richardson ekstrapolasyonu uygular.

    Döndürür: {'E_h':..., 'E_h2':..., 'E_extrap':..., 'delta':...}
    """
    s1 = solve_hydrogenic(Z=Z, l=l, n_states=k + 1, h=h, r_max=r_max)
    s2 = solve_hydrogenic(Z=Z, l=l, n_states=k + 1, h=h / 2.0, r_max=r_max)
    E1, E2 = float(s1.E[k]), float(s2.E[k])
    E_ex = (4.0 * E2 - E1) / 3.0
    return {"E_h": E1, "E_h2": E2, "E_extrap": E_ex, "delta": abs(E1 - E2),
            "richardson_shift": abs(E2 - E_ex)}


def solve_yukawa(
    Z: float,
    alpha: float,
    l: int = 0,
    n_states: int = 2,
    h: float = 0.02,
    r_max: float = 160.0,
) -> Solution:
    """
    Perdeli (Yukawa) potansiyel:   V(r) = -Z e^{-alpha r} / r

    Bu problem İÇİN KAPALI FORM YOKTUR. Bu yüzden evrimsel algoritmanın burada
    bulacağı şey bir "ezber" değil, sayısal çözücünün pahalı çıktısını taklit
    eden ucuz bir VEKİL MODEL (surrogate)'dir. Dürüstlük gereği: bu bir
    keşif değil, sıkıştırmadır; geçerlik bölgesi ile birlikte raporlanır.
    """
    N = int(round(r_max / h))
    if N % 2 == 0:
        N += 1
    r = (np.arange(1, N + 1)) * h
    V = -Z * np.exp(-alpha * r) / r + l * (l + 1) / (2.0 * r**2)
    d = 1.0 / h**2 + V
    e = np.full(N - 1, -0.5 / h**2)
    k = min(n_states, max(2, N // 4))
    E, U = eigh_tridiagonal(d, e, select="i", select_range=(0, k - 1))
    w = _simpson_weights(N, h)
    for j in range(U.shape[1]):
        nn = float(np.sum(w * U[:, j] ** 2))
        U[:, j] /= np.sqrt(abs(nn))
    rho = U**2
    Vexp = np.array([float(np.sum(w * rho[:, j] * V)) for j in range(U.shape[1])])
    T = np.array([float(E[j]) - float(Vexp[j]) for j in range(U.shape[1])])
    inv_r = np.array([float(np.sum(w * rho[:, j] / r)) for j in range(U.shape[1])])
    r_exp = np.array([float(np.sum(w * rho[:, j] * r)) for j in range(U.shape[1])])
    norm_res = np.array([float(np.sum(w * rho[:, j])) for j in range(U.shape[1])]) - 1.0
    action_res = 2.0 * T + Vexp
    with np.errstate(divide="ignore", invalid="ignore"):
        virial = -Vexp / T
    return Solution(Z=float(Z), l=int(l), h=float(h), r=r, E=E, u=U, rho=rho,
                    T=T, V=Vexp, inv_r=inv_r, r_exp=r_exp,
                    norm_residual=norm_res, action_residual=action_res,
                    virial_ratio=virial)


def solver_self_test(verbose: bool = True) -> dict:
    """
    Çözücünün kendi kendini doğrulaması (evrimsel algoritmadan bağımsız).

    Döndürür: norm, eylem durağanlığı, virial ve ızgara yakınsaması ölçütleri.
    """
    out = {}
    s = solve_hydrogenic(Z=1.0, l=0, n_states=3, h=0.01)
    lv = s.level(1)
    out["norm_residual"] = abs(lv["norm_residual"])
    out["action_residual"] = abs(lv["action_residual"])
    out["virial_ratio"] = lv["virial_ratio"]
    ric = richardson_energy(1.0, 0, 0, h=0.02)
    out["grid_delta"] = ric["delta"]
    out["E_extrap"] = ric["E_extrap"]
    if verbose:
        print("[fizik] çözücü öz-testi")
        print(f"        |norm-1|            = {out['norm_residual']:.3e}")
        print(f"        |eylem artığı|      = {out['action_residual']:.3e}  (2<T>+<V>, 0 olmalı)")
        print(f"        virial -<V>/<T>     = {out['virial_ratio']:.6f}  (2 olmalı)")
        print(f"        ızgara farkı (h,h/2) = {out['grid_delta']:.3e}")
    return out


if __name__ == "__main__":
    solver_self_test()
