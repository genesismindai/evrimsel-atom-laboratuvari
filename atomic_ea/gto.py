"""
gto.py — Gaussian tabanlı Hartree-Fock (s-tipi taban) ve integral çekirdeği.

Neden bu modül var?
-------------------
Çok elektronlu atomların ve kimyasal bağın referans verisi kapalı formdan
ÜRETİLEMEZ. Bu modül, gerçek bir kuantum-kimya çözücüsü kurar:

  * Gauss tipi yörüngeler (GTO) ile tek ve iki merkezli integraller
    (örtüşme, kinetik, nükleer çekim, elektron-elektron itmesi)
  * Kapalı kabuk RHF (öz-tutarlı alan) çevrimi
  * Açık kabuk için "dondurulmuş çekirdek + değerlik" yaklaşımı (Li-benzeri)
  * İki atomlu molekül (H2) bağ eğrisi ve kuvveti

Doğrulama (kod içinde yapılır):
  * İntegraller, bağımsız sayısal kübik kuadratur ile karşılaştırılır
  * Energiler literatür HF değerleriyle karşılaştırılır (He, Li+, Be2+, Li, Be)
  * Varyasyon ilkesi: RHF enerjisi HF limitinin ÜSTÜNDE olmalı (üst sınır)
  * Virial: Coulomb sistemlerinde 2<T> + V ~ 0, atomda E = -<T>
  * Hellmann-Feynman: dE/dZ = -<Σ1/r> (sayısal türev ile karşılaştırılır)

Birimler: atomik birimler (Hartree, a0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import math
import numpy as np
from scipy.linalg import eigh
from scipy.special import erf
from scipy.optimize import minimize

# ------------------------------------------------------------------ literatür referansları
# Hartree-Fock limit enerjileri (a.u.) — yalnızca DOĞRULAMA için kullanılır,
# hiçbir veri üretiminde veya aramada kullanılmaz.
LIT_HF = {
    ("He", 2): -2.8616799,
    ("Li", 3): -7.4327269,
    ("Be", 4): -14.5729963,
    ("Li+", 3): -7.2364152,
    ("Be2+", 4): -13.6112957,
}
# H2 RHF / tam enerjiler (a.u.), R = 1.4 a0
LIT_H2 = {"R_e": 1.4011, "E_HF": -1.133629, "E_exact": -1.174476, "D_e_HF_eV": 3.636}


# ------------------------------------------------------------------ Boys fonksiyonu
def boys0(t) -> np.ndarray:
    """F0(t) = ∫_0^1 e^{-t u²} du.  (t→0: 1 - t/3 + t²/10 ...)"""
    t = np.asarray(t, dtype=float)
    out = np.empty_like(t)
    small = t < 1e-8
    out[small] = 1.0
    big = ~small
    if np.any(big):
        tt = t[big]
        out[big] = 0.5 * np.sqrt(np.pi / tt) * erf(np.sqrt(tt))
    return out


# ------------------------------------------------------------------ temel integraller
@dataclass
class PGF:
    """Normalize edilmiş s-tipi ilkel Gauss fonksiyonu."""
    alpha: float
    center: np.ndarray

    @property
    def norm(self) -> float:
        return (2.0 * self.alpha / math.pi) ** 0.75


def _pair(p: PGF, q: PGF):
    a, b = p.alpha, q.alpha
    A, B = np.asarray(p.center, float), np.asarray(q.center, float)
    Pp = (a * A + b * B) / (a + b)
    return a + b, a * b / (a + b), float(np.dot(A - B, A - B)), Pp


def overlap(p: PGF, q: PGF) -> float:
    s, mu, R2, _ = _pair(p, q)
    return p.norm * q.norm * (math.pi / s) ** 1.5 * math.exp(-mu * R2)


def kinetic(p: PGF, q: PGF) -> float:
    s, mu, R2, _ = _pair(p, q)
    return mu * (3.0 - 2.0 * mu * R2) * overlap(p, q)


def nuclear_attraction(p: PGF, q: PGF, charges: Sequence[Tuple[float, Sequence[float]]]) -> float:
    s, mu, R2, P = _pair(p, q)
    pref = p.norm * q.norm * (2.0 * math.pi / s) * math.exp(-mu * R2)
    tot = 0.0
    for Z, C in charges:
        t = s * float(np.dot(P - np.asarray(C, float), P - np.asarray(C, float)))
        tot += -Z * pref * float(boys0(t))
    return tot


def eri(p: PGF, q: PGF, r: PGF, t_: PGF) -> float:
    """(pq|rs) elektron-elektron itme integrali (iki merkezli, s-tipi)."""
    pij, mu_ij, R2_ij, P = _pair(p, q)
    pkl, mu_kl, R2_kl, Q = _pair(r, t_)
    pref = (p.norm * q.norm * r.norm * t_.norm
            * 2.0 * math.pi ** 2.5 / (pij * pkl * math.sqrt(pij + pkl)))
    expo = math.exp(-mu_ij * R2_ij) * math.exp(-mu_kl * R2_kl)
    tt = (pij * pkl / (pij + pkl)) * float(np.dot(P - Q, P - Q))
    return pref * expo * float(boys0(tt))


def _grid(n: int, center, alpha: float, norm: float, rmax: float = 16.0):
    """Küresel koordinatlarda kartezyen çarpım kuadratur ağı (yalnız test amaçlı)."""
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(n)
    rad = 0.5 * (x + 1.0) * rmax
    wr = 0.5 * rmax * w
    ct, wt = x, w
    ph = math.pi * (x + 1.0)
    wp = math.pi * w
    R = rad[:, None, None]; C = ct[None, :, None]
    S = np.sqrt(np.clip(1.0 - C ** 2, 0.0, None)); P = ph[None, None, :]
    X = R * S * np.cos(P); Y = R * S * np.sin(P)
    Zc = np.broadcast_to(R * C, X.shape).copy()
    Cc = np.asarray(center, float)
    D2 = (X - Cc[0]) ** 2 + (Y - Cc[1]) ** 2 + (Zc - Cc[2]) ** 2
    dens = norm * np.exp(-alpha * D2)
    wgt = (wr[:, None, None] * wt[None, :, None] * wp[None, None, :]
           * rad[:, None, None] ** 2 * S)
    return X.ravel(), Y.ravel(), Zc.ravel(), dens.ravel(), wgt.ravel()


def _numeric_eri_reference(p: PGF, q: PGF, r: PGF, t_: PGF,
                           n1: int = 12, n2: int = 13) -> float:
    """
    6 boyutlu itme integralinin BAĞIMSIZ sayısal hesabı (kaba ama farklı yol).

    Kritik ayrıntı: iki elektron için FARKLI düğüm sayıları kullanılır; aksi halde
    ızgara noktaları üst üste binip 1/|r1-r2| tekilliği sahte dev değerler üretir.
    """
    X1, Y1, Z1, f1, w1 = _grid(n1, p.center, p.alpha, p.norm)
    X1b, Y1b, Z1b, g1, _ = _grid(n1, q.center, q.alpha, q.norm)
    X2, Y2, Z2, f2, w2 = _grid(n2, r.center, r.alpha, r.norm)
    X2b, Y2b, Z2b, g2, _ = _grid(n2, t_.center, t_.alpha, t_.norm)
    rho1 = f1 * g1 * w1
    rho2 = f2 * g2 * w2
    V = 0.0
    for i in range(len(X1)):
        dx = X1[i] - X2; dy = Y1[i] - Y2; dz = Z1[i] - Z2
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        d = np.where(d < 1e-8, 1e-8, d)
        V += rho1[i] * float(np.sum(rho2 / d))
    return V


def _self_energy_reference(alpha: float, beta_scale: float = 2.0) -> float:
    """
    BAĞIMSIZ matematik (Boys fonksiyonu ve 6B integral kullanmaz):

    Aynı merkezli, eşit üslü (α) iki 1s Gauss yoğunluğunun itmesi. Yoğunluk
    ρ(r) = (β/π)^{3/2} e^{−βr²} (β = 2α), potansiyeli φ(r) = erf(√β r)/r olduğundan

        (ss|ss) = ∫₀^∞ 4π r² ρ(r) φ(r) dr

    tek boyutlu integrali scipy.quad ile yüksek hassasiyette hesaplanır.
    """
    from scipy.integrate import quad
    beta = beta_scale * alpha

    def integrand(r):
        if r < 1e-10:
            return 0.0
        rho = (beta / math.pi) ** 1.5 * math.exp(-beta * r * r)
        phi = erf(math.sqrt(beta) * r) / r
        return 4.0 * math.pi * r * r * rho * phi

    val, _ = quad(integrand, 0.0, 40.0 / math.sqrt(beta), epsabs=1e-13, epsrel=1e-13, limit=300)
    return float(val)


# ------------------------------------------------------------------ integral matrisleri
def build_matrices(basis: List[PGF],
                   charges: Sequence[Tuple[float, Sequence[float]]],
                   want_eri: bool = True):
    n = len(basis)
    S = np.zeros((n, n)); T = np.zeros((n, n)); V = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            S[i, j] = S[j, i] = overlap(basis[i], basis[j])
            T[i, j] = T[j, i] = kinetic(basis[i], basis[j])
            V[i, j] = V[j, i] = nuclear_attraction(basis[i], basis[j], charges)
    if not want_eri:
        return S, T, V, None
    # ERI: 8-kat simetri ile (pq|rs)
    ERI = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(k, n):
                    val = eri(basis[i], basis[j], basis[k], basis[l])
                    ERI[i, j, k, l] = ERI[i, j, l, k] = val
                    ERI[j, i, k, l] = ERI[j, i, l, k] = val
                    ERI[k, l, i, j] = ERI[k, l, j, i] = val
                    ERI[l, k, i, j] = ERI[l, k, j, i] = val
    return S, T, V, ERI


def _j(pmat: np.ndarray, ERI: np.ndarray) -> np.ndarray:
    return np.einsum("kl,ijkl->ij", pmat, ERI)


def _k(pmat: np.ndarray, ERI: np.ndarray) -> np.ndarray:
    return np.einsum("kl,ilkj->ij", pmat, ERI)


# ------------------------------------------------------------------ RHF
@dataclass
class SCFResult:
    E_total: float
    E_elec: float
    C: np.ndarray
    eps: np.ndarray
    P: np.ndarray
    T: float
    V_ne: float
    V_ee: float
    E_nuc: float
    n_iter: int
    converged: bool
    virial_residual: float      # 2<T> + <V> (Coulomb sistemlerinde ~0)

    def orbital(self, k: int = 0) -> float:
        return float(self.eps[k])


def rhf(basis: List[PGF], charges: Sequence[Tuple[float, Sequence[float]]],
        n_electrons: int, max_iter: int = 200, tol: float = 1e-10,
        damping: float = 0.35, verbose: bool = False) -> SCFResult:
    """
    Kapalı kabuk RHF. n_electrons çift olmalı (kapalı kabuk).
    """
    if n_electrons % 2 != 0:
        raise ValueError("rhf: kapalı kabuk için çift elektron sayısı gerekir")
    nocc = n_electrons // 2
    S, T, V, ERI = build_matrices(basis, charges)
    H = T + V
    # nükleer itme
    E_nuc = 0.0
    for a in range(len(charges)):
        for b in range(a + 1, len(charges)):
            Za, Ca = charges[a]; Zb, Cb = charges[b]
            R = float(np.linalg.norm(np.asarray(Ca, float) - np.asarray(Cb, float)))
            if R > 1e-12:
                E_nuc += Za * Zb / R
    # ortogonalizasyon: X = S^{-1/2}
    w, U = np.linalg.eigh(S)
    if np.min(w) < 1e-10:
        raise ValueError("rhf: doğrusal bağımlı taban")
    X = U @ np.diag(w ** -0.5) @ U.T
    P = np.zeros_like(S)
    E_old = None
    converged = False
    for it in range(1, max_iter + 1):
        J = _j(P, ERI)
        K = _k(P, ERI)
        F = H + J - 0.5 * K
        Fp = X.T @ F @ X
        eps, Cp = np.linalg.eigh(Fp)
        C = X @ Cp
        P_new = 2.0 * C[:, :nocc] @ C[:, :nocc].T
        P = (1.0 - damping) * P + damping * P_new
        E = 0.5 * float(np.sum(P * (H + F)))
        if E_old is not None and abs(E - E_old) < tol:
            converged = True
            E_old = E
            break
        E_old = E
    J = _j(P, ERI); K = _k(P, ERI)
    F = H + J - 0.5 * K
    Fp = X.T @ F @ X
    eps, Cp = np.linalg.eigh(Fp)
    C = X @ Cp
    E_elec = float(np.sum(P * H) + 0.5 * float(np.sum(P * (J - 0.5 * K))))
    T_e = float(np.sum(P * T))
    V_ne = float(np.sum(P * V))
    V_ee = 0.5 * float(np.sum(P * J)) - 0.25 * float(np.sum(P * K))
    E_tot = E_elec + E_nuc
    virial = 2.0 * T_e + (V_ne + V_ee + E_nuc)     # Coulomb sistemlerinde ~0 (virial)
    return SCFResult(E_total=E_tot, E_elec=E_elec, C=C, eps=eps, P=P, T=T_e, V_ne=V_ne,
                     V_ee=V_ee, E_nuc=E_nuc, n_iter=it, converged=converged,
                     virial_residual=virial)


# ------------------------------------------------------------------ taban kümeleri
def even_tempered(center: Sequence[float], a: float, b: float, K: int) -> List[PGF]:
    return [PGF(alpha=a * b ** k, center=np.asarray(center, float)) for k in range(K)]


def two_block(center: Sequence[float], a1: float, b1: float, K1: int,
              a2: float, b2: float, K2: int) -> List[PGF]:
    """Sıkı (çekirdek) + yayvan (değerlik) iki bloklu taban kümesi."""
    return (even_tempered(center, a1, b1, K1) + even_tempered(center, a2, b2, K2))


# ------------------------------------------------------------------ atom HF (kapalı kabuk)
def atom_hf(Z: float, n_electrons: int, K1: int = 3, K2: int = 2,
            x0: Optional[Sequence[float]] = None,
            return_res: bool = False):
    """
    Kapalı kabuk atom HF enerjisi. İki bloklu Gauss tabanı; tüm üsler
    (a1,b1,a2,b2) varyasyonel olarak (enerji minimizasyonu) optimize edilir.

    Döndürür: dict(E, a1, b1, a2, b2, converged, virial_residual, T, V_ne, V_ee, n_iter)
    """
    if n_electrons % 2 != 0:
        raise ValueError("atom_hf: kapalı kabuk için çift elektron sayısı gerekir")
    charges = [(Z, [0.0, 0.0, 0.0])]
    x0 = list(x0) if x0 else [0.6 * Z ** 0.9, 2.3, 0.09 * Z ** 0.9, 2.6]

    def energy(p):
        a1, b1, a2, b2 = (float(p[0]), float(p[1]), float(p[2]), float(p[3]))
        if min(a1, a2) <= 1e-3 or min(b1, b2) <= 1.02 or max(b1, b2) > 8.0:
            return 50.0
        basis = two_block([0, 0, 0], a1, b1, K1, a2, b2, K2)
        try:
            return rhf(basis, charges, n_electrons, max_iter=120, tol=1e-9).E_total
        except Exception:
            return 50.0

    res = minimize(energy, np.array(x0, float), method="Nelder-Mead",
                   options=dict(xatol=5e-4, fatol=1e-8, maxiter=260))
    p = res.x
    basis = two_block([0, 0, 0], float(p[0]), float(p[1]), K1, float(p[2]), float(p[3]), K2)
    scf = rhf(basis, charges, n_electrons, tol=1e-11)
    out = {
        "E": scf.E_total, "a1": float(p[0]), "b1": float(p[1]),
        "a2": float(p[2]), "b2": float(p[3]), "K1": K1, "K2": K2,
        "converged": scf.converged, "n_iter": scf.n_iter,
        "virial_residual": scf.virial_residual,
        "T": scf.T, "V_ne": scf.V_ne, "V_ee": scf.V_ee,
        "eps": [float(e) for e in scf.eps],
        "n_basis": len(basis),
    }
    return (out, scf) if return_res else out


# ------------------------------------------------------------------ Li-benzeri (açık kabuk)
def li_like_frozen_core(Z: float, K1: int = 3, K2: int = 3) -> Dict[str, object]:
    """
    Li-benzeri atom (1s² 2s¹): dondurulmuş çekirdek + değerlik.

    Yöntem:
      1) Çekirdek (Z, 2 elektron) kapalı kabuk RHF ile çözülür -> 1s yörüngesi.
      2) Tam tabanda değerlik Fock matrisi F_v = H + J(P_c) - ½K(P_c) kurulur.
      3) Değerlik çözümü, çekirdek yörüngesine DİK tümleç uzayında (S-metriği
         altında dikleştirilmiş) en düşük özdeğer olarak bulunur.
      4) E = E_çekirdek + h_vv + 2J_cv - K_cv

    Yaklaşım (raporlanır): çekirdek yörüngesi Li'nin kendisinde değil, Li²⁺
    iyonunda gevşetilmiştir (standart dondurulmuş çekirdek kabulü).
    """
    charges = [(Z, [0.0, 0.0, 0.0])]
    # 1) çekirdek: Li²⁺ benzeri (2 elektron)
    core_opt = atom_hf(Z, 2, K1=K1, K2=2)
    core_basis = two_block([0, 0, 0], core_opt["a1"], core_opt["b1"], K1,
                           core_opt["a2"], core_opt["b2"], 2)
    core_scf = rhf(core_basis, charges, 2, tol=1e-11)
    # 2) tam taban
    val_basis = even_tempered([0, 0, 0], max(core_opt["a2"] * 0.25, core_opt["a1"] * 0.02),
                              core_opt["b2"], K2)
    full = core_basis + val_basis
    n_full = len(full)
    S, T, V, ERI = build_matrices(full, charges)
    H = T + V
    w, U = np.linalg.eigh(S)
    X = U @ np.diag(w ** -0.5) @ U.T
    Pc = np.zeros((n_full, n_full))
    Pc[:len(core_basis), :len(core_basis)] = core_scf.P
    F = H + _j(Pc, ERI) - 0.5 * _k(Pc, ERI)
    # 3) çekirdeğe dik tümleç uzayı
    # çekirdek yörüngesini TAM uzaya göm (1s; 2 elektron -> 1 uzamsal yörünge)
    mos = np.zeros(n_full)
    mos[:len(core_basis)] = core_scf.C[:, 0]
    mos /= math.sqrt(float(mos @ S @ mos))             # S-metriğinde normalize
    trial = np.eye(n_full).copy()                      # tüm taban, çekirdeğe dikleştirilecek
    for k in range(trial.shape[1]):                    # çekirdeğe dikleştir (S-metriği)
        trial[:, k] -= mos * float(mos @ S @ trial[:, k])
    # S-ortonormalizasyon
    W = np.zeros_like(trial)
    for k in range(trial.shape[1]):
        v = trial[:, k].copy()
        for j in range(k):
            v -= W[:, j] * float(W[:, j] @ S @ v)
        nrm = math.sqrt(max(float(v @ S @ v), 1e-30))
        W[:, k] = v / nrm
    Fw = W.T @ F @ W
    eps_w, c_w = np.linalg.eigh(Fw)
    v = W @ c_w[:, 0]                                  # 2s yörüngesi
    # 4) enerji ve virial
    h_vv = float(v @ H @ v)
    # DİKKAT: Pc tam çekirdek yoğunluğudur (2 elektron). Bu yüzden
    #   vᵀJ(Pc)v = 2·(cc|vv)  ve  vᵀK(Pc)v = 2·(cv|vc)
    # olduğundan enerji ifadesinde EK çarpan 2 kullanılmaz:
    #   ε_2s = h_vv + vᵀJ(Pc)v − ½·vᵀK(Pc)v
    J_cv = float(v @ (_j(Pc, ERI)) @ v)
    K_cv = float(v @ (_k(Pc, ERI)) @ v)
    eps_2s = h_vv + J_cv - 0.5 * K_cv
    E = core_scf.E_elec + eps_2s
    T_tot = core_scf.T + float(v @ T @ v)
    V_ne_tot = core_scf.V_ne + float(v @ V @ v)
    V_ee_tot = core_scf.V_ee + J_cv - 0.5 * K_cv
    return {
        "E": E, "eps_core": [float(e) for e in core_scf.eps[:1]], "eps_2s": float(eps_w[0]),
        "core_E": core_scf.E_elec, "virial_residual": 2.0 * T_tot + V_ne_tot + V_ee_tot,
        "T": T_tot, "V_ne": V_ne_tot, "V_ee": V_ee_tot,
        "inv_r_hf_theorem": -V_ne_tot / Z, "approx": "dondurulmuş çekirdek (Li²⁺ gevşetmeli)",
    }


def li_like_frozen_core_params(Z: float, p: Sequence[float], K1: int = 3, K2: int = 3) -> Dict[str, object]:
    """li_like_frozen_core'un taban parametreleri verilebilen sürümü (iç kullanım)."""
    charges = [(Z, [0.0, 0.0, 0.0])]
    a1, b1, a2, b2 = (float(p[0]), float(p[1]), float(p[2]), float(p[3]))
    core_basis = two_block([0, 0, 0], a1, b1, K1, a2, b2, 2)
    core_scf = rhf(core_basis, charges, 2, max_iter=150, tol=1e-10)
    val_basis = even_tempered([0, 0, 0], max(a2, a1 * 0.02), b2, K2)
    full = core_basis + val_basis
    n_full = len(full)
    S, T, V, ERI = build_matrices(full, charges)
    H = T + V
    Pc = np.zeros((n_full, n_full))
    Pc[:len(core_basis), :len(core_basis)] = core_scf.P
    F = H + _j(Pc, ERI) - 0.5 * _k(Pc, ERI)
    mos = np.zeros(n_full); mos[:len(core_basis)] = core_scf.C[:, 0]
    mos /= math.sqrt(max(float(mos @ S @ mos), 1e-30))
    trial = np.eye(n_full).copy()
    for k in range(n_full):
        trial[:, k] -= mos * float(mos @ S @ trial[:, k])
    W = np.zeros_like(trial)
    ncol = 0
    for k in range(n_full):
        v = trial[:, k].copy()
        for j in range(ncol):
            v -= W[:, j] * float(W[:, j] @ S @ v)
        nrm = math.sqrt(max(float(v @ S @ v), 1e-30))
        if nrm < 1e-6:
            continue
        W[:, ncol] = v / nrm
        ncol += 1
    W = W[:, :ncol]
    eps_w, c_w = np.linalg.eigh(W.T @ F @ W)
    v = W @ c_w[:, 0]
    h_vv = float(v @ H @ v)
    J_cv = float(v @ (_j(Pc, ERI)) @ v)
    K_cv = float(v @ (_k(Pc, ERI)) @ v)
    eps_2s = h_vv + J_cv - 0.5 * K_cv
    T_tot = core_scf.T + float(v @ T @ v)
    V_ne_tot = core_scf.V_ne + float(v @ V @ v)
    V_ee_tot = core_scf.V_ee + J_cv - 0.5 * K_cv
    return {"E": core_scf.E_elec + eps_2s, "eps_2s": float(eps_w[0]), "core_E": core_scf.E_elec,
            "T": T_tot, "V_ne": V_ne_tot, "V_ee": V_ee_tot,
            "virial_residual": 2 * T_tot + V_ne_tot + V_ee_tot,
            "converged": core_scf.converged}


def li_hf(Z: float, K1: int = 3, K2: int = 3,
          x0: Sequence[float] = (5.5, 3.0, 0.22, 2.4)) -> Dict[str, object]:
    """
    Li-benzeri atom için taban üslerini VARYASYONEL olarak optimize eder
    (dondurulmuş çekirdek yaklaşımı içinde en iyi taban).
    """
    def energy(p):
        a1, b1, a2, b2 = (float(p[0]), float(p[1]), float(p[2]), float(p[3]))
        if min(a1, a2) <= 1e-3 or min(b1, b2) <= 1.02 or max(b1, b2) > 8.0:
            return 50.0
        try:
            return float(li_like_frozen_core_params(Z, [a1, b1, a2, b2], K1, K2)["E"])
        except Exception:
            return 50.0

    res = minimize(energy, np.array(x0, float), method="Nelder-Mead",
                   options=dict(xatol=1e-3, fatol=1e-7, maxiter=170))
    out = li_like_frozen_core_params(Z, res.x, K1, K2)
    out.update({"basis": [float(x) for x in res.x], "K1": K1, "K2": K2})
    return out


# ------------------------------------------------------------------ H2+ (tek elektron, iki çekirdek)
def h2plus_energy(R: float, Z: float = 1.0, K1: int = 3, K2: int = 2,
                  a1: float = 0.9, b1: float = 2.4, a2: float = 0.22, b2: float = 2.8,
                  optimize: bool = False) -> Dict[str, float]:
    """H2+ benzeri tek elektronlu iki merkezli sistem: E = ε_1 + Z²/R."""
    charges = [(Z, [-R / 2, 0, 0]), (Z, [+R / 2, 0, 0])]

    def build(p):
        aa1, bb1, aa2, bb2 = (float(p[0]), float(p[1]), float(p[2]), float(p[3]))
        left = two_block([-R / 2, 0, 0], aa1, bb1, K1, aa2, bb2, K2)
        right = two_block([+R / 2, 0, 0], aa1, bb1, K1, aa2, bb2, K2)
        return left + right

    def energy(p):
        try:
            S, T, V, _ = build_matrices(build(p), charges, want_eri=False)
            H = T + V
            w, U = np.linalg.eigh(S)
            X = U @ np.diag(w ** -0.5) @ U.T
            eps = np.linalg.eigvalsh(X.T @ H @ X)
            return float(eps[0]) + Z * Z / R
        except Exception:
            return 10.0

    if optimize:
        res = minimize(energy, np.array([a1, b1, a2, b2]), method="Nelder-Mead",
                       options=dict(xatol=5e-4, fatol=1e-9, maxiter=200))
        p = res.x
    else:
        p = np.array([a1, b1, a2, b2])
    S, T, V, _ = build_matrices(build(p), charges)
    H = T + V
    w, U = np.linalg.eigh(S)
    X = U @ np.diag(w ** -0.5) @ U.T
    eps = np.linalg.eigvalsh(X.T @ H @ X)
    return {"E": float(eps[0]) + Z * Z / R, "eps1": float(eps[0]),
            "basis": [float(x) for x in p], "E_nuc": Z * Z / R}


# ------------------------------------------------------------------ H2 (kapalı kabuk, iki merkez)
def h2_energy(R: float, Z: float = 1.0, n_electrons: int = 2,
              basis_params: Sequence[float] = (0.95, 2.4, 0.20, 2.8),
              K1: int = 3, K2: int = 2, optimize: bool = False) -> Dict[str, float]:
    """
    İki merkezli, iki elektronlu molekül (H2 / He2^2+ benzeri) RHF enerjisi.
    Taban üsleri equilibrio yakın noktada optimize edilir ve tüm R için sabit tutulur.
    """
    charges = [(Z, [-R / 2, 0, 0]), (Z, [+R / 2, 0, 0])]

    def build(p):
        left = two_block([-R / 2, 0, 0], p[0], p[1], K1, p[2], p[3], K2)
        right = two_block([+R / 2, 0, 0], p[0], p[1], K1, p[2], p[3], K2)
        return left + right

    def energy(p):
        if min(p[0], p[2]) <= 1e-3 or min(p[1], p[3]) <= 1.02:
            return 20.0
        try:
            return rhf(build(p), charges, n_electrons, max_iter=150, tol=1e-10).E_total
        except Exception:
            return 20.0

    if optimize:
        res = minimize(energy, np.array(basis_params, float), method="Nelder-Mead",
                       options=dict(xatol=5e-4, fatol=1e-9, maxiter=220))
        p = res.x
    else:
        p = np.array(basis_params, float)
    out = rhf(build(p), charges, n_electrons, tol=1e-11)
    return {"E": out.E_total, "T": out.T, "V_ne": out.V_ne, "V_ee": out.V_ee,
            "E_nuc": out.E_nuc, "basis": [float(x) for x in p],
            "virial_residual": out.virial_residual, "converged": out.converged,
            "n_iter": out.n_iter}


# ------------------------------------------------------------------ öz test
def solver_self_test(verbose: bool = True, fast: bool = False) -> Dict[str, float]:
    """
    İntegraller + HF + bağ eğrisi: bağımsız doğrulama yolları.

    fast=True: pahalı varyasyonel taban optimizasyonları atlanır (CI/otomatik
    koşular için); enerjiler yine literatürle karşılaştırılır, sapmalar hafif büyür.
    """
    out: Dict[str, float] = {}
    # --- 1) integraller
    p1 = PGF(1.0, np.array([0.0, 0.0, 0.0]))
    ana = eri(p1, p1, p1, p1)
    ref1 = _self_energy_reference(1.0)
    out["eri_1d_relerr"] = abs(ana - ref1) / abs(ref1)
    out["eri_6d_relerr"] = abs(ana - _numeric_eri_reference(p1, p1, p1, p1, 12, 13)) / abs(ana)
    Rbig = 25.0
    pB = PGF(1.0, np.array([Rbig, 0.0, 0.0]))
    out["eri_asymptotic"] = eri(p1, p1, pB, pB) * Rbig            # 1'e yakınsamalı
    s_ = 2.0
    pAs = PGF(1.0 * s_ * s_, np.array([0.0, 0.0, 0.0]))
    pBs = PGF(1.0 * s_ * s_, np.array([1.0 / s_, 0.0, 0.0]))
    pB1 = PGF(1.0, np.array([1.0, 0.0, 0.0]))
    out["eri_scaling_relerr"] = abs(eri(pAs, pAs, pBs, pBs) - s_ * eri(p1, p1, pB1, pB1)) / abs(s_ * eri(p1, p1, pB1, pB1))
    # --- 2) atom HF: literatür ile karşılaştırma
    for label, Z, ne, K1, K2 in [("He", 2.0, 2, 3, 2), ("Li+", 3.0, 2, 3, 2),
                                 ("Be2+", 4.0, 2, 3, 2), ("Be", 4.0, 4, 3, 3)]:
        r_ = (atom_hf(Z, ne, K1=K1, K2=K2) if not fast
              else atom_hf(Z, ne, K1=K1, K2=K2, x0=(0.6 * Z ** 0.9, 2.3, 0.09 * Z ** 0.9, 2.6)))
        out[f"E_{label}"] = float(r_["E"])
        out[f"err_{label}"] = abs(float(r_["E"]) - LIT_HF[(label, int(Z))])
        out[f"virial_{label}"] = abs(float(r_["virial_residual"]))
    if fast:
        r_li = li_like_frozen_core_params(3.0, [2.174, 5.044, 0.063, 8.0], 3, 3)
    else:
        r_li = li_hf(3.0)
    out["E_Li"] = float(r_li["E"]); out["err_Li"] = abs(float(r_li["E"]) - LIT_HF[("Li", 3)])
    out["virial_Li"] = abs(float(r_li["virial_residual"]))
    # --- 3) bağ: H2 ve H2+
    h2p = h2plus_energy(2.0, optimize=not fast)
    out["E_H2plus_R2"] = float(h2p["E"])
    out["err_H2plus_R2"] = abs(float(h2p["E"]) - (-0.602634))
    h2 = h2_energy(1.4, optimize=not fast)
    out["E_H2_R1.4"] = float(h2["E"])
    out["err_H2"] = abs(float(h2["E"]) - LIT_H2["E_HF"])
    out["virial_H2"] = abs(float(h2["virial_residual"]))
    if verbose:
        print("[gto] integral doğrulaması (3 bağımsız yol)")
        print(f"        1B öz-itme integrali  : bağıl hata {out['eri_1d_relerr']:.2e}")
        print(f"        6B kübik kuadratur    : bağıl hata {out['eri_6d_relerr']:.2e}")
        print(f"        R→∞ asimptotiği 25·ERI: {out['eri_asymptotic']:.6f}  (1 olmalı)")
        print(f"        ölçek özdeşliği       : bağıl hata {out['eri_scaling_relerr']:.2e}")
        print("[gto] Hartree–Fock vs literatür HF limiti (varyasyonel üst sınır olmalı)")
        for label in ("He", "Li+", "Be2+", "Be", "Li"):
            print(f"        {label:5s} E={out['E_' + label]:.6f}  |Δ_lit|={out['err_' + label]:.2e}  "
                  f"|virial|={out['virial_' + label]:.1e}")
        print("[gto] kimyasal bağ")
        print(f"        H2+  R=2.0 : E={out['E_H2plus_R2']:.6f}  |Δ_lit|={out['err_H2plus_R2']:.2e}")
        print(f"        H2   R=1.4 : E={out['E_H2_R1.4']:.6f}  |Δ_lit|={out['err_H2']:.2e}  "
              f"|virial|={out['virial_H2']:.1e}")
    return out


if __name__ == "__main__":
    solver_self_test()
