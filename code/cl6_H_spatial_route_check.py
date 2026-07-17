#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cl6_H_spatial_route_check.py
============================================================================
CLOSING CHECK for the H (quaternion) spatial-direction route.
CentralizerUnification_Consolidation.md, §5 (open thread).

QUESTION UNDER INVESTIGATION:

    The separate ℍ factor of the division algebra chain ℝ⊗ℂ⊗ℍ⊗𝕆 hosts
    SU(2)_L. Could the ℍ structure provide spatial rotation generators
    that escape the Cl(6) centralizer obstruction?

    The conjecture (§5 of the consolidation): spatial directions from ℍ
    would double-book with SU(2)_L or SU(2)_R — a different factor, same
    disease. This script verifies it explicitly.

STRATEGY:

    ℍ carries two natural su(2) algebras, acting on the same space:

      su(2)_L = {L_i, L_j, L_k} / 2   (left multiplication by imaginary
                                          quaternion units i, j, k)
      su(2)_R = {R_i, R_j, R_k} / 2   (right multiplication)

    These are the only non-trivial Lie algebra content of ℍ beyond the
    scalar. Left and right multiplications commute: [L_a, R_b] = 0.

    The spatial-route question: is there a generator in the ℍ algebra
    that commutes with SU(2)_L (so it doesn't double-book with weak
    isospin) and is not itself another internal gauge symmetry?

    CHECKS:
    (A) Build L_a, R_a as 4×4 real matrices (left/right multiplication
        on ℝ⁴ = span{1,i,j,k}).
    (B) Verify su(2)_L closure, su(2)_R closure, cross-commutativity.
    (C) Compute centralizer of su(2)_L in span{L_a, R_b, I}: expect
        su(2)_R ⊕ ℝ (4-dimensional). The ONLY escape from SU(2)_L
        double-booking is to land in su(2)_R.
    (D) Verify su(2)_R is an internal gauge symmetry (closes as su(2),
        acts on the same representation space). Not a free slot.
    (E) Compute centralizer of su(2)_L ⊕ su(2)_R in the full algebra:
        expect span{I} only (trivial — just the scalar). No generator
        commutes with BOTH internal SU(2)'s.
    (F) Conclusion: every non-scalar generator in ℍ double-books with
        either SU(2)_L or SU(2)_R. The ℍ spatial route is closed.

CONVENTIONS:
    Quaternion units 1, i, j, k with i²=j²=k²=-1, ij=k, jk=i, ki=j.
    Basis for ℝ⁴: e_0=1, e_1=i, e_2=j, e_3=k.
    L_q: 4×4 real matrix of LEFT multiplication by q.
    R_q: 4×4 real matrix of RIGHT multiplication by q.
    SU(2)_L generators: T_a = L_a / 2; [T_a, T_b] = ε_{abc} T_c.
    SU(2)_R generators: S_a = -R_a / 2; [S_a, S_b] = ε_{abc} S_c.

ENVIRONMENT:  Python 3.11, NumPy. Pure linear algebra; no external data.
OUTPUT: Printed to stdout AND teed to
    cl6_H_spatial_route_check_report.txt in the script's run directory.
============================================================================
"""

import sys
import os
import numpy as np
from itertools import product as iproduct


# ============================================================================
# Tee: mirror stdout to a local .txt report
# ============================================================================
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()


def _open_report():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "cl6_H_spatial_route_check_report.txt")
    fh = open(path, "w", encoding="utf-8")
    return fh, path


# ============================================================================
# Linear algebra helpers
# ============================================================================
def fnorm(M):
    return float(np.linalg.norm(M, "fro"))


def commutator(A, B):
    return A @ B - B @ A


def real_nullspace(A, tol=1e-10):
    """Real nullspace of a real matrix A via SVD."""
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    n = A.shape[1]
    rank = int(np.sum(s > tol * max(1.0, s[0] if s.size else 1.0)))
    return Vt[rank:].T, rank


# ============================================================================
# 1. Quaternion product and matrix construction
# ============================================================================

# Quaternion multiplication table:
# quat_prod[a][b] = (sign, index) meaning e_a * e_b = sign * e_{index}
# Basis: 0=1, 1=i, 2=j, 3=k
# Rules: ij=k, jk=i, ki=j; ji=-k, kj=-i, ik=-j; i²=j²=k²=-1; 1*x=x*1=x

QUAT_MUL = {}
# (a, b) -> (sign, result_index)
for a in range(4):
    for b in range(4):
        if a == 0:
            QUAT_MUL[(a, b)] = (1, b)
        elif b == 0:
            QUAT_MUL[(a, b)] = (1, a)
        elif a == b:
            QUAT_MUL[(a, b)] = (-1, 0)   # i²=j²=k²=-1
        else:
            # Levi-Civita for {i,j,k} = {1,2,3}
            cycle = {(1, 2): (1, 3), (2, 3): (1, 1), (3, 1): (1, 2),
                     (2, 1): (-1, 3), (3, 2): (-1, 1), (1, 3): (-1, 2)}
            QUAT_MUL[(a, b)] = cycle[(a, b)]


def quat_mul_vec(a_idx, b_idx):
    """Returns (sign, result_basis_index) for e_a * e_b."""
    return QUAT_MUL[(a_idx, b_idx)]


def build_L(q_idx):
    """4×4 real matrix of LEFT multiplication by e_{q_idx}."""
    M = np.zeros((4, 4), dtype=float)
    for c in range(4):
        sign, r = quat_mul_vec(q_idx, c)
        M[r, c] = sign
    return M


def build_R(q_idx):
    """4×4 real matrix of RIGHT multiplication by e_{q_idx}."""
    M = np.zeros((4, 4), dtype=float)
    for c in range(4):
        sign, r = quat_mul_vec(c, q_idx)
        M[r, c] = sign
    return M


# ============================================================================
# MAIN
# ============================================================================
def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=100)

    print("=" * 72)
    print("  ℍ SPATIAL ROUTE CHECK — closing the quaternion escape")
    print("  CentralizerUnification_Consolidation.md §5")
    print("=" * 72)

    # Imaginary quaternion unit indices: i=1, j=2, k=3
    imag = [1, 2, 3]
    names = {1: "i", 2: "j", 3: "k"}

    # Build L and R matrices for imaginary units
    L = {a: build_L(a) for a in imag}   # L_i, L_j, L_k
    R = {a: build_R(a) for a in imag}   # R_i, R_j, R_k
    I4 = np.eye(4, dtype=float)

    # ---------------------------------------------------------------
    # CHECK A: su(2)_L closure
    # [L_i, L_j] = 2 L_k  (T_a = L_a/2 -> [T_a,T_b] = eps_abc T_c)
    # ---------------------------------------------------------------
    print("\n  --- CHECK A: su(2)_L closure [L_a, L_b] = 2 eps_{abc} L_c ---")
    eps = {(1,2,3): 1, (2,3,1): 1, (3,1,2): 1,
           (2,1,3): -1, (1,3,2): -1, (3,2,1): -1}
    suL_err = 0.0
    for a in imag:
        for b in imag:
            if a >= b:
                continue
            c = [x for x in imag if x not in [a, b]][0]
            sign = eps.get((a, b, c), 0)
            res = fnorm(commutator(L[a], L[b]) - 2 * sign * L[c])
            suL_err = max(suL_err, res)
            print(f"    [L_{names[a]}, L_{names[b]}] - 2*{sign:+d}*L_{names[c]} : "
                  f"residual = {res:.2e}")
    print(f"  su(2)_L closes: max residual = {suL_err:.2e}",
          "PASS" if suL_err < 1e-12 else "FAIL")

    # ---------------------------------------------------------------
    # CHECK B: su(2)_R closure
    # [R_i, R_j] = -2 R_k  (right mult reverses order; S_a=-R_a/2 closes)
    # ---------------------------------------------------------------
    print("\n  --- CHECK B: su(2)_R closure [R_a, R_b] = -2 eps_{abc} R_c ---")
    suR_err = 0.0
    for a in imag:
        for b in imag:
            if a >= b:
                continue
            c = [x for x in imag if x not in [a, b]][0]
            sign = eps.get((a, b, c), 0)
            res = fnorm(commutator(R[a], R[b]) + 2 * sign * R[c])
            suR_err = max(suR_err, res)
            print(f"    [R_{names[a]}, R_{names[b]}] + 2*{sign:+d}*R_{names[c]} : "
                  f"residual = {res:.2e}")
    print(f"  su(2)_R closes: max residual = {suR_err:.2e}",
          "PASS" if suR_err < 1e-12 else "FAIL")
    print("  (S_a = -R_a/2 satisfies [S_a, S_b] = eps_abc S_c — genuine su(2))")

    # ---------------------------------------------------------------
    # CHECK C: Cross-commutativity [L_a, R_b] = 0
    # Left and right multiplications commute: (ax)b = a(xb).
    # ---------------------------------------------------------------
    print("\n  --- CHECK C: Cross-commutativity [L_a, R_b] = 0 (all 9 pairs) ---")
    cross_err = 0.0
    for a in imag:
        for b in imag:
            res = fnorm(commutator(L[a], R[b]))
            cross_err = max(cross_err, res)
    print(f"  max ||[L_a, R_b]|| over all 9 pairs = {cross_err:.2e}",
          "PASS (commute)" if cross_err < 1e-12 else "FAIL")
    print("  Interpretation: SU(2)_L and SU(2)_R are INDEPENDENT — one does")
    print("  not obstruct the other's action. The centralizer of su(2)_L")
    print("  therefore contains all of su(2)_R.")

    # ---------------------------------------------------------------
    # CHECK D: Centralizer of su(2)_L in span{L_a, R_b, I}
    # The algebra: 7-dimensional real space (3+3+1).
    # Find all M = sum c_a L_a + d_b R_b + e I with [M, L_a] = 0 for all a.
    # Expected: d_b unconstrained (span{R_i,R_j,R_k,I}), c_a = 0.
    # ---------------------------------------------------------------
    print("\n  --- CHECK D: Centralizer of su(2)_L in span{L_a, R_b, I} ---")

    # Basis for the 7-dim algebra: L_i, L_j, L_k, R_i, R_j, R_k, I
    basis_labels = ["L_i", "L_j", "L_k", "R_i", "R_j", "R_k", "I"]
    basis_mats = [L[1], L[2], L[3], R[1], R[2], R[3], I4]
    n_basis = len(basis_mats)

    # Build linear system: [M, L_a] = 0 for a in {i,j,k}
    # M = sum_k c_k B_k; condition: sum_k c_k [B_k, L_a] = 0 for each a
    # Stack into (3 * 16, 7) real matrix (16 = 4×4 entries)
    rows = []
    for a in imag:
        for B in basis_mats:
            rows.append(commutator(B, L[a]).reshape(-1))
    G = np.array(rows).T   # shape (7, 48), need to transpose: system is G c = 0
    # Each column of G is [B_k, L_a] vectorized, stacked over a
    # Actually: G[k, :] = commutator(B_k, L_a).reshape(-1) for all a
    # We want G c = 0 where G is (48, 7) and c is (7,)
    G_mat = np.zeros((3 * 16, n_basis))
    for k, B in enumerate(basis_mats):
        col = []
        for a in imag:
            col.append(commutator(B, L[a]).reshape(-1))
        G_mat[:, k] = np.concatenate(col)

    cent_basis, cent_rank = real_nullspace(G_mat, tol=1e-10)
    cent_dim = cent_basis.shape[1]

    print(f"  Centralizer dimension: {cent_dim}  (expected 4 = su(2)_R + scalar)")

    # Express centralizer basis in terms of the 7-dim algebra
    print(f"  Centralizer basis vectors (coefficients in [L_i,L_j,L_k,R_i,R_j,R_k,I]):")
    for col in range(cent_dim):
        v = cent_basis[:, col]
        # Normalize for readability
        nrm = np.max(np.abs(v))
        if nrm > 1e-10:
            v = v / nrm
        parts = []
        for k, lab in enumerate(basis_labels):
            if abs(v[k]) > 1e-8:
                parts.append(f"{v[k]:+.4f}·{lab}")
        print(f"    v_{col+1}: {' '.join(parts) if parts else '(zero)'}")

    # Verify centralizer lies in span{R_a, I}
    # Project each centralizer vector onto span{R_i,R_j,R_k,I} (indices 3,4,5,6)
    R_I_indices = [3, 4, 5, 6]
    leak_err = 0.0
    for col in range(cent_dim):
        v = cent_basis[:, col]
        leak = np.linalg.norm([v[k] for k in [0, 1, 2]])   # L_a components
        leak_err = max(leak_err, leak)
    print(f"  Max L_a component in centralizer vectors: {leak_err:.2e}",
          "PASS (no L_a content)" if leak_err < 1e-10 else "FAIL")
    print(f"  Centralizer of su(2)_L = span{{R_i, R_j, R_k, I}} = su(2)_R ⊕ ℝ",
          "CONFIRMED" if cent_dim == 4 and leak_err < 1e-10 else "CHECK MANUALLY")

    # ---------------------------------------------------------------
    # CHECK E: su(2)_R is itself an internal gauge symmetry
    # It closes as su(2) (CHECK B confirmed), so it acts as another
    # internal SU(2) — not a free slot for spatial rotation.
    # Show: R_a generators are non-commuting (nonabelian) and close.
    # ---------------------------------------------------------------
    print("\n  --- CHECK E: su(2)_R is a genuine nonabelian internal su(2) ---")
    suR_nonab = max(fnorm(commutator(R[a], R[b]))
                    for a in imag for b in imag if a != b)
    print(f"  max ||[R_a, R_b]|| (a≠b) = {suR_nonab:.4f}",
          "(nonabelian)" if suR_nonab > 1e-6 else "(abelian — unexpected)")
    print(f"  su(2)_R closes (CHECK B: max residual {suR_err:.2e})")
    print(f"  Verdict: su(2)_R is a genuine su(2) Lie algebra —")
    print(f"    assigning it as spatial so(3) double-books spatial")
    print(f"    rotation with internal SU(2)_R.")

    # ---------------------------------------------------------------
    # CHECK F: Centralizer of su(2)_L ⊕ su(2)_R in the full algebra
    # Expected: span{I} only (the scalar — not a rotation generator).
    # ---------------------------------------------------------------
    print("\n  --- CHECK F: Centralizer of su(2)_L ⊕ su(2)_R in span{L_a,R_b,I} ---")

    # Condition: [M, L_a] = 0 AND [M, R_b] = 0 for all a, b
    G_full = np.zeros((6 * 16, n_basis))
    row_block = 0
    for a in imag:
        for B in basis_mats:
            pass  # already computed above
        for k, B in enumerate(basis_mats):
            G_full[row_block * 16:(row_block + 1) * 16, k] = \
                commutator(B, L[a]).reshape(-1)
        row_block += 1
    for b in imag:
        for k, B in enumerate(basis_mats):
            G_full[row_block * 16:(row_block + 1) * 16, k] = \
                commutator(B, R[b]).reshape(-1)
        row_block += 1

    full_cent_basis, _ = real_nullspace(G_full, tol=1e-10)
    full_cent_dim = full_cent_basis.shape[1]

    print(f"  Centralizer dimension: {full_cent_dim}  (expected 1 = scalar only)")
    if full_cent_dim >= 1:
        print(f"  Centralizer basis vectors:")
        for col in range(full_cent_dim):
            v = full_cent_basis[:, col]
            nrm = np.max(np.abs(v))
            if nrm > 1e-10:
                v = v / nrm
            parts = []
            for k, lab in enumerate(basis_labels):
                if abs(v[k]) > 1e-8:
                    parts.append(f"{v[k]:+.4f}·{lab}")
            print(f"    v_{col+1}: {' '.join(parts) if parts else '(zero)'}")

    scalar_only = (full_cent_dim == 1)
    # Check it's actually the identity
    if scalar_only:
        v = full_cent_basis[:, 0]
        I_component = abs(v[6]) / (np.linalg.norm(v) + 1e-30)
        print(f"  Identity component fraction: {I_component:.6f}",
              "(is the scalar)" if I_component > 0.99 else "(unexpected)")

    print(f"  Centralizer of su(2)_L ⊕ su(2)_R = span{{I}} only:",
          "CONFIRMED" if scalar_only else "FAILS — check manually")
    print(f"  The identity (scalar) is NOT a rotation generator.")
    print(f"  No non-trivial generator in the ℍ algebra commutes with")
    print(f"  BOTH SU(2)_L and SU(2)_R simultaneously.")

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  SUMMARY AND VERDICT")
    print("=" * 72)

    all_pass = (suL_err < 1e-12 and suR_err < 1e-12 and
                cross_err < 1e-12 and cent_dim == 4 and
                leak_err < 1e-10 and scalar_only)

    print(f"""
  The ℍ factor of ℝ⊗ℂ⊗ℍ⊗𝕆 carries exactly two su(2) algebras:

    su(2)_L = span{{L_i/2, L_j/2, L_k/2}}  (left multiplication)
    su(2)_R = span{{-R_i/2, -R_j/2, -R_k/2}} (right multiplication)

  These are the ONLY nonabelian Lie algebraic content of ℍ beyond
  the real scalar I.

  Key results (all verified above):

  1. su(2)_L closes as su(2) ................................ {'PASS' if suL_err < 1e-12 else 'FAIL'}
  2. su(2)_R closes as su(2) ................................ {'PASS' if suR_err < 1e-12 else 'FAIL'}
  3. [L_a, R_b] = 0 (cross-commutativity) .................. {'PASS' if cross_err < 1e-12 else 'FAIL'}
  4. Centralizer of su(2)_L = su(2)_R ⊕ ℝ (dim=4) ......... {'PASS' if cent_dim == 4 else 'FAIL'}
  5. su(2)_R is nonabelian internal su(2) ................... {'PASS' if suR_nonab > 1e-6 else 'FAIL'}
  6. Centralizer of su(2)_L ⊕ su(2)_R = span{{I}} (dim=1) .. {'PASS' if scalar_only else 'FAIL'}

  SPATIAL DIRECTION IMPLICATION:

  For a generator X in the ℍ algebra to serve as a spatial rotation
  WITHOUT double-booking SU(2)_L, X must be in the centralizer of
  su(2)_L = su(2)_R ⊕ ℝ  [Check D].

  If X is assigned as spatial rotation so(3), it must ALSO not
  double-book SU(2)_R. So X must be in the centralizer of
  su(2)_L ⊕ su(2)_R = span{{I}} only  [Check F].

  The scalar (identity) is not a rotation generator.

  CONCLUSION: The ℍ spatial route is CLOSED.

  Every non-scalar generator in ℍ double-books with either SU(2)_L
  or SU(2)_R. This is the same partition-vs-double-booking dilemma
  as the Cl(6) case, now on the ℍ factor:
    - To avoid SU(2)_L double-booking: must use su(2)_R
    - su(2)_R is itself an internal gauge symmetry
    - No generator escapes both internal SU(2)'s simultaneously

  Even if this algebraic obstruction were somehow circumvented,
  the Coleman–Mandula theorem would apply independently: mixing
  Poincaré spatial rotations with internal su(2)_R is forbidden for
  a gapped interacting 4D theory, regardless of which tensor factor
  the mixing occurs on.

  OVERALL VERDICT: {'ALL CHECKS PASS — ℍ route closed.' if all_pass else 'ONE OR MORE CHECKS FAILED.'}
""")

    print("  The three-prong (3+1) no-go is now complete:")
    print("    (i)  Partition kills su(3) color (Cl(6), dimension theorem)")
    print("    (ii) Double-booking triggers C-M (Cl(6), centralizer)")
    print("    (iii)Chain gives only 1+1 (multi-HSMI/ladder closure)")
    print("    (iv) ℍ factor gives su(2)_L vs su(2)_R double-booking")
    print("         (this check)")
    print()
    print("  Under Stance 1, no clean route to (3+1) remains.")
    print("  Stance 2 (imported spacetime + SU(2)_L on ℍ) is the")
    print("  algebraically grounded fallback.")
    print("=" * 72)


# ============================================================================
if __name__ == "__main__":
    report_fh, report_path = _open_report()
    old_stdout = sys.stdout
    sys.stdout = _Tee(old_stdout, report_fh)
    try:
        main()
    finally:
        sys.stdout = old_stdout
        report_fh.close()
        print(f"\n[report written to: {report_path}]")
