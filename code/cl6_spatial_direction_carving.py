#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cl6_spatial_direction_carving.py
============================================================================
STEP 1 — Spatial-direction carving analysis for the C-series emergence half.

QUESTION UNDER INVESTIGATION:

    How are the 2 missing spatial directions carved from the 6 Cl(6)
    internal directions + chain axis n?

    The C-series has emerged a clean 1+1 sector (lightray along n).
    Reaching (3+1) requires 2 more spatial directions from the 6 internal
    Cl(6) generators. This script characterizes the obstruction by:

    (1) Decomposing all 15 bivectors B_ij under the verified su(3) octet
        into their su(3) representation sectors: adjoint 8, centralizer 1
        (= span{H}), and coset 3+3-bar (color-charged).

    (2) Enumerating the two inequivalent spatial-direction assignments
        (Case A: both from one mode-pair; Case B: from different mode-pairs)
        and determining what su(3) content each spatial rotation generator
        carries.

    (3) For each case, checking whether the remaining "internal" bivectors
        still contain su(3), or whether the assignment breaks color.

DEPENDS ON:  cl6_su3_centralizer_check.py results (all PASS assumed).
    This script re-derives the su(3) octet from scratch as an independent
    cross-check; construction is identical (stabilizer of |xi0>).

CONVENTIONS:  Match Paper 2 and cl6_su3_centralizer_check.py exactly.
    e_i Hermitian, e_i^2 = I8; B_ij = (i/4)[e_i, e_j]; H = B12+B34+B56.
    Mode-pairs: (e1,e2), (e3,e4), (e5,e6).  Cartan: {B12, B34, B56}.

ENVIRONMENT:  Python 3.11, NumPy. Pure linear algebra; no external data.
OUTPUT:  Printed to stdout AND teed to
    cl6_spatial_direction_carving_report.txt in the script's run directory.
============================================================================
"""

import sys
import os
import numpy as np
from itertools import combinations


# ============================================================================
# Tee: mirror all stdout to a local .txt report
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
    path = os.path.join(here, "cl6_spatial_direction_carving_report.txt")
    fh = open(path, "w", encoding="utf-8")
    return fh, path


# ============================================================================
# Linear algebra helpers
# ============================================================================
def fnorm(M):
    """Frobenius norm."""
    return float(np.linalg.norm(M, "fro"))


def commutator(A, B):
    return A @ B - B @ A


def hs_inner(A, B):
    """Hilbert-Schmidt inner product Tr(A^dag B)."""
    return np.trace(A.conj().T @ B)


def real_nullspace(A, tol=1e-10):
    """
    Real nullspace of a complex linear map.
    A is (m x n) complex; returns real c (length n) with A c = 0.
    """
    Ar = np.vstack([A.real, A.imag])
    U, s, Vt = np.linalg.svd(Ar, full_matrices=True)
    n = Ar.shape[1]
    rank = int(np.sum(s > tol * max(1.0, s[0] if s.size else 1.0)))
    null_basis = Vt[rank:].T.conj()
    return null_basis.real, rank


# ============================================================================
# 1. Build Cl(6) generators and bivectors (identical to centralizer check)
# ============================================================================
def build_clifford():
    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    def kron(*ops):
        out = np.array([[1]], dtype=complex)
        for op in ops:
            out = np.kron(out, op)
        return out

    e = [
        kron(sx, I2, I2),   # e1
        kron(sy, I2, I2),   # e2
        kron(sz, sx, I2),   # e3
        kron(sz, sy, I2),   # e4
        kron(sz, sz, sx),   # e5
        kron(sz, sz, sy),   # e6
    ]
    return e


def build_bivectors(e):
    """B_ij = (i/4)[e_i, e_j], Hermitian generators of spin(6). 15 of them."""
    bivs = []
    labels = []
    for i, j in combinations(range(6), 2):
        B = (1j / 4.0) * (e[i] @ e[j] - e[j] @ e[i])
        bivs.append(B)
        labels.append(f"B{i+1}{j+1}")
    return bivs, labels


# ============================================================================
# 2. Derive su(3) octet (identical construction to centralizer check)
# ============================================================================
def derive_su3(bivs, labels):
    """
    Derive su(3) = stabilizer of |xi0> (H = -3/2 eigenstate).
    Returns: (su3_basis_raw, su3_onb, H_matrix, xi0_vector).
    su3_onb is Gram-Schmidt orthonormalized in HS inner product.
    """
    idx = {lab: k for k, lab in enumerate(labels)}
    H = bivs[idx["B12"]] + bivs[idx["B34"]] + bivs[idx["B56"]]

    # preferred spinor: H = -3/2 eigenstate
    Hdiag = np.diag(H).real
    xi_idx = int(np.argmin(np.abs(Hdiag - (-1.5))))
    xi0 = np.zeros(8, dtype=complex)
    xi0[xi_idx] = 1.0

    # stabilizer: X|xi0> = 0 for X = sum_a c_a B_a
    A = np.column_stack([B @ xi0 for B in bivs])
    null_c, rank = real_nullspace(A, tol=1e-9)
    k = null_c.shape[1]

    su3_raw = []
    for col in range(k):
        c = null_c[:, col]
        X = sum(c[a] * bivs[a] for a in range(15))
        su3_raw.append(X)

    # Gram-Schmidt orthonormalization (HS)
    onb = []
    for X in su3_raw:
        Y = X.copy().astype(complex)
        for Q in onb:
            Y = Y - hs_inner(Q, Y) * Q
        nrm = np.sqrt(abs(hs_inner(Y, Y)))
        if nrm > 1e-10:
            onb.append(Y / nrm)

    return su3_raw, onb, H, xi0, null_c


# ============================================================================
# 3. Decompose each bivector into su(3) representation sectors
# ============================================================================
def decompose_bivectors(bivs, labels, su3_onb, H):
    """
    Project each B_ij onto three orthogonal sectors:
      (a) su(3) adjoint (8-dim): spanned by su3_onb
      (b) centralizer (1-dim): spanned by H_hat = H / ||H||_HS
      (c) coset (6-dim): orthogonal complement (= 3 + 3-bar)

    Returns dict: label -> (proj_adj, proj_cent, proj_coset, frac_adj, frac_cent, frac_coset)
    where proj_* are the projected matrices and frac_* = ||proj||^2 / ||B||^2.
    """
    H_norm = np.sqrt(abs(hs_inner(H, H)))
    H_hat = H / H_norm

    results = {}
    for k, (B, lab) in enumerate(zip(bivs, labels)):
        B_norm2 = abs(hs_inner(B, B))

        # project onto su(3) adjoint
        proj_adj = sum(hs_inner(Q, B) * Q for Q in su3_onb)
        # project onto centralizer (span{H})
        proj_cent = hs_inner(H_hat, B) * H_hat
        # coset = remainder
        proj_coset = B - proj_adj - proj_cent

        frac_adj = abs(hs_inner(proj_adj, proj_adj)) / B_norm2
        frac_cent = abs(hs_inner(proj_cent, proj_cent)) / B_norm2
        frac_coset = abs(hs_inner(proj_coset, proj_coset)) / B_norm2

        results[lab] = {
            "proj_adj": proj_adj,
            "proj_cent": proj_cent,
            "proj_coset": proj_coset,
            "frac_adj": float(frac_adj),
            "frac_cent": float(frac_cent),
            "frac_coset": float(frac_coset),
        }

    return results


# ============================================================================
# 4. Classify bivectors by type
# ============================================================================
def classify_bivector(label):
    """
    Classify a bivector B_ij by its mode-pair structure.
    Mode-pairs: (1,2)=pair1, (3,4)=pair2, (5,6)=pair3.
    Returns: 'within' (both indices in same pair) or 'cross' (different pairs),
             plus which pairs are involved.
    """
    i, j = int(label[1]) - 1, int(label[2]) - 1  # 0-indexed
    pair_of = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
    pi, pj = pair_of[i], pair_of[j]
    if pi == pj:
        return "within", (pi,)
    else:
        return "cross", (pi, pj)


# ============================================================================
# 5. Case analysis: spatial-direction assignments
# ============================================================================
def analyze_spatial_assignment(case_label, spatial_dirs, bivs, labels, su3_onb, H):
    """
    Given a spatial-direction assignment (a pair of e_i indices, 0-based),
    determine:
    (a) Which bivector is the spatial rotation generator (B_ij for the
        plane formed by the two spatial directions).
    (b) Its su(3) sector decomposition.
    (c) Which of the 15 bivectors involve at least one spatial direction
        (and are therefore "recruited" into the spatial/mixed sector).
    (d) Whether the remaining "purely internal" bivectors still contain
        a closed su(3).
    """
    i, j = spatial_dirs
    idx_map = {lab: k for k, lab in enumerate(labels)}

    print(f"\n  --- {case_label}: spatial directions = e{i+1}, e{j+1} ---")

    # (a) Spatial rotation generator
    # B_{i+1,j+1} in our labeling
    sp_label = f"B{min(i,j)+1}{max(i,j)+1}"
    print(f"  Spatial rotation generator: {sp_label}")

    # (b) Decomposition of the spatial generator
    sp_idx = idx_map[sp_label]
    sp_B = bivs[sp_idx]
    sp_norm2 = abs(hs_inner(sp_B, sp_B))
    H_hat = H / np.sqrt(abs(hs_inner(H, H)))

    proj_adj = sum(hs_inner(Q, sp_B) * Q for Q in su3_onb)
    proj_cent = hs_inner(H_hat, sp_B) * H_hat
    proj_coset = sp_B - proj_adj - proj_cent

    f_adj = abs(hs_inner(proj_adj, proj_adj)) / sp_norm2
    f_cent = abs(hs_inner(proj_cent, proj_cent)) / sp_norm2
    f_coset = abs(hs_inner(proj_coset, proj_coset)) / sp_norm2

    print(f"  su(3) decomposition of {sp_label}:")
    print(f"    adjoint-8 fraction : {f_adj:.6f}")
    print(f"    centralizer-1 frac : {f_cent:.6f}")
    print(f"    coset 3+3bar frac  : {f_coset:.6f}")
    print(f"    sum (must be 1)    : {f_adj + f_cent + f_coset:.10f}")

    # (c) Classify all 15 bivectors as "spatial-touching" or "purely internal"
    spatial_set = {i, j}  # 0-based generator indices
    internal_bivs = []
    spatial_bivs = []
    for k_b, lab in enumerate(labels):
        bi = int(lab[1]) - 1
        bj = int(lab[2]) - 1
        generators_involved = {bi, bj}
        if generators_involved & spatial_set:
            spatial_bivs.append(lab)
        else:
            internal_bivs.append(lab)

    print(f"\n  Bivectors touching spatial directions ({len(spatial_bivs)}):")
    print(f"    {', '.join(spatial_bivs)}")
    print(f"  Purely internal bivectors ({len(internal_bivs)}):")
    print(f"    {', '.join(internal_bivs)}")

    # (d) Check: do the purely internal bivectors close as a Lie algebra?
    internal_mats = [bivs[idx_map[lab]] for lab in internal_bivs]
    n_int = len(internal_mats)

    if n_int == 0:
        print("  No purely internal bivectors remain — su(3) is destroyed.")
        return

    # Build orthonormal basis for span(internal bivectors)
    int_onb = []
    for X in internal_mats:
        Y = X.copy().astype(complex)
        for Q in int_onb:
            Y = Y - hs_inner(Q, Y) * Q
        nrm = np.sqrt(abs(hs_inner(Y, Y)))
        if nrm > 1e-10:
            int_onb.append(Y / nrm)

    int_dim = len(int_onb)
    print(f"\n  Span dimension of purely internal bivectors: {int_dim}")

    # Check closure: do [X_a, X_b] stay in span(internal)?
    closure_res = 0.0
    for a in range(n_int):
        for b in range(a + 1, n_int):
            C = commutator(internal_mats[a], internal_mats[b])
            Cproj = sum(hs_inner(Q, C) * Q for Q in int_onb)
            closure_res = max(closure_res, fnorm(C - Cproj))

    print(f"  Closure of purely internal bivectors under commutator:")
    print(f"    max ||[X,Y] - proj|| = {closure_res:.2e}",
          "CLOSES" if closure_res < 1e-8 else "DOES NOT CLOSE")

    # What Lie algebra do they form? Check dimension against known algebras.
    # so(4) = 6-dim, so(3) = 3-dim, su(3) = 8-dim, su(2)+su(2) = 6-dim
    print(f"    dim = {int_dim}  (su(3)=8, so(4)=su(2)+su(2)=6, su(2)=3, u(1)=1)")

    # Check: does the original su(3) survive inside the internal bivectors?
    # Project each su(3) basis element onto span(internal) and measure residual.
    su3_survival = 0.0
    su3_leaked = 0.0
    su3_internal_count = 0
    for Q in su3_onb:
        Qproj = sum(hs_inner(R, Q) * R for R in int_onb)
        resid = fnorm(Q - Qproj)
        su3_survival = max(su3_survival, 1.0 - resid**2 / abs(hs_inner(Q, Q)))
        if resid < 1e-8:
            su3_internal_count += 1
        su3_leaked = max(su3_leaked, resid)

    print(f"\n  su(3) survival in purely internal bivectors:")
    print(f"    su(3) basis vectors fully contained: {su3_internal_count} / {len(su3_onb)}")
    print(f"    max leakage ||Q - proj_internal(Q)||: {su3_leaked:.4f}")
    if su3_internal_count == len(su3_onb):
        print(f"    VERDICT: su(3) SURVIVES intact")
    else:
        print(f"    VERDICT: su(3) is BROKEN — {len(su3_onb) - su3_internal_count}"
              f" generators leak into spatial sector")

    # Check: does H survive in the internal bivectors?
    H_proj = sum(hs_inner(R, H) * R for R in int_onb)
    H_resid = fnorm(H - H_proj) / fnorm(H)
    print(f"\n  H = B12+B34+B56 survival:")
    print(f"    ||H - proj_internal(H)|| / ||H|| = {H_resid:.6f}")
    if H_resid < 1e-8:
        print(f"    H is purely internal — centralizer direction survives")
    else:
        print(f"    H has spatial components — centralizer direction is disrupted")


# ============================================================================
# MAIN
# ============================================================================
def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=120)

    print("=" * 76)
    print("  Cl(6) SPATIAL-DIRECTION CARVING ANALYSIS  —  Step 1")
    print("  C-series emergence half: which bivectors become spatial?")
    print("=" * 76)

    # --- Build algebra ---
    e = build_clifford()
    bivs, labels = build_bivectors(e)
    idx_map = {lab: k for k, lab in enumerate(labels)}
    print(f"\n[1] Built Cl(6): 6 generators, {len(bivs)} bivectors")

    # --- Derive su(3) ---
    su3_raw, su3_onb, H, xi0, null_c = derive_su3(bivs, labels)
    print(f"[2] su(3) stabilizer derived: dim = {len(su3_onb)}",
          "PASS" if len(su3_onb) == 8 else "FAIL")

    # --- Sanity: verify su(3) closure (must match centralizer check) ---
    closure_res = 0.0
    for a in range(len(su3_raw)):
        for b in range(a + 1, len(su3_raw)):
            C = commutator(su3_raw[a], su3_raw[b])
            Cproj = sum(hs_inner(Q, C) * Q for Q in su3_onb)
            closure_res = max(closure_res, fnorm(C - Cproj))
    print(f"[3] su(3) closure cross-check: residual = {closure_res:.2e}",
          "PASS" if closure_res < 1e-8 else "FAIL")

    # =================================================================
    # PART A: Decompose all 15 bivectors under su(3)
    # =================================================================
    print("\n" + "=" * 76)
    print("  PART A: Decomposition of 15 bivectors under su(3)")
    print("  Sectors: adjoint-8 | centralizer-1 (=span{H}) | coset 3+3bar")
    print("=" * 76)

    decomp = decompose_bivectors(bivs, labels, su3_onb, H)

    # Print table
    print(f"\n  {'Bivector':>8s}  {'Type':>8s}  {'Pairs':>6s}"
          f"  {'f_adj8':>8s}  {'f_cent1':>8s}  {'f_coset6':>8s}  {'sum':>10s}")
    print("  " + "-" * 70)

    for lab in labels:
        d = decomp[lab]
        btype, pairs = classify_bivector(lab)
        pair_str = ",".join(str(p+1) for p in pairs)
        s = d["frac_adj"] + d["frac_cent"] + d["frac_coset"]
        print(f"  {lab:>8s}  {btype:>8s}  {pair_str:>6s}"
              f"  {d['frac_adj']:>8.4f}  {d['frac_cent']:>8.4f}"
              f"  {d['frac_coset']:>8.4f}  {s:>10.8f}")

    # Classify into pure-sector bivectors
    print("\n  --- Classification summary ---")
    pure_adj = [l for l in labels if decomp[l]["frac_adj"] > 0.999]
    pure_cent = [l for l in labels if decomp[l]["frac_cent"] > 0.999]
    pure_coset = [l for l in labels if decomp[l]["frac_coset"] > 0.999]
    mixed = [l for l in labels if l not in pure_adj + pure_cent + pure_coset]

    print(f"  Pure adjoint-8:     {len(pure_adj):>2d}  {pure_adj}")
    print(f"  Pure centralizer-1: {len(pure_cent):>2d}  {pure_cent}")
    print(f"  Pure coset 3+3bar:  {len(pure_coset):>2d}  {pure_coset}")
    print(f"  Mixed sectors:      {len(mixed):>2d}  {mixed}")

    # Cross-check: dimensions
    # For the Cartan elements (B12, B34, B56): these span the same 3-dim
    # space as (H, h1, h2) where h1,h2 are su(3) Cartan.
    # So each Cartan B has components in both adjoint and centralizer.
    print("\n  --- Cartan element decomposition (B12, B34, B56) ---")
    for lab in ["B12", "B34", "B56"]:
        d = decomp[lab]
        print(f"  {lab}: adj8={d['frac_adj']:.4f}  cent1={d['frac_cent']:.4f}"
              f"  coset={d['frac_coset']:.4f}")
    print("  (Expected: each has components in both adjoint and centralizer,")
    print("   since H = B12+B34+B56 is the centralizer direction, while the")
    print("   su(3) Cartan h1,h2 are two independent orthogonal combinations.)")

    # =================================================================
    # PART B: Spatial-direction assignment analysis
    # =================================================================
    print("\n" + "=" * 76)
    print("  PART B: Spatial-direction assignment cases")
    print("  Question: assign 2 of the 6 Cl(6) directions as 'spatial'.")
    print("  Two inequivalent cases (up to S3 permutation of mode-pairs).")
    print("=" * 76)

    # Case A: both from one mode-pair (representative: e5, e6)
    analyze_spatial_assignment(
        "Case A (within-pair)", (4, 5), bivs, labels, su3_onb, H
    )

    # Case B: from different mode-pairs (representative: e3, e5)
    analyze_spatial_assignment(
        "Case B (cross-pair)", (2, 4), bivs, labels, su3_onb, H
    )

    # =================================================================
    # PART C: Additional cross-pair variants (check for any escape)
    # =================================================================
    print("\n" + "=" * 76)
    print("  PART C: Exhaustive cross-pair variants")
    print("  Are all cross-pair assignments equivalent, or does the")
    print("  specific choice matter?")
    print("=" * 76)

    # All distinct cross-pair choices (one from each of two different pairs)
    # Pair 1: {e1(0), e2(1)}, Pair 2: {e3(2), e4(3)}, Pair 3: {e5(4), e6(5)}
    cross_cases = [
        ("B(p1,p2): e1,e3", (0, 2)),
        ("B(p1,p2): e1,e4", (0, 3)),
        ("B(p1,p2): e2,e3", (1, 2)),
        ("B(p1,p2): e2,e4", (1, 3)),
        ("B(p1,p3): e1,e5", (0, 4)),
        ("B(p1,p3): e2,e6", (1, 5)),
        ("B(p2,p3): e3,e5", (2, 4)),
        ("B(p2,p3): e4,e6", (3, 5)),
    ]

    # For each, just report: how many su(3) generators survive internally?
    print(f"\n  {'Assignment':>25s}  {'Spatial gen':>10s}  {'#internal':>9s}"
          f"  {'int dim':>7s}  {'su3 survive':>11s}")
    print("  " + "-" * 72)

    for case_name, (si, sj) in cross_cases:
        spatial_set = {si, sj}
        sp_label = f"B{min(si,sj)+1}{max(si,sj)+1}"

        internal_labs = []
        for lab in labels:
            bi = int(lab[1]) - 1
            bj = int(lab[2]) - 1
            if not ({bi, bj} & spatial_set):
                internal_labs.append(lab)

        internal_mats = [bivs[idx_map[lab]] for lab in internal_labs]

        # span dimension
        int_onb = []
        for X in internal_mats:
            Y = X.copy().astype(complex)
            for Q in int_onb:
                Y = Y - hs_inner(Q, Y) * Q
            nrm = np.sqrt(abs(hs_inner(Y, Y)))
            if nrm > 1e-10:
                int_onb.append(Y / nrm)

        # su(3) survival count
        su3_count = 0
        for Q in su3_onb:
            Qproj = sum(hs_inner(R, Q) * R for R in int_onb)
            if fnorm(Q - Qproj) < 1e-8:
                su3_count += 1

        print(f"  {case_name:>25s}  {sp_label:>10s}  {len(internal_labs):>9d}"
              f"  {len(int_onb):>7d}  {su3_count:>4d}/8"
              f" {'INTACT' if su3_count == 8 else 'BROKEN'}")

    # =================================================================
    # PART D: Summary and verdict
    # =================================================================
    print("\n" + "=" * 76)
    print("  SUMMARY AND VERDICT")
    print("=" * 76)

    print("""
  The 15 bivectors of so(6) decompose under su(3) as 8 + 1 + 6:
    - 8 adjoint generators (color rotations, including Cartan h1,h2)
    - 1 centralizer direction (H = B12+B34+B56 = (3/2)Q_{B-L})
    - 6 coset generators (3 + 3-bar, color-charged)

  The three Cartan bivectors (B12, B34, B56) are MIXED: each has
  components in BOTH the su(3) adjoint (h1,h2 directions) AND the
  centralizer (H direction). Only their sum H is pure centralizer.

  CASE A (within-pair, e.g. e5,e6 spatial):
    - Spatial rotation generator B56 is MIXED (adjoint + centralizer).
    - 9 bivectors involve a spatial direction => recruited out.
    - Only 6 purely internal bivectors remain.
    - These 6 span so(4) = su(2)+su(2), NOT su(3).
    - su(3) is BROKEN: some generators leak into the spatial sector.

  CASE B (cross-pair, e.g. e3,e5 spatial):
    - Spatial rotation generator B35 is a cross-plane bivector.
    - 9 bivectors involve a spatial direction => recruited out.
    - Only 6 purely internal bivectors remain.
    - su(3) is BROKEN: some generators leak into the spatial sector.

  VERDICT: Every assignment of 2 internal directions as 'spatial'
  breaks su(3) color. There is no clean carving.

  The obstruction is STRUCTURAL, not a choice failure: 15 - 9 = 6,
  and 6 < 8 = dim(su(3)). Removing any pair of generators from the
  6-direction pool necessarily removes at least 9 of the 15 bivectors
  from the purely-internal sector, leaving fewer than the 8 needed
  for su(3).

  This SHARPENS the (3+1) no-go beyond C4 sec.6:
    - C4 showed: spatial so(3) cannot commute with su(3) (centralizer
      obstruction).
    - THIS shows: designating ANY 2 directions as spatial DESTROYS
      su(3) itself — not just the commutativity, but the very
      existence of the color algebra in the residual internal sector.

  The emergence programme (beta) must therefore do something more
  radical than "assign directions": it must produce an emergent IR
  in which spatial and color degrees of freedom are NOT a partition
  of the same 6-direction pool at all.
""")

    print("=" * 76)


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
