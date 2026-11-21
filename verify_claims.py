#!/usr/bin/env python3
"""Numerical verification of every concrete claim and example in main.tex.

Run:  python3 verify_claims.py
Exit status is 0 only if every check passes.  Requires numpy.
"""
import itertools
import sys

import numpy as np

rng = np.random.default_rng(12345)
failures = []


def check(name, condition):
    print(("PASS  " if condition else "FAIL  ") + name)
    if not condition:
        failures.append(name)


def polar(A):
    """Return (U, P, sigma) with A = U P, U orthogonal, P = (A^T A)^{1/2}."""
    V, s, Wt = np.linalg.svd(A)
    return V @ Wt, Wt.T @ np.diag(s) @ Wt, s


# ---------------------------------------------------------------- Section: reals
# Theorem 1: b -> (sgn b, |b|) is a group isomorphism R* -> {+-1} x (0, inf)
for b, c in rng.normal(size=(1000, 2)):
    check_ok = (np.sign(b * c) == np.sign(b) * np.sign(c)) and np.isclose(abs(b * c), abs(b) * abs(c))
    if not check_ok:
        failures.append("sgn/abs multiplicativity")
        break
else:
    check("Thm 1: sgn and |.| are multiplicative (1000 random pairs)", True)

# Proposition 2: magnitude classes and the factor |bx| = |b||x|
for b in [3, 0.5, 1, -1, -3, -0.5, 0]:
    for x in [-7.0, -0.3, 2.0, 5.0]:
        assert np.isclose(abs(b * x), abs(b) * abs(x))
check("Prop 2: |bx| = |b||x| for the example factors", True)

# Remark: for x<0 and b>1, bx < x although |bx| > |x|
check("Remark: b=2, x=-5 gives bx=-10 < x but |bx| > |x|", 2 * -5 < -5 and abs(2 * -5) > abs(-5))

# Original-paper axioms A1 / A2 are false for negative a (documented in the paper)
check("Original axiom A1 fails: a=-5,b=2 -> a*b < a", (-5) * 2 < -5)
check("Original axiom A2 fails: a=-4,b=.5 -> a*b > a", (-4) * 0.5 > -4)

# Sign rule: reflection composed with reflection is the identity
check("Sign rule: (-1)(-1) = 1 as m_{-1} o m_{-1} = id",
      all(np.isclose((-1) * ((-1) * x), x) for x in rng.normal(size=100)))

# Inverses: inverse of expansion is reduction; reflection is its own inverse
check("Inverse of b=3 is 1/3 (reduction); inverse of -1 is -1", np.isclose(3 * (1 / 3), 1) and (-1) * (-1) == 1)

# -------------------------------------------------------------- Section: complex
for b, label in [(1 + 1j, "1+i"), (-2 + 0j, "-2"), (1j, "i")]:
    r, th = abs(b), np.angle(b)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    M = np.array([[b.real, -b.imag], [b.imag, b.real]])
    check(f"Thm 3: m_b = scaling |b| o rotation arg(b) for b={label}", np.allclose(M, r * rot))
check("b=1+i: |b|=sqrt2 and arg=pi/4",
      np.isclose(abs(1 + 1j), np.sqrt(2)) and np.isclose(np.angle(1 + 1j), np.pi / 4))
check("b=-2: |b|=2 and arg=pi", np.isclose(abs(-2 + 0j), 2) and np.isclose(abs(np.angle(-2 + 0j)), np.pi))
check("-I in the plane is a rotation by pi with det +1",
      np.isclose(np.linalg.det(-np.eye(2)), 1) and np.allclose(-np.eye(2), [[np.cos(np.pi), -np.sin(np.pi)], [np.sin(np.pi), np.cos(np.pi)]]))

# ------------------------------------------------------------- Section: matrices
# Theorems (polar / SVD), random 4x4 matrices
ok = True
for _ in range(300):
    A = rng.normal(size=(4, 4))
    U, P, s = polar(A)
    ok &= np.allclose(U @ P, A) and np.allclose(U.T @ U, np.eye(4))
    ok &= np.allclose(P, P.T) and np.all(np.linalg.eigvalsh(P) > -1e-12)
    ok &= np.allclose(P @ P, A.T @ A)
    ok &= np.linalg.matrix_rank(A) == int((s > 1e-12).sum())
    ok &= np.sign(np.linalg.det(U)) == np.sign(np.linalg.det(A))
check("Polar/SVD/rank/determinant-sign identities (300 random 4x4)", bool(ok))

# Theorem 6: sigma_min |x| <= |Ax| <= sigma_max |x|, with equality attained
ok = True
for _ in range(300):
    A = rng.normal(size=(5, 5))
    V, s, Wt = np.linalg.svd(A)
    for x in rng.normal(size=(20, 5)):
        nx, nAx = np.linalg.norm(x), np.linalg.norm(A @ x)
        ok &= s[-1] * nx - 1e-9 <= nAx <= s[0] * nx + 1e-9
    ok &= np.isclose(np.linalg.norm(A @ Wt[0]), s[0]) and np.isclose(np.linalg.norm(A @ Wt[-1]), s[-1])
check("Thm 6: singular-value bounds on |Ax| and attainment (300 random 5x5)", bool(ok))

# Commutation iff normal
A = rng.normal(size=(3, 3))
U, P, _ = polar(A)
check("Non-normal A: UP != PU", not np.allclose(U @ P, P @ U))
S = A + A.T
U, P, _ = polar(S)
check("Symmetric (normal) A: UP = PU", np.allclose(U @ P, P @ U))
Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
U, P, _ = polar(Q)
check("Orthogonal (normal) A: UP = PU", np.allclose(U @ P, P @ U))

# Worked example 1: shear A = [[1,1],[0,1]]
phi = (1 + np.sqrt(5)) / 2
A = np.array([[1.0, 1.0], [0.0, 1.0]])
U, P, s = polar(A)
U_exact = np.array([[2, 1], [-1, 2]]) / np.sqrt(5)
P_exact = np.array([[2, 1], [1, 3]]) / np.sqrt(5)
check("Example shear: sigma = (phi, 1/phi)", np.allclose(s, [phi, 1 / phi]))
check("Example shear: det A = 1 = sigma1*sigma2", np.isclose(np.linalg.det(A), 1) and np.isclose(s[0] * s[1], 1))
check("Example shear: U = (1/sqrt5)[[2,1],[-1,2]]", np.allclose(U, U_exact))
check("Example shear: P = (1/sqrt5)[[2,1],[1,3]]", np.allclose(P, P_exact))
check("Example shear: UP = A", np.allclose(U_exact @ P_exact, A))
theta = np.degrees(np.arctan2(U_exact[1, 0], U_exact[0, 0]))
check("Example shear: rotation angle = -arctan(1/2) = -26.565 deg", np.isclose(theta, -np.degrees(np.arctan(0.5))))
PU = P_exact @ U_exact
check("Example shear: PU = (1/5)[[3,4],[-1,7]] != A", np.allclose(PU, np.array([[3, 4], [-1, 7]]) / 5) and not np.allclose(PU, A))
V, _, Wt = np.linalg.svd(A)
major_deg = np.degrees(np.arctan2(V[1, 0], V[0, 0])) % 180
print(f"      info: image ellipse major axis at {major_deg:.3f} deg; semi-axes {s[0]:.4f}, {s[1]:.4f}")
check("Example shear: major axis of image ellipse at arctan(1/phi) = 31.717 deg",
      np.isclose(major_deg, np.degrees(np.arctan(1 / phi))))

# Worked example 2: B = [[1,1],[1,1]] (nullification)
B = np.array([[1.0, 1.0], [1.0, 1.0]])
_, _, sB = polar(B)
check("Example B: sigma = (2, 0), rank 1", np.allclose(sB, [2, 0]) and np.linalg.matrix_rank(B) == 1)
check("Example B: collapses (1,-1)", np.allclose(B @ np.array([1.0, -1.0]), 0))

# Worked example 3: C = diag(3, -1/2)
C = np.diag([3.0, -0.5])
U, P, s = polar(C)
check("Example C: U = diag(1,-1), P = diag(3, 1/2), det U = -1",
      np.allclose(U, np.diag([1, -1])) and np.allclose(P, np.diag([3, 0.5])) and np.isclose(np.linalg.det(U), -1))

# Example: swap matrix is an isometric reflection
Sw = np.array([[0.0, 1.0], [1.0, 0.0]])
_, _, sS = polar(Sw)
check("Example swap: sigma=(1,1), det=-1 (isometric reflection)", np.allclose(sS, [1, 1]) and np.isclose(np.linalg.det(Sw), -1))

# Connectivity: det changes sign along any path from det>0 to det<0 (IVT), so a singular matrix is crossed
R = np.diag([1.0, -1.0])
crossed = False
for t in np.linspace(0, 1, 2001):
    M = (1 - t) * np.eye(2) + t * R          # straight-line path I -> R
    if abs(np.linalg.det(M)) < 1e-9:
        crossed = True
check("Path I -> diag(1,-1) passes through a singular matrix", crossed)
# whereas I -> -I can be done through rotations without ever being singular
dets = [np.linalg.det(np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])) for t in np.linspace(0, np.pi, 500)]
check("Rotation path I -> -I stays invertible (det = 1 throughout)", np.allclose(dets, 1))

# Uniqueness of the polar factor for invertible A
A = rng.normal(size=(4, 4))
U, P, _ = polar(A)
U2 = A @ np.linalg.inv(P)
check("Invertible A: U = A P^{-1} is forced", np.allclose(U, U2))

# Linear isometries of R: only +-1
check("Linear isometries of R are exactly q=+-1 (|q x|=|x| => |q|=1)",
      all(np.isclose(abs(q), 1) for q in [1.0, -1.0]) and not np.isclose(abs(2.0), 1))


# ---- additional checks for claims added in v1.1
# Prop 3(d): bx > x  iff  (b>1 and x>0) or (b<1 and x<0)
ok = True
for b, x in rng.normal(scale=3, size=(5000, 2)):
    ok &= (b * x > x) == ((b > 1 and x > 0) or (b < 1 and x < 0))
check("Prop 3(d): bx > x iff (b>1,x>0) or (b<1,x<0) (5000 random pairs)", bool(ok))

# Prop 3(b): inversion exchanges expansion and reduction, fixes isometry, preserves sign
ok = True
for b in rng.normal(scale=3, size=2000):
    inv = 1 / b
    ok &= (abs(b) > 1) == (abs(inv) < 1) and np.sign(inv) == np.sign(b)
check("Prop 3(b): 1/b swaps expansion/reduction and keeps the sign (2000 random)", bool(ok))

# det A = det U * prod(sigma); |det A| = prod(sigma); eigenvalues of P are the sigma_i
ok = True
for _ in range(300):
    A = rng.normal(size=(4, 4))
    U, P, s = polar(A)
    ok &= np.isclose(np.linalg.det(A), np.linalg.det(U) * np.prod(s))
    ok &= np.isclose(abs(np.linalg.det(A)), np.prod(s))
    ok &= np.allclose(np.sort(np.linalg.eigvalsh(P))[::-1], s)
check("det A = det U * prod sigma; |det A| = prod sigma; eig(P) = sigma (300 random 4x4)", bool(ok))

# Example B stretches (1,1) by 2
B = np.array([[1.0, 1.0], [1.0, 1.0]])
v = np.array([1.0, 1.0])
check("Example B: stretches (1,1) by exactly 2", np.isclose(np.linalg.norm(B @ v) / np.linalg.norm(v), 2))

# Complex-matrix remark: [[a,-b],[b,a]] has P = |b| I and U = rotation by arg(a+ib)
ok = True
for a, b in rng.normal(size=(200, 2)):
    M = np.array([[a, -b], [b, a]])
    U, P, s = polar(M)
    th = np.arctan2(b, a)
    ok &= np.allclose(P, np.hypot(a, b) * np.eye(2)) and np.allclose(U, [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
check("Remark: [[a,-b],[b,a]] has P=|z|I and U=rotation by arg z (200 random)", bool(ok))

# Figure data: A e2 = (1,1) lies on the image ellipse (semi-axes phi, 1/phi, rotated 31.717 deg)
th = np.radians(np.degrees(np.arctan(1 / phi)))
R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])  # rotate by -th
for pt in [(1.0, 0.0), (1.0, 1.0)]:
    u, w = R @ np.array(pt)
    check(f"Figure 3: image point {pt} lies on the drawn ellipse", np.isclose((u / phi) ** 2 + (w * phi) ** 2, 1))

# Quaternion remark is not numerically checked (statement is classical); nothing to assert.

print()
if failures:
    print(f"{len(failures)} check(s) FAILED:", *failures, sep="\n  - ")
    sys.exit(1)
print("All checks passed.")
