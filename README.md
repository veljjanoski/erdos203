# Erdős problem #203 — search for a 2D covering system (negative outcome)

**Problem.** Is there an integer m ≥ 1 with gcd(m, 6) = 1 such that 2^k 3^l m + 1 is composite for all k, l ≥ 0?
(https://www.erdosproblems.com/203)

**Outcome.** Not resolved. No such m was found, and we do not prove that none exists. What this repository
contains is a systematic attempt at the only known constructive mechanism (a 2D covering system of primes,
combined with algebraic factorisations of a perfect-power m), the data it produced, and the structural reasons
why this mechanism appears to be out of reach with primes of small order.

## Setup

For a prime p ≥ 5 let H_p = ⟨2, 3⟩ ≤ F_p^*, e_p = |H_p| = lcm(ord_p 2, ord_p 3), h a generator of H_p,
2 = h^a, 3 = h^b. Then p | 2^k 3^l m + 1 for exactly the (k, l) in one coset
{a k + b l ≡ c (mod e_p)} of a lattice K_p ≤ Z² of index e_p (the coset c fixes m mod p).
A finite set of primes with chosen cosets proves the compositeness of 2^k 3^l m + 1 for all k, l
iff the cosets cover Z². Necessary: Σ 1/e_p ≥ 1 (AnimishSharma, comment of 21 Jun 2026).

Algebraic factorisations, if m = t^{N0}: for odd q | N0 the cells k ≡ l ≡ 0 (mod q) are free
(X^q + 1), and if 4 | N0 the cells (k, l) ≡ (2, 0) (mod 4) are free (4Y^4 + 1 = (2Y²+2Y+1)(2Y²−2Y+1),
used by Izotov for Sierpiński numbers). By Capelli's theorem these are the only factorisations of
2^k 3^l t^{N0} + 1 as a polynomial in t, so every known mechanism for such an m is a covering system,
possibly combined with these free sets. Using them restricts the admissible cosets: m ≡ −h^{−c} (mod p)
must be an N0-th power residue.

## What was done

1. `primes203.py` — for a modulus N, all primes p ≤ 10^7 with e_p | N and their (e, a, b, h).
   `fullpool.py`, `smallpool.py` — complete pools of any size via gcd(Φ_a(2), Φ_b(3)) for a, b | N
   (resp. lcm(a, b) ≤ E): every prime with e_p = lcm(a, b) divides this gcd. Results:
   all primes with e_p ≤ 2000: 442 primes, Σ 1/e_p = 1.973 (`primes_e2000.json`); all primes with e_p ≤ 5000:
   1006 primes, 2.146; e_p | 55440: 55 primes, 1.163 (complete); primes p ≤ 10^7 with e_p | 720720: 99 primes,
   1.249.
2. `cover203.py`, `strip_gpu.py` — GPU (CuPy) covering search on the torus Z_N², greedy + iterated local
   search over the coset choices, with or without the algebraic free sets (`--N0`).
   Best coverage found: N = 5040 (31 primes, density 1.021): 71.7 % (matches the 71–72 % reported in the
   thread); N = 55440 (55 primes, density 1.163): 76.0 %; with m a cube: 74 %; with m a 12th power: 70 %.
   Longer iterated local search does not move these plateaus (they are close to the random-choice value
   1 − exp(−Σ 1/e_p)).
3. `level203.py`, `maxJ.py` — a structured family. Substituting k' = k + J l, primes with
   (3·2^{−J})^d ≡ 1 (mod p) give cosets that depend on l only mod d, so the covering lives on a strip
   Z_N × Z_d and the lattices nest like 1D moduli. For every such family the density Σ 1/e_p over the
   admissible primes is computed exactly (branch and bound over J; `level203.py scanJ` for every divisor d).
   Rigorous consequence: for N = 720720 and the 99 primes with e_p | N, no strip covering exists for any J
   and any strip height d < 720 (the maximal density is < 1 for every divisor d < 720 of N, and the pool for
   a general d equals the pool for gcd(d, N)); at d = 720 the maximal density is 1.050 and the search leaves
   28 % uncovered. Over all 442 primes
   with e_p ≤ 2000 the maximal strip density is at least 0.62 (d = 1), 0.93 (d = 2), 1.13 (d = 12), 1.22 (d = 60)
   (branch and bound stopped by a time limit).
4. Structure of the lattices (why exact coverings are hard here). Exact coverings in 1D rely on nested
   moduli with small index ratios (2 | 4 | 8 | … | 64, with two primes of order 64). Among the 442 primes
   with e_p ≤ 2000: all 442 lattices K_p are distinct; only 77 pairs satisfy K_q ⊂ K_p, all with p ≤ 61 and
   index ratio e_q / e_p ≥ 12; the only primes with e_p a power of 2 are 5, 17 and 257, and none of their
   lattices contains another. So no chain of the Fermat type exists.

5. `strip_sample.py`, `maxdir.py` — strips with an astronomically large period (all 1006 primes with
   e_p ≤ 5000, Σ 1/e_p = 2.146, `primes_e5000.json`), where no bitmap is possible; coverage is estimated on
   120 000 random sample cells (validated against the exact bitmap at N = 720720: 27.6 % vs 27.9 %).
   Best admissible pools: d = 12: 216 primes, density 1.163 → 22.6 % uncovered; d = 60: 230 primes, 1.250 →
   20.1 %; other orientation (l' = l + J k): d = 12: 221 primes, 1.208 → 21.6 %; d = 60: 235 primes, 1.312 →
   18.7 %. With m a 12th power the residue constraint removes cosets and the result is worse (35 %).
   Every configuration plateaus close to the random-choice value 1 − exp(−Σ 1/e_p) minus a few percent.

## Conclusion

Within reach of computation, a 2D covering system for 2^k 3^l m + 1 does not exist, with or without the
perfect-power factorisations, and the lattice structure of the available primes makes one unlikely at any
modulus of practical size. This is evidence, not a proof; the problem remains open in both directions.

## Files

`primes_N*.json`, `primes_e2000.json` (prime pools), `best_N*.json`, `strip_*.json` (best coset choices
found), `*.log` (search logs).
