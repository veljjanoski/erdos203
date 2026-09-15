"""Structured ("level d") covering search for Erdős #203.

Substitute k' = k + J l.  For a prime p with e_p | N, let zeta_p = 3 * 2^{-J} mod p.  If zeta_p^d = 1 then
2^k 3^l = 2^{k'} zeta_p^l, so the cells killed by p (coset c_p) are  a_p k' + w_p l == c_p (mod e_p), zeta_p = h^{w_p},
and this depends on l only modulo d.  So the covering problem lives on the torus Z_N x Z_d (N x d cells).
Algebraic sets (m = t^{N0}) in (k', l) coordinates:  q | N0 odd: k' == l == 0 (mod q);  4 | N0: k' == 2, l == 0 (mod 4).
Usage:
  python level203.py scanJ N d              -> best J values (pool density) for given N, d
  python level203.py cover N d J [--N0 n] [--rounds r] [--restarts s]
"""
import json, sys, time, argparse
from math import gcd
import numpy as np
from cover203 import allowed_cosets

def prime_pool(N, d, J, N0):
    P = json.load(open(f'primes_N{N}.json'))
    pool = []
    for dd in P:
        p, e, a, b, h = dd['p'], dd['e'], dd['a'], dd['b'], dd['h']
        zeta = 3 * pow(pow(2, J, p), -1, p) % p
        if pow(zeta, d, p) != 1: continue
        # w with h^w = zeta
        tbl = {}; x = 1
        for i in range(e): tbl[x] = i; x = x * h % p
        w = tbl[zeta]
        al = allowed_cosets(dd, N0)
        if not al: continue
        pool.append(dict(p=p, e=e, a=a, w=w, h=h, allowed=al, n=e // gcd(a, e)))
    return pool

def scanJ(N, d, top=10):
    P = json.load(open(f'primes_N{N}.json'))
    dens = np.zeros(N)
    for dd in P:
        p, e, a, b = dd['p'], dd['e'], dd['a'], dd['b']
        g = gcd(d, e); step = e // g
        # J admissible iff 2^{J d} == 3^d  iff  J a d == b d (mod e)  iff  J a == b (mod e/g)
        M = step; aa, bb = a % M, b % M
        if M == 1: dens += 1 / e; continue
        gg = gcd(aa, M)
        if bb % gg: continue
        M2 = M // gg; r = (bb // gg) * pow(aa // gg, -1, M2) % M2
        dens[r::M2] += 1 / e
    order = np.argsort(-dens)[:top]
    return [(int(J), float(dens[J])) for J in order], dens

class Strip:
    """covering state on Z_N x Z_d"""
    def __init__(self, N, d, pool, N0):
        self.N, self.d, self.pool = N, d, pool
        self.cnt = np.zeros((d, N), dtype=np.uint8)
        self.kk = np.arange(N, dtype=np.int64)[None, :]; self.ll = np.arange(d, dtype=np.int64)[:, None]
        for q in (3, 5, 7, 11, 13):
            if N0 % q == 0:
                assert d % q == 0 and N % q == 0
                self.cnt[0::q, 0::q] += 1
        if N0 % 4 == 0:
            assert d % 4 == 0 and N % 4 == 0
            self.cnt[0::4, 2::4] += 1
        self.choice = {}
        self.v = {}
    def vals(self, dd):
        if dd['p'] not in self.v:
            self.v[dd['p']] = ((dd['a'] * self.kk + dd['w'] * self.ll) % dd['e']).astype(np.int32)
        return self.v[dd['p']]
    def apply(self, dd, c, delta):
        m = self.vals(dd) == c
        if delta > 0: self.cnt[m] += 1
        else: self.cnt[m] -= 1
    def best(self, dd):
        v = self.vals(dd); hist = np.bincount(v[self.cnt == 0], minlength=dd['e'])
        al = dd['allowed']; c = al[int(np.argmax(hist[al]))]; return c, int(hist[c])
    def uncovered(self): return int((self.cnt == 0).sum())

def cover(N, d, J, N0, rounds, restarts, seed, verbose=True):
    pool = prime_pool(N, d, J, N0)
    dens = sum(1 / dd['e'] for dd in pool)
    total = N * d
    rng = np.random.default_rng(seed)
    S = Strip(N, d, pool, N0)
    if verbose: print(f"N={N} d={d} J={J} N0={N0}: pool {len(pool)} primes, density {dens:.3f}, algebraic free {1 - S.uncovered()/total:.3f}", flush=True)
    bestu = None; bestchoice = None
    for rs in range(restarts):
        S.cnt[:] = 0
        for q in (3, 5, 7, 11, 13):
            if N0 % q == 0: S.cnt[0::q, 0::q] += 1
        if N0 % 4 == 0: S.cnt[0::4, 2::4] += 1
        S.choice = {}
        order = sorted(pool, key=lambda dd: -dd['e']) if rs == 0 else list(rng.permutation(pool))
        for dd in order:
            c, g = S.best(dd); S.apply(dd, c, +1); S.choice[dd['p']] = c
        for r in range(rounds):
            improved = False
            for dd in rng.permutation(pool):
                S.apply(dd, S.choice[dd['p']], -1)
                c, g = S.best(dd); S.apply(dd, c, +1)
                if c != S.choice[dd['p']]: S.choice[dd['p']] = c; improved = True
            if not improved: break
        u = S.uncovered()
        if bestu is None or u < bestu: bestu, bestchoice = u, dict(S.choice)
        if verbose: print(f"  restart {rs}: uncovered {u} / {total} = {u/total:.4f}", flush=True)
        if u == 0: break
    return pool, bestu, bestchoice

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd'); ap.add_argument('N', type=int); ap.add_argument('d', type=int)
    ap.add_argument('J', type=int, nargs='?', default=0); ap.add_argument('--N0', type=int, default=1)
    ap.add_argument('--rounds', type=int, default=30); ap.add_argument('--restarts', type=int, default=5); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--top', type=int, default=10)
    a = ap.parse_args()
    if a.cmd == 'scanJ':
        best, dens = scanJ(a.N, a.d, a.top)
        print(f"N={a.N} d={a.d}: max pool density {best[0][1]:.3f}; mean {dens.mean():.3f}")
        for J, v in best: print(f"  J={J}: {v:.3f}")
    elif a.cmd == 'cover':
        pool, u, ch = cover(a.N, a.d, a.J, a.N0, a.rounds, a.restarts, a.seed)
        if u == 0:
            json.dump(dict(N=a.N, d=a.d, J=a.J, N0=a.N0, choice=ch), open(f'COVER_N{a.N}_d{a.d}_J{a.J}_N0{a.N0}.json', 'w'))
            print("COVERED!")
