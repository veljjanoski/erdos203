"""GPU covering search on the strip Z_N x Z_d (see level203.py for the maths), with iterated local search.
Usage: python strip_gpu.py primes.json N d J [--N0 1] [--iters 200] [--kick 3] [--seed 0]
Cell (l, k') killed by prime p with coset c iff  a_p k' + w_p l == c (mod e_p),  zeta_p = 3*2^{-J} = h^{w_p}, zeta_p^d = 1.
"""
import json, sys, time, argparse
from math import gcd
import numpy as np, cupy as cp
from cover203 import _hist, _apply, allowed_cosets

def prime_pool(P, N, d, J, N0):
    pool = []
    for dd in P:
        p, e, a, b, h = dd['p'], dd['e'], dd['a'], dd['b'], dd['h']
        n = e // gcd(a, e)
        if N % n: continue
        zeta = 3 * pow(pow(2, J % e, p), -1, p) % p   # 2^J mod p depends on J mod e (2 has order n | e)
        if pow(zeta, d, p) != 1: continue
        tbl = {}; x = 1
        for i in range(e): tbl[x] = i; x = x * h % p
        w = tbl[zeta]
        al = allowed_cosets(dd, N0)
        if not al: continue
        pool.append(dict(p=p, e=e, a=a, w=w, h=h, allowed=al, n=n))
    return pool

class Strip:
    def __init__(self, N, d, pool, N0):
        self.N, self.d, self.pool, self.N0 = N, d, pool, N0
        self.cnt = cp.zeros((d, N), dtype=cp.uint8)
        self.vec = {}
        for dd in pool:
            e = dd['e']
            self.vec[dd['p']] = (cp.asarray((dd['w'] * np.arange(d, dtype=np.int64)) % e, dtype=cp.int32),
                                 cp.asarray((dd['a'] * np.arange(N, dtype=np.int64)) % e, dtype=cp.int32))
        self.grid = (min(64, (N + 255) // 256), d); self.block = (256,)
        self.choice = {}
        self.reset()
    def reset(self):
        self.cnt[:] = 0; self.choice = {}
        for q in (3, 5, 7, 11, 13):
            if self.N0 % q == 0: self.cnt[0::q, 0::q] += 1
        if self.N0 % 4 == 0: self.cnt[0::4, 2::4] += 1
    def apply(self, dd, c, delta):
        rv, cv = self.vec[dd['p']]
        _apply(self.grid, self.block, (self.cnt, rv, cv, np.int32(self.N), np.int32(dd['e']), np.int32(c), np.int32(delta)))
    def best(self, dd):
        e = dd['e']; rv, cv = self.vec[dd['p']]
        hist = cp.zeros(e, dtype=cp.int32)
        _hist(self.grid, self.block, (self.cnt, rv, cv, np.int32(self.N), np.int32(e), hist), shared_mem=(4 * e if e <= 8192 else 0))
        hist = hist.get(); al = dd['allowed']
        c = al[int(np.argmax(hist[al]))]; return c, int(hist[c])
    def uncovered(self): return int((self.cnt == 0).sum())
    def set_choice(self, dd, c):
        p = dd['p']
        if p in self.choice: self.apply(dd, self.choice[p], -1)
        self.apply(dd, c, +1); self.choice[p] = c
    def local_search(self, rng, maxrounds=50):
        for r in range(maxrounds):
            improved = False
            for dd in rng.permutation(self.pool):
                old = self.choice.get(dd['p'])
                if old is not None: self.apply(dd, old, -1)
                c, g = self.best(dd); self.apply(dd, c, +1); self.choice[dd['p']] = c
                if c != old: improved = True
            if not improved: break
        return self.uncovered()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('primes'); ap.add_argument('N', type=int); ap.add_argument('d', type=int); ap.add_argument('J', type=int)
    ap.add_argument('--N0', type=int, default=1); ap.add_argument('--iters', type=int, default=200); ap.add_argument('--kick', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    P = json.load(open(a.primes))
    pool = prime_pool(P, a.N, a.d, a.J, a.N0)
    dens = sum(1 / dd['e'] for dd in pool); total = a.N * a.d
    S = Strip(a.N, a.d, pool, a.N0); rng = np.random.default_rng(a.seed); t0 = time.time()
    print(f"N={a.N} d={a.d} J={a.J} N0={a.N0}: pool {len(pool)} primes, density {dens:.4f} (+alg {1 - S.uncovered()/total:.3f}); cells {total:.2e}", flush=True)
    for dd in sorted(pool, key=lambda x: -x['e']):
        c, g = S.best(dd); S.set_choice(dd, c)
    u = S.local_search(rng); best = u; bestchoice = dict(S.choice)
    print(f"greedy+LS: uncovered {u} = {u/total:.4e} [{time.time()-t0:.0f}s]", flush=True)
    for it in range(a.iters):
        if best == 0: break
        # kick: re-randomise a few primes' cosets, then local search; accept if not worse
        saved = dict(S.choice)
        for dd in rng.choice(pool, size=min(a.kick, len(pool)), replace=False):
            S.set_choice(dd, int(rng.choice(dd['allowed'])))
        u = S.local_search(rng)
        if u <= best:
            if u < best: print(f"iter {it}: uncovered {u} = {u/total:.4e} [{time.time()-t0:.0f}s]", flush=True)
            best = u; bestchoice = dict(S.choice)
        else:
            for dd in pool: S.set_choice(dd, saved[dd['p']])
    print(f"FINAL: uncovered {best} / {total} = {best/total:.4e}  [{time.time()-t0:.0f}s]")
    json.dump(dict(N=a.N, d=a.d, J=a.J, N0=a.N0, uncovered=best, choice=bestchoice, primes=[dd['p'] for dd in pool]),
              open(f"strip_N{a.N}_d{a.d}_J{a.J}_N0{a.N0}.json", 'w'))
    if best == 0: print("COVERED!")

if __name__ == '__main__': main()
