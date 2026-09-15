"""Strip covering search with an astronomically large period, coverage estimated on random sample cells.
Strip Z x Z_d; prime p (zeta_p = 3*2^{-J} = h^{w_p}, zeta_p^d = 1) with coset c kills (k', l) iff a k' + w l == c (mod e).
Cells are sampled: k' uniform in [0, 2^62), l in 0..d-1.  Objective: estimated uncovered fraction.
Usage: python strip_sample.py bestJ.json [--N0 1] [--samples 60000] [--iters 300] [--kick 3] [--seed 0]
"""
import json, sys, time, argparse
from math import gcd
import numpy as np
from cover203 import allowed_cosets

def build_pool(pool_file, J, d, N0):
    P = json.load(open(pool_file)); pool = []
    for dd in P:
        p, e, a, b, h = dd['p'], dd['e'], dd['a'], dd['b'], dd['h']
        if pool_file.endswith('_swapped.json'):
            # swapped orientation: roles of 2 and 3 exchanged (a<->b already swapped in file); zeta = 2*3^{-J}
            zeta = 2 * pow(pow(3, J % e, p), -1, p) % p
        else:
            zeta = 3 * pow(pow(2, J % e, p), -1, p) % p
        if pow(zeta, d, p) != 1: continue
        tbl = {}; x = 1
        for i in range(e): tbl[x] = i; x = x * h % p
        w = tbl[zeta]
        al = allowed_cosets(dd, N0)
        if not al: continue
        pool.append(dict(p=p, e=e, a=a, w=w, allowed=np.array(al)))
    return pool

class Sampler:
    def __init__(self, pool, d, N0, samples, rng):
        self.pool, self.d = pool, d
        self.k = rng.integers(0, 2**62, size=samples, dtype=np.int64)
        self.l = rng.integers(0, d, size=samples, dtype=np.int64)
        self.n = samples
        # base coverage from algebraic sets (in (k', l) coordinates: k' == l == 0 mod q; (k',l) == (2,0) mod 4)
        self.base = np.zeros(samples, dtype=np.uint8)
        for q in (3, 5, 7, 11, 13):
            if N0 % q == 0: self.base += ((self.k % q == 0) & (self.l % q == 0))
        if N0 % 4 == 0: self.base += ((self.k % 4 == 2) & (self.l % 4 == 0))
        self.v = {}
        for dd in pool:   # coset value of each sample cell for prime p
            self.v[dd['p']] = ((dd['a'] * (self.k % dd['e']) + dd['w'] * self.l) % dd['e']).astype(np.int32)
        self.cnt = self.base.copy(); self.choice = {}
    def reset(self): self.cnt = self.base.copy(); self.choice = {}
    def set_choice(self, dd, c):
        p = dd['p']; v = self.v[p]
        if p in self.choice: self.cnt -= (v == self.choice[p])
        self.cnt += (v == c); self.choice[p] = c
    def best(self, dd):
        v = self.v[dd['p']]; hist = np.bincount(v[self.cnt == 0], minlength=dd['e'])
        al = dd['allowed']; c = int(al[np.argmax(hist[al])]); return c, int(hist[c])
    def uncovered(self): return int((self.cnt == 0).sum())
    def local_search(self, rng, maxrounds=60):
        for r in range(maxrounds):
            improved = False
            for dd in rng.permutation(self.pool):
                p = dd['p']; old = self.choice.get(p)
                if old is not None: self.cnt -= (self.v[p] == old)
                c, g = self.best(dd); self.cnt += (self.v[p] == c); self.choice[p] = c
                if c != old: improved = True
            if not improved: break
        return self.uncovered()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('bestJ'); ap.add_argument('--N0', type=int, default=1)
    ap.add_argument('--samples', type=int, default=60000); ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--kick', type=int, default=3); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    B = json.load(open(a.bestJ)); J = int(B['J']); d = B['d']
    pool = build_pool(B['pool'], J, d, a.N0)
    dens = sum(1 / dd['e'] for dd in pool)
    rng = np.random.default_rng(a.seed)
    S = Sampler(pool, d, a.N0, a.samples, rng); t0 = time.time()
    print(f"{B['pool']} d={d} N0={a.N0}: pool {len(pool)} primes, density {dens:.4f}, alg free {(S.base>0).mean():.3f}, samples {a.samples}", flush=True)
    for dd in sorted(pool, key=lambda x: -x['e']):
        c, g = S.best(dd); S.set_choice(dd, c)
    u = S.local_search(rng); best = u; bestchoice = dict(S.choice)
    print(f"greedy+LS: uncovered {u/S.n:.4f} [{time.time()-t0:.0f}s]", flush=True)
    for it in range(a.iters):
        saved = dict(S.choice)
        for dd in rng.choice(pool, size=min(a.kick, len(pool)), replace=False):
            S.set_choice(dd, int(rng.choice(dd['allowed'])))
        u = S.local_search(rng)
        if u <= best:
            if u < best: print(f"iter {it}: uncovered {u/S.n:.4f} [{time.time()-t0:.0f}s]", flush=True)
            best = u; bestchoice = dict(S.choice)
        else:
            for dd in pool: S.set_choice(dd, saved[dd['p']])
    print(f"FINAL: estimated uncovered {best/S.n:.4f} [{time.time()-t0:.0f}s]")
    json.dump(dict(bestJ=a.bestJ, N0=a.N0, est_uncovered=best / S.n, choice={str(k): v for k, v in bestchoice.items()}),
              open(a.bestJ.replace('bestJ_', 'sample_').replace('.json', f'_N0{a.N0}.json'), 'w'))
if __name__ == '__main__': main()
