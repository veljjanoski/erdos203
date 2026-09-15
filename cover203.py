"""GPU covering search for Erdős #203 on the torus Z_N^2 (N = lcm of the e_p used).

Cell (k,l) is killed by prime p (with chosen coset c_p) iff a_p k + b_p l == c_p (mod e_p); see primes203.py.
State: cnt[k, l] = number of chosen cosets (plus algebraic sets) covering cell (k, l), uint8.  Goal: no zero cell.
Greedy: for each prime pick the coset c maximising the number of currently-uncovered cells it hits.
Then local search: repeatedly remove one prime's coset and re-pick it optimally.
Usage: python cover203.py N [--N0 12] [--order asc|desc|random] [--rounds R] [--seed s]
"""
import json, sys, time, argparse
from math import gcd
import numpy as np, cupy as cp

KSRC = r'''
extern "C" {
// hist[c] += number of uncovered cells (cnt==0) with rowv[k] + colv[l] (mod e) == c ; rowv/colv already reduced mod e
__global__ void hist_uncovered(const unsigned char* cnt, const int* rowv, const int* colv, int N, int e, int* hist) {
    extern __shared__ int sh[];
    bool use_sh = (e <= 8192);
    if (use_sh) { for (int i = threadIdx.x; i < e; i += blockDim.x) sh[i] = 0; __syncthreads(); }
    int k = blockIdx.y; int rv = rowv[k];
    for (int l = blockIdx.x * blockDim.x + threadIdx.x; l < N; l += gridDim.x * blockDim.x) {
        if (cnt[(size_t)k * N + l] == 0) {
            int v = rv + colv[l]; if (v >= e) v -= e;
            if (use_sh) atomicAdd(&sh[v], 1); else atomicAdd(&hist[v], 1);
        }
    }
    if (use_sh) { __syncthreads(); for (int i = threadIdx.x; i < e; i += blockDim.x) if (sh[i]) atomicAdd(&hist[i], sh[i]); }
}
// cnt[k,l] += delta on the coset rowv[k] + colv[l] == c (mod e)
__global__ void apply_coset(unsigned char* cnt, const int* rowv, const int* colv, int N, int e, int c, int delta) {
    int k = blockIdx.y; int rv = rowv[k];
    for (int l = blockIdx.x * blockDim.x + threadIdx.x; l < N; l += gridDim.x * blockDim.x) {
        int v = rv + colv[l]; if (v >= e) v -= e;
        if (v == c) cnt[(size_t)k * N + l] += delta;
    }
}
}'''
_mod = cp.RawModule(code=KSRC)
_hist = _mod.get_function('hist_uncovered'); _apply = _mod.get_function('apply_coset')

def allowed_cosets(d, N0):
    """cosets c for which m = -h^{-c} (mod p) is an N0-th power residue mod p (needed since m = t^N0)"""
    p, e, h = d['p'], d['e'], d['h']
    g = gcd(N0, p - 1); ex = (p - 1) // g
    hinv = pow(h, -1, p)
    return [c for c in range(e) if pow((-pow(hinv, c, p)) % p, ex, p) == 1]

def load(N, N0, fname=None):
    P = json.load(open(fname or f'primes_N{N}.json'))
    for d in P: d['allowed'] = allowed_cosets(d, N0)
    P = [d for d in P if d['allowed']]
    alg = []
    for q in (3, 5, 7, 11, 13):
        if N0 % q == 0: alg.append(('pow', q))
    if N0 % 4 == 0: alg.append(('sg', 4))
    return P, alg

class Cover:
    def __init__(self, N, P, alg):
        self.N, self.P = N, P
        self.cnt = cp.zeros((N, N), dtype=cp.uint8)
        self.choice = {}
        self.vec = {}
        for d in P:
            a, b, e = d['a'], d['b'], d['e']
            self.vec[d['p']] = (cp.asarray((a * np.arange(N, dtype=np.int64)) % e, dtype=cp.int32),
                                cp.asarray((b * np.arange(N, dtype=np.int64)) % e, dtype=cp.int32))
        for kind, q in alg:
            if kind == 'pow': self.cnt[cp.ix_(cp.arange(0, N, q), cp.arange(0, N, q))] += 1
            else: self.cnt[cp.ix_(cp.arange(2, N, 4), cp.arange(0, N, 4))] += 1
        self.grid = (min(64, (N + 255) // 256), N); self.block = (256,)
    def apply(self, d, c, delta):
        rv, cv = self.vec[d['p']]
        _apply(self.grid, self.block, (self.cnt, rv, cv, np.int32(self.N), np.int32(d['e']), np.int32(c), np.int32(delta)))
    def best_coset(self, d):
        e = d['e']; rv, cv = self.vec[d['p']]
        hist = cp.zeros(e, dtype=cp.int32)
        shmem = 4 * e if e <= 8192 else 0
        _hist(self.grid, self.block, (self.cnt, rv, cv, np.int32(self.N), np.int32(e), hist), shared_mem=shmem)
        hist = hist.get(); al = d['allowed']
        c = al[int(np.argmax(hist[al]))]; return c, int(hist[c])
    def uncovered(self):
        return int((self.cnt == 0).sum())

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('N', type=int); ap.add_argument('--N0', type=int, default=12)
    ap.add_argument('--order', default='desc'); ap.add_argument('--rounds', type=int, default=20); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--maxprimes', type=int, default=0); ap.add_argument('--primes', default=None)
    a = ap.parse_args()
    P, alg = load(a.N, a.N0, a.primes)
    P = sorted(P, key=lambda d: d['e'], reverse=(a.order == 'desc'))
    if a.maxprimes: P = P[:a.maxprimes]
    rng = np.random.default_rng(a.seed)
    if a.order == 'random': rng.shuffle(P)
    C = Cover(a.N, P, alg)
    total = a.N * a.N; t0 = time.time()
    print(f"N={a.N}, {len(P)} primes, density {sum(1/d['e'] for d in P):.3f}, algebraic {alg}; initial uncovered {C.uncovered()/total:.4f}", flush=True)
    for d in P:
        c, gain = C.best_coset(d); C.apply(d, c, +1); C.choice[d['p']] = c
    print(f"greedy: uncovered {C.uncovered()} / {total} = {C.uncovered()/total:.3e}  [{time.time()-t0:.0f}s]", flush=True)
    best = C.uncovered()
    for r in range(a.rounds):
        improved = False
        for d in P:
            C.apply(d, C.choice[d['p']], -1)
            c, gain = C.best_coset(d); C.apply(d, c, +1)
            if c != C.choice[d['p']]: C.choice[d['p']] = c; improved = True
        u = C.uncovered()
        print(f"round {r+1}: uncovered {u} = {u/total:.3e}  [{time.time()-t0:.0f}s]", flush=True)
        if u < best: best = u; json.dump(dict(N=a.N, N0=a.N0, uncovered=u, choice=C.choice), open(f'best_N{a.N}_N0{a.N0}.json', 'w'))
        if u == 0: print("COVERED"); break
        if not improved: print("local optimum"); break

if __name__ == '__main__':
    main()
