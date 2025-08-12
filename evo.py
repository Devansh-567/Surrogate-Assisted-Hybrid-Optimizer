import numpy as np
import random
from functools import partial
from time import time
from copy import deepcopy
from math import sqrt, log, exp
from collections import defaultdict

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import qmc, norm

# ---------------------------
# Utilities for mixed variables
# ---------------------------
class Space:
    """
    Mixed-variable search space specification and encoding helpers.
    variables: list of dicts each with:
      - name: str
      - type: 'continuous' or 'categorical' or 'integer'
      - bounds: (low, high) for continuous/integer OR list of categories for categorical
      - transform: optional callable to transform real->model (e.g., log) (not used here but hookable)
    Example:
      [
        {'name':'x1','type':'continuous','bounds':(-5,5)},
        {'name':'x2','type':'integer','bounds':(0,10)},
        {'name':'cat','type':'categorical','bounds':['red','green','blue']}
      ]
    """
    def __init__(self, variables):
        self.vars = variables
        # Build continuous bounds array for continuous/integer encoded into reals
        self.cont_indices = []
        self.cat_indices = []
        self.cat_sizes = []
        self.names = []
        for i,v in enumerate(self.vars):
            self.names.append(v.get('name', f'var{i}'))
            if v['type'] in ('continuous','integer'):
                self.cont_indices.append(i)
            elif v['type'] == 'categorical':
                self.cat_indices.append(i)
                self.cat_sizes.append(len(v['bounds']))
            else:
                raise ValueError("Unsupported var type: " + str(v['type']))
        # Create encoders for categorical values to one-hot
        if len(self.cat_indices) > 0:
            categories = [self.vars[i]['bounds'] for i in self.cat_indices]
            self._ohe = OneHotEncoder(categories=categories, sparse=False, handle_unknown='error')
            # Fit requires example rows; create identity-coded rows
            sample = []
            for cats in categories:
                sample.append([cats[0]])
            # Actually call fit with cartesian product? Simpler: use categories param only, sklearn will handle fit when transform used.
            # We'll call fit with minimal dummy to set categories from parameter via OneHotEncoder(categories=...)
            self._ohe.fit(np.array([[cats[0] for cats in categories]]))
        else:
            self._ohe = None

    def dim_cont(self):
        return len(self.cont_indices)

    def dim_ohe(self):
        return sum(self.cat_sizes) if self.cat_sizes else 0

    def encode(self, x):
        """
        x: list or dict of variable values in user form.
        Returns a 1D numpy array: continuous parts then one-hot encoded categorical parts.
        For integers, encode as continuous but round when decoding.
        """
        if isinstance(x, dict):
            vals = [x[name] for name in self.names]
        else:
            vals = x
        cont = []
        cats = []
        for i,v in enumerate(self.vars):
            if v['type'] in ('continuous','integer'):
                lo,hi = v['bounds']
                cont.append(float(vals[i]))
            else:
                cats.append([vals[i]])
        cont = np.array(cont) if len(cont)>0 else np.empty((0,))
        if self._ohe is not None:
            ohe = self._ohe.transform(np.array(cats).T.reshape(1,-1))[0] if len(cats)>0 else np.empty((0,))
            # Note: OneHotEncoder expects shape (n_samples, n_features); our usage is a little tricky: we instead build manually below
            # Simpler: manually encode categories
            ohe = []
            for idx in self.cat_indices:
                categories = self.vars[idx]['bounds']
                val = vals[idx]
                vec = [1.0 if val==c else 0.0 for c in categories]
                ohe.extend(vec)
            ohe = np.array(ohe) if len(ohe)>0 else np.empty((0,))
        else:
            ohe = np.empty((0,))
        return np.concatenate([cont, ohe])

    def decode(self, z):
        """
        z: encoded 1D array
        returns list of original-typed values (integers are rounded/clipped)
        """
        cont_len = self.dim_cont()
        cont = z[:cont_len] if cont_len>0 else np.array([])
        ohe = z[cont_len:] if len(z)>cont_len else np.array([])
        out = []
        ci = 0
        oi = 0
        for i,v in enumerate(self.vars):
            if v['type'] in ('continuous','integer'):
                val = cont[ci]
                ci += 1
                lo,hi = v['bounds']
                # clip
                val = float(np.clip(val, lo, hi))
                if v['type']=='integer':
                    val = int(round(val))
                out.append(val)
            else:
                # reconstruct categorical from one-hot block
                cats = v['bounds']
                size = len(cats)
                block = ohe[oi:oi+size]
                oi += size
                if len(block)==0:
                    out.append(cats[0])
                else:
                    idx = int(np.argmax(block))
                    out.append(cats[idx])
        return out

    def sample_lhs(self, n):
        """
        Return n samples in the encoded space (float array) using LHS for continuous dims and random choices for categories.
        """
        cont_n = self.dim_cont()
        sampler = qmc.LatinHypercube(d=cont_n, seed=None)
        if cont_n>0:
            uni = sampler.random(n=n)
            cont_lo = np.array([self.vars[i]['bounds'][0] for i in self.cont_indices])
            cont_hi = np.array([self.vars[i]['bounds'][1] for i in self.cont_indices])
            cont = qmc.scale(uni, cont_lo, cont_hi)
        else:
            cont = np.empty((n,0))
        # categories: random picks for each categorical var
        ohe_list = []
        for _ in range(n):
            cats_vec = []
            for idx in self.cat_indices:
                choices = self.vars[idx]['bounds']
                pick = random.choice(choices)
                # one-hot encode
                cats_vec.extend([1.0 if pick==c else 0.0 for c in choices])
            ohe_list.append(cats_vec)
        ohe = np.array(ohe_list) if len(ohe_list)>0 else np.empty((n,0))
        return np.hstack([cont, ohe])

# ---------------------------
# CMA-ES style mutation engine
# ---------------------------
class SimpleCMA:
    """
    Lightweight covariance-adaptive mutation engine.
    Maintains mean, step-size sigma, covariance matrix C (diagonal + low-rank approx optional).
    This is not a full CMA-ES implementation but captures major benefits: correlated mutations and adaptive sigma.
    """
    def __init__(self, dim, bounds, seed=None):
        self.dim = dim
        self.bounds = np.array(bounds, dtype=float)
        self.mean = np.zeros(dim)
        self.sigma = 0.3  # relative to bounds
        self.C = np.eye(dim)
        self.pc = np.zeros(dim)  # evolution path
        self.ps = np.zeros(dim)
        # strategy parameters (defaults inspired by CMA-ES)
        self.mu = max(1, dim//2)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu+1))
        self.weights = self.weights / np.sum(self.weights)
        self.cc = (4 + self.mu/dim) / (dim + 4 + 2*self.mu/dim)
        self.cs = (self.mu + 2) / (dim + self.mu + 5)
        self.damps = 1 + 2*max(0, sqrt((self.mu-1)/(dim+1)) - 1) + self.cs
        if seed is not None:
            np.random.seed(seed)
        # initialize mean at center of bounds
        lows = self.bounds[:,0]
        highs = self.bounds[:,1]
        self.mean = (lows + highs)/2.0

    def ask(self, n):
        """Generate n candidates from current multivariate normal (mean, sigma^2 * C)"""
        L = np.linalg.cholesky(self.C + 1e-8*np.eye(self.dim))
        z = np.random.normal(size=(n, self.dim))
        x = self.mean + self.sigma * (z.dot(L.T))
        # clip to bounds
        lo = self.bounds[:,0]
        hi = self.bounds[:,1]
        x = np.clip(x, lo, hi)
        return x

    def tell(self, solutions, fitnesses):
        """
        Update internal mean, sigma, and covariance using selected top mu solutions.
        solutions: array (n,dim)
        fitnesses: array (n,) higher is better
        """
        # sort by fitness descending
        idx = np.argsort(fitnesses)[::-1]
        selected = solutions[idx[:self.mu]]
        # compute new mean as weighted recombination
        new_mean = np.sum(selected.T * self.weights, axis=1)
        y = (new_mean - self.mean) / (self.sigma + 1e-12)
        # update evolution paths
        # For brevity, use simplified updates
        self.ps = (1 - self.cs) * self.ps + sqrt(self.cs*(2-self.cs)*self.mu) * (y / np.sqrt(np.diag(self.C)+1e-12))
        # adapt sigma
        norm_ps = np.linalg.norm(self.ps)
        self.sigma *= np.exp((self.cs / self.damps) * (norm_ps / (np.sqrt(self.dim)) - 1))
        # update covariance (rank-one + rank-mu simplified)
        c1 = 2.0 / ((self.dim+1.3)**2 + self.mu)
        cmu = min(1 - c1, 2*(self.mu-2+1)/((self.dim+2)**2 + self.mu))
        # rank-one update
        self.C = (1 - c1 - cmu) * self.C + c1 * np.outer(y, y)
        # rank-mu update with selected solutions' deviations
        for k, w in enumerate(self.weights):
            dk = (selected[k] - self.mean) / (self.sigma + 1e-12)
            self.C += cmu * w * np.outer(dk, dk)
        self.mean = new_mean

# ---------------------------
# Surrogate ensemble
# ---------------------------
class SurrogateEnsemble:
    """
    Trains a GP, RandomForest and MLP surrogate, then stacks them by cross-validated performance (or fit weights).
    Supports predicting mean and std (where only GP gives std; other models we estimate via residuals).
    Multi-fidelity: if data contains 'fidelity' column (0..F-1), we train separate surrogates per fidelity and stack.
    """
    def __init__(self, gp_kernel=None, rf_params=None, mlp_params=None, gp_opts=None, random_state=None):
        self.gp_kernel = gp_kernel or (ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(1e-6))
        self.rf_params = rf_params or {'n_estimators':100, 'random_state':random_state}
        self.mlp_params = mlp_params or {'hidden_layer_sizes':(100,100), 'max_iter':500, 'random_state':random_state}
        self.gp_opts = gp_opts or {'normalize_y':True, 'n_restarts_optimizer':2}
        self.random_state = random_state

        # models
        self.gp = None
        self.rf = None
        self.mlp = None
        self.weights = None  # stacking weights [w_gp, w_rf, w_mlp] sum to 1

    def fit(self, X, y):
        """
        X: 2D numpy array
        y: 1D numpy array
        """
        # Fit GP
        try:
            self.gp = GaussianProcessRegressor(kernel=self.gp_kernel, **self.gp_opts)
            self.gp.fit(X, y)
        except Exception as e:
            # fallback: simpler kernel
            self.gp = None
            # print("GP fit failed:", e)

        # Fit RF
        try:
            self.rf = RandomForestRegressor(**self.rf_params)
            self.rf.fit(X, y)
        except Exception:
            self.rf = None

        # Fit MLP
        try:
            self.mlp = MLPRegressor(**self.mlp_params)
            self.mlp.fit(X, y)
        except Exception:
            self.mlp = None

        # Determine stacking weights based on OOB/cv-ish performance approximated by train residuals
        preds = []
        if self.gp is not None:
            gp_pred = self.gp.predict(X)
            preds.append(gp_pred)
        else:
            preds.append(np.zeros_like(y))
        if self.rf is not None:
            rf_pred = self.rf.predict(X)
            preds.append(rf_pred)
        else:
            preds.append(np.zeros_like(y))
        if self.mlp is not None:
            mlp_pred = self.mlp.predict(X)
            preds.append(mlp_pred)
        else:
            preds.append(np.zeros_like(y))
        preds = np.vstack(preds)  # (3, n)
        # compute inverse MSE weights
        mse = np.mean((preds - y.reshape(1,-1))**2, axis=1) + 1e-12
        inv = 1.0 / mse
        w = inv / np.sum(inv)
        self.weights = w

    def predict(self, X, return_std=True):
        """
        Returns: mu (n,), sigma (n,)
        Combines GP mean + RF + MLP by weights. For sigma, uses GP std if available; else estimate residual std across ensemble.
        """
        n = X.shape[0]
        preds = []
        if self.gp is not None:
            mu_gp, sigma_gp = self.gp.predict(X, return_std=True)
            preds.append(mu_gp)
        else:
            mu_gp = np.zeros(n); sigma_gp = np.ones(n) * 1e6
            preds.append(mu_gp)
        if self.rf is not None:
            preds.append(self.rf.predict(X))
        else:
            preds.append(np.zeros(n))
        if self.mlp is not None:
            preds.append(self.mlp.predict(X))
        else:
            preds.append(np.zeros(n))

        preds = np.vstack(preds)  # (3,n)
        if self.weights is None:
            w = np.ones(preds.shape[0]) / preds.shape[0]
        else:
            w = self.weights
        mu = np.dot(w, preds)
        # sigma: combine gp std with residual estimate
        if return_std:
            # residuals between ensemble members => approx uncertainty
            residuals = preds - mu
            var_est = np.mean(residuals**2, axis=0)
            gp_var = sigma_gp**2
            # weight gp var by w[0] (assuming gp is first)
            sigma = np.sqrt(w[0]*gp_var + (1-w[0])*var_est + 1e-12)
            return mu, sigma
        else:
            return mu, None

# ---------------------------
# Acquisition functions (EI, UCB) and batch (greedy q-EI via Kriging Believer)
# ---------------------------
def expected_improvement(mu, sigma, y_best, xi=0.01):
    sigma = np.maximum(sigma, 1e-12)
    z = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma==0.0] = 0.0
    return ei

def ucb(mu, sigma, kappa=2.576):
    return mu + kappa * sigma

def kriging_believer_batch(model, candidates, batch_size, acq='ei', y_best=None, xi=0.01, kappa=2.576):
    """
    Greedy q-EI approximation: select one candidate at a time, 'fantasize' its value = mu (believer) and update model's
    predicted mean (fast heuristic). Since our SurrogateEnsemble doesn't support fast update, we'll approximate by:
      - For candidate i: compute acquisition (ei or ucb) using current mu/sigma
      - choose top candidate, optionally reduce its sigma to a small value to simulate belief and repeat
    This is a cheap approximation that often works in practice.
    """
    cand = np.array(candidates)
    mu, sigma = model.predict(cand, return_std=True)
    chosen = []
    mu_curr = mu.copy()
    sigma_curr = sigma.copy()
    idxs = list(range(len(cand)))
    for _ in range(batch_size):
        if acq == 'ei':
            scores = expected_improvement(mu_curr, sigma_curr, y_best, xi=xi)
        else:
            scores = ucb(mu_curr, sigma_curr, kappa=kappa)
        j = int(np.argmax(scores))
        chosen.append(cand[j])
        # 'believe' the candidate has value mu_curr[j], reduce its variance to near zero and slightly increase mu of nearby by correlation heuristic
        # Simplified: set sigma_curr[j] to tiny, subtract candidate from choices
        sigma_curr[j] = 1e-6
        mu_curr[j] = mu_curr[j]  # unchanged
        # mask out chosen to avoid repeating
        mu_curr[j] = -1e12  # ensure not chosen again
    return chosen

# ---------------------------
# Main EvoFusionUltimate class
# ---------------------------
class EvoFusionUltimate:
    def __init__(self,
                 space,                      # Space instance
                 obj_func,                   # user-provided function: accepts decoded var list and fidelity (optional)
                 bounds=None,                # legacy: if space is None, allow simple continuous bounds list [[lo,hi],...]
                 pop_size=30,
                 generations=200,
                 init_samples=None,
                 ensemble_train_max=1000,
                 acq_strategy='ei',
                 batch_size=5,
                 gp_retrain_every=2,
                 use_cma=True,
                 seed=None,
                 verbose=True,
                 early_stopping_patience=None,
                 fidelity_levels=None,       # list of fidelities if using multi-fidelity
                 constraint_func=None        # optional: function(x_decoded) -> (feasible:bool, penalty:float)
                 ):
        if space is None and bounds is not None:
            # build a simple continuous-only space
            variables = []
            for i,b in enumerate(bounds):
                variables.append({'name':f'x{i}','type':'continuous','bounds':b})
            space = Space(variables)
        self.space = space
        self.dim = self.space.dim_cont() + self.space.dim_ohe()
        self.obj_func = obj_func
        self.pop_size = pop_size
        self.generations = generations
        self.init_samples = init_samples or max(pop_size*3, 30)
        self.ensemble_train_max = ensemble_train_max
        self.acq_strategy = acq_strategy
        self.batch_size = batch_size
        self.gp_retrain_every = gp_retrain_every
        self.use_cma = use_cma
        self.seed = seed
        if seed is not None:
            np.random.seed(seed); random.seed(seed)
        self.verbose = verbose
        self.early_stopping_patience = early_stopping_patience
        self.history = []  # list of dicts: {'x_encoded':..., 'x_decoded':..., 'y':..., 'fidelity':...}
        self.ensemble = None
        self.cma = None
        self.fidelity_levels = fidelity_levels
        self.constraint_func = constraint_func

        # default kernels
        self.gp_kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(self.dim), nu=2.5) + WhiteKernel(1e-6)
        # bounds for continuous dims for CMA
        cont_bounds = []
        for idx in self.space.cont_indices:
            cont_bounds.append(self.space.vars[idx]['bounds'])
        if self.use_cma:
            if len(cont_bounds)>0:
                self.cma = SimpleCMA(dim=len(cont_bounds), bounds=cont_bounds, seed=seed)
            else:
                self.cma = None

    # -------------------------
    # History & evaluation
    # -------------------------
    def _record(self, x_enc, y, fidelity=0):
        x_dec = self.space.decode(x_enc)
        self.history.append({'x_encoded': np.array(x_enc), 'x_decoded': x_dec, 'y': float(y), 'fidelity': fidelity})

    def _evaluate(self, x_enc, fidelity=0):
        """Decode and call user obj_func. obj_func can accept (decoded,) or (decoded, fidelity)"""
        x_dec = self.space.decode(x_enc)
        if self.constraint_func is not None:
            feasible, penalty = self.constraint_func(x_dec)
            if not feasible:
                # large negative reward or penalize
                # We'll return a heavily penalized value; user objective is assumed higher is better
                y = -1e12 + penalty
            else:
                try:
                    y = self.obj_func(x_dec, fidelity) if fidelity is not None else self.obj_func(x_dec)
                except TypeError:
                    y = self.obj_func(x_dec)
        else:
            try:
                y = self.obj_func(x_dec, fidelity) if fidelity is not None else self.obj_func(x_dec)
            except TypeError:
                y = self.obj_func(x_dec)
        return y

    # -------------------------
    # Surrogate building
    # -------------------------
    def _build_ensemble(self, fidelity=None):
        """
        Build a SurrogateEnsemble using available history; if fidelity is provided, prefer that fidelity data
        but include lower fidelities with weighting (simple approach).
        """
        # Collect X,y from history
        X = np.array([h['x_encoded'] for h in self.history])
        y = np.array([h['y'] for h in self.history])
        if X.shape[0] == 0:
            return None
        # If multi-fidelity and fidelity requested, build data selection/weighting
        if fidelity is not None and self.fidelity_levels is not None:
            # prefer same fidelity, then lower fidelities weighted less
            mask = np.array([h['fidelity']<=fidelity for h in self.history])
            Xsel = X[mask]
            ysel = y[mask]
            # optionally weight high-fidelity more: we won't implement weighted training in sklearn; just use selected subset
            X_train, y_train = Xsel, ysel
        else:
            X_train, y_train = X, y

        # cap training size
        n = X_train.shape[0]
        if n > self.ensemble_train_max:
            # keep top 20% by y and random rest
            idx_sort = np.argsort(y_train)
            top_k = max(int(0.2*n), 10)
            top_idx = idx_sort[-top_k:]
            remaining = [i for i in range(n) if i not in top_idx]
            random.shuffle(remaining)
            chosen = np.concatenate([top_idx, remaining[:self.ensemble_train_max-top_k]])
            X_train = X_train[chosen]
            y_train = y_train[chosen]

        ens = SurrogateEnsemble(gp_kernel=self.gp_kernel, random_state=self.seed)
        ens.fit(X_train, y_train)
        self.ensemble = ens
        return ens

    # -------------------------
    # Candidate generation
    # -------------------------
    def _propose_candidates(self, pop, num_candidates=200):
        """
        Mix of:
         - LHS candidates
         - CMA-ES asked candidates
         - Mutated children from pop via crossover
         - Some random jitter around elites
        Returns encoded candidates (list of arrays)
        """
        candidates = []

        # LHS
        lhs = self.space.sample_lhs(min(num_candidates//4, num_candidates))
        for row in lhs:
            candidates.append(row)

        # CMA-ES proposals
        if self.cma is not None:
            try:
                cma_x = self.cma.ask(min(num_candidates//4, num_candidates))
                # cma_x corresponds only to continuous dims; we must merge with categorical one-hot trivial choices (use random)
                for cx in cma_x:
                    # build full encoded vector by inserting categorical one-hot randomly
                    full = []
                    cont_iter = iter(cx)
                    for i,v in enumerate(self.space.vars):
                        if v['type'] in ('continuous','integer'):
                            full.append(next(cont_iter))
                        else:
                            # random one-hot
                            cats = v['bounds']
                            pick = random.choice(cats)
                            full.extend([1.0 if pick==c else 0.0 for c in cats])
                    candidates.append(np.array(full))
            except Exception:
                pass

        # Crossover + mutation from population
        # convert pop (list of encoded arrays)
        if len(pop)>0:
            for _ in range(max(10, num_candidates//4)):
                p1 = random.choice(pop)
                p2 = random.choice(pop)
                # blend crossover on continuous parts
                cont_len = self.space.dim_cont()
                if cont_len>0:
                    alpha = np.random.uniform(0,1, size=cont_len)
                    child_cont = alpha * p1[:cont_len] + (1-alpha) * p2[:cont_len]
                else:
                    child_cont = np.array([])
                # categorical: pick from parents
                child_cat = []
                oi = cont_len
                for idx in self.space.cat_indices:
                    size = len(self.space.vars[idx]['bounds'])
                    # p1 one-hot block
                    block1 = p1[oi:oi+size]
                    block2 = p2[oi:oi+size]
                    oi += size
                    # if both have same category, keep; else pick parent's category
                    if np.argmax(block1) == np.argmax(block2):
                        child_cat.extend(block1)
                    else:
                        if random.random() < 0.5:
                            child_cat.extend(block1)
                        else:
                            child_cat.extend(block2)
                child = np.concatenate([child_cont, np.array(child_cat)]) if len(child_cat)>0 else child_cont
                # mutate: add small gaussian noise to cont parts
                if cont_len>0:
                    widths = np.array([self.space.vars[i]['bounds'][1]-self.space.vars[i]['bounds'][0] for i in self.space.cont_indices])
                    noise = np.random.normal(0, 0.05, size=cont_len) * widths
                    child[:cont_len] = np.clip(child[:cont_len] + noise, [self.space.vars[i]['bounds'][0] for i in self.space.cont_indices], [self.space.vars[i]['bounds'][1] for i in self.space.cont_indices])
                candidates.append(child)

        # random jitter around elites
        if len(self.history)>0:
            # take top 5 elites
            elites = sorted(self.history, key=lambda h: h['y'], reverse=True)[:max(1, min(5, len(self.history)))]
            for e in elites:
                base = e['x_encoded']
                for _ in range(3):
                    cand = base.copy()
                    # jitter continuous parts small
                    cont_len = self.space.dim_cont()
                    if cont_len>0:
                        widths = np.array([self.space.vars[i]['bounds'][1]-self.space.vars[i]['bounds'][0] for i in self.space.cont_indices])
                        cand[:cont_len] = np.clip(cand[:cont_len] + np.random.normal(0, 0.02, size=cont_len) * widths,
                                                 [self.space.vars[i]['bounds'][0] for i in self.space.cont_indices],
                                                 [self.space.vars[i]['bounds'][1] for i in self.space.cont_indices])
                    candidates.append(cand)

        # deduplicate near-duplicates
        uniq = []
        seen = set()
        for c in candidates:
            key = tuple(np.round(c,6))
            if key not in seen:
                uniq.append(c)
                seen.add(key)
            if len(uniq) >= num_candidates:
                break

        return [np.array(u) for u in uniq]

    # -------------------------
    # Main run loop
    # -------------------------
    def run(self):
        start_time = time()
        # initialize: sample init_samples via LHS and evaluate (lowest fidelity or fidelity 0)
        init = self.space.sample_lhs(self.init_samples)
        for x in init:
            y = self._evaluate(x, fidelity=0)
            self._record(x, y, fidelity=0)

        # initialize population: top pop_size from history
        pop = []
        sorted_hist = sorted(self.history, key=lambda h: h['y'], reverse=True)
        for h in sorted_hist[:self.pop_size]:
            pop.append(h['x_encoded'])
        # if not enough elites, sample LHS
        while len(pop) < self.pop_size:
            pop.append(self.space.sample_lhs(1)[0])

        best = None
        best_y = -np.inf
        no_improve = 0

        for gen in range(1, self.generations+1):
            # train ensemble periodically
            if gen % self.gp_retrain_every == 0 or self.ensemble is None:
                try:
                    self._build_ensemble()
                except Exception as e:
                    if self.verbose:
                        print("Ensemble fit failed:", e)

            # propose many candidates
            candidates = self._propose_candidates(pop, num_candidates=400)

            # score candidates with ensemble
            Xcand = np.array(candidates)
            if self.ensemble is not None:
                mu, sigma = self.ensemble.predict(Xcand, return_std=True)
            else:
                # fallback: random scoring
                mu = np.random.normal(size=Xcand.shape[0])
                sigma = np.ones_like(mu)*1.0

            # use acquisition to select batch_size suggestions (q-EI via kriging believer)
            y_hist_best = max(h['y'] for h in self.history)
            chosen = kriging_believer_batch(self.ensemble if self.ensemble is not None else type('A',(),{'predict':lambda *_: (mu,sigma)}),
                                            Xcand, batch_size=self.batch_size, acq=self.acq_strategy, y_best=y_hist_best)
            # Evaluate chosen (use fidelity handling if desired — here we evaluate at fidelity 0)
            for c in chosen:
                y = self._evaluate(c, fidelity=0)
                self._record(c, y, fidelity=0)

            # Fill the population for next generation:
            # Strategy: keep elites, add candidates with top predicted mu, plus mutated offspring via CMA or crossover
            # evaluate all candidates (or just top k) to produce scored list
            # For efficiency only evaluate top predicted-mu candidates
            topk_idx = np.argsort(mu)[-min(200, len(mu)):] if len(mu)>0 else []
            evaluated = []
            for idx in topk_idx:
                x = Xcand[idx]
                # check if already evaluated
                found = next((h for h in self.history if np.allclose(h['x_encoded'], x)), None)
                if found:
                    evaluated.append((x, found['y']))
                else:
                    y = self._evaluate(x, fidelity=0)
                    self._record(x, y, fidelity=0)
                    evaluated.append((x, y))
            # combine with history top and existing pop
            combined = evaluated + [(h['x_encoded'], h['y']) for h in self.history]
            combined = sorted(combined, key=lambda t: t[1], reverse=True)
            # new pop: elites + some top predicted + some offspring
            n_elite = max(1, int(0.1 * self.pop_size))
            new_pop = [combined[i][0] for i in range(n_elite)]

            # add top predicted mu survivors
            i = 0
            while len(new_pop) < int(0.6*self.pop_size) and i < len(combined):
                cand = combined[i][0]
                if not any(np.allclose(cand, p) for p in new_pop):
                    new_pop.append(cand)
                i += 1

            # fill remaining with CMA offspring or mutated crossovers
            while len(new_pop) < self.pop_size:
                if self.cma is not None and random.random() < 0.6:
                    # ask 1 from CMA and incorporate decoded full vector random cats
                    try:
                        arr = self.cma.ask(1)[0]
                        # merge into full encoded vector:
                        full = []
                        cont_iter = iter(arr)
                        for idx,v in enumerate(self.space.vars):
                            if v['type'] in ('continuous','integer'):
                                full.append(next(cont_iter))
                            else:
                                # choose category by sampling weighted by historical frequency or random
                                cats = v['bounds']
                                # sample randomly
                                pick = random.choice(cats)
                                full.extend([1.0 if pick==c else 0.0 for c in cats])
                        new_pop.append(np.array(full))
                    except Exception:
                        # fallback: mutate one of elites
                        p = random.choice(new_pop)
                        child = p.copy()
                        cont_len = self.space.dim_cont()
                        if cont_len>0:
                            widths = np.array([self.space.vars[i]['bounds'][1]-self.space.vars[i]['bounds'][0] for i in self.space.cont_indices])
                            child[:cont_len] = np.clip(child[:cont_len] + np.random.normal(0, 0.03, size=cont_len)*widths,
                                                       [self.space.vars[i]['bounds'][0] for i in self.space.cont_indices],
                                                       [self.space.vars[i]['bounds'][1] for i in self.space.cont_indices])
                        new_pop.append(child)
                else:
                    # crossover + small mutation
                    p1 = random.choice(new_pop)
                    p2 = random.choice(new_pop)
                    cont_len = self.space.dim_cont()
                    if cont_len>0:
                        alpha = np.random.uniform(0,1,size=cont_len)
                        child_cont = alpha * p1[:cont_len] + (1-alpha) * p2[:cont_len]
                    else:
                        child_cont = np.array([])
                    child_cat = []
                    oi = cont_len
                    for idx in self.space.cat_indices:
                        size = len(self.space.vars[idx]['bounds'])
                        block1 = p1[oi:oi+size]; block2 = p2[oi:oi+size]
                        oi += size
                        if np.argmax(block1)==np.argmax(block2):
                            child_cat.extend(block1)
                        else:
                            child_cat.extend(block1 if random.random()<0.5 else block2)
                    child = np.concatenate([child_cont, np.array(child_cat)]) if len(child_cat)>0 else child_cont
                    # mutate
                    if cont_len>0:
                        widths = np.array([self.space.vars[i]['bounds'][1]-self.space.vars[i]['bounds'][0] for i in self.space.cont_indices])
                        child[:cont_len] = np.clip(child[:cont_len] + np.random.normal(0, 0.05, size=cont_len)*widths,
                                                   [self.space.vars[i]['bounds'][0] for i in self.space.cont_indices],
                                                   [self.space.vars[i]['bounds'][1] for i in self.space.cont_indices])
                    new_pop.append(child)

            pop = new_pop[:self.pop_size]

            # If CMA used, update with top solutions of the evaluated set (to adapt covariance)
            if self.cma is not None:
                # gather top solutions in continuous space
                top_solutions = []
                top_fits = []
                # get top mu preds or top actual history items
                top_hist = sorted(self.history, key=lambda h: h['y'], reverse=True)[:max(5, self.cma.dim)]
                for h in top_hist:
                    enc = h['x_encoded']
                    # extract continuous components
                    cont = enc[:self.space.dim_cont()]
                    top_solutions.append(cont)
                    top_fits.append(h['y'])
                if len(top_solutions) >= 2:
                    sols = np.array(top_solutions)
                    fits = np.array(top_fits)
                    try:
                        self.cma.tell(sols, fits)
                    except Exception:
                        pass

            # track best
            gen_best = max(self.history, key=lambda h: h['y'])
            if gen_best['y'] > best_y + 1e-12:
                best_y = gen_best['y']
                best = gen_best
                no_improve = 0
            else:
                no_improve += 1

            if self.verbose:
                elapsed = time() - start_time
                print(f"Gen {gen:03d} | Evaluations {len(self.history):4d} | GenBest {gen_best['y']:.6f} | Best {best_y:.6f} | Time {elapsed:.1f}s")

            # early stopping
            if self.early_stopping_patience is not None and no_improve >= self.early_stopping_patience:
                if self.verbose:
                    print(f"Early stopping after {gen} generations without improvement.")
                break

        return best, best_y

# ---------------------------
# Example usage (Ackley)
# ---------------------------
if __name__ == "__main__":
    # Ackley objective in original continuous 2D
    def ackley_obj(x, fidelity=0):
        # x is decoded list of values (floats)
        x0, x1 = x[0], x[1]
        val = -20 * np.exp(-0.2*np.sqrt(0.5*(x0**2 + x1**2))) \
              - np.exp(0.5*(np.cos(2*np.pi*x0)+np.cos(2*np.pi*x1))) \
              + np.e + 20
        # user wants to maximize -ackley typically; here we'll return negative ackley to maximize
        return -val

    # build space
    variables = [
        {'name':'x','type':'continuous','bounds':(-5,5)},
        {'name':'y','type':'continuous','bounds':(-5,5)}
    ]
    space = Space(variables)
    ef = EvoFusionUltimate(space=space,
                           obj_func=ackley_obj,
                           pop_size=24,
                           generations=60,
                           init_samples=80,
                           ensemble_train_max=300,
                           acq_strategy='ei',
                           batch_size=6,
                           gp_retrain_every=2,
                           use_cma=True,
                           seed=42,
                           verbose=True,
                           early_stopping_patience=12)

    best, best_y = ef.run()
    print("BEST:", best_y, best)
    print("Decoded best:", best['x_decoded'])
