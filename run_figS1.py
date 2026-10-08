"""
Welfare at the end of the replicator dynamics, with and without AI, as a
function of the task share T, for one switching cost c.

Numerical scheme (as reported in the caption of Fig. 3 and the SI)
------------------------------------------------------------------
* Integrator   : scipy.integrate.solve_ivp, LSODA (switches automatically
                 between Adams and BDF), rtol = RTOL, atol = ATOL.
* Stopping rule: integrate until the state has converged to a stable rest
                 point, i.e. max_i |dx_i/dt| < CONV_TOL and no strategy has a
                 growth rate f_i - f_avg above GROWTH_TOL; at most up to
                 t = T_END_MAX. Runs that reach T_END_MAX are reported as not
                 converged. (The growth-rate condition stops the integration
                 from ending while the population passes slowly near an
                 unstable vertex.)
* Initial cond.: the reported curve starts from the fixed interior points
                 X0_WITHOUT_AI and X0_WITH_AI.
* Continuation : none. Every value of T is integrated independently from the
                 same initial condition; nothing is carried over from
                 neighbouring T, so there is no sweep direction.
* Basin check  : at every BASIN_EVERY-th value of T, N_IC further initial
                 conditions are drawn uniformly from the simplex interior
                 (Dirichlet(1,...,1), numpy default_rng([SEED, index of T]))
                 and integrated the same way. The largest welfare difference
                 between them and the reported run is stored and summarised.

The values of T are computed in parallel; the results do not depend on the
number of workers or the order in which they finish.

Output (unchanged, read by the plotting scripts)
------------------------------------------------
switching_cost{c}.npz with keys T_vals, final_welfare_withoutAI,
final_welfare_withAI. Convergence and basin-check diagnostics go to a
separate file, switching_cost{c}_diagnostics.npz.

Strategies
----------
A strategy is a pair (mode on task 1, mode on task 2) with mode
H = do it yourself, A = use AI, N = don't do it. The order matches x1, x2, ...
    with AI   : HH HN HA NN NH NA AA AH AN
    without AI: HH HN NH NN
Against an opponent, the payoff on one task is
    -cost(own mode) + (q(own mode) + q(opponent mode)) / 2,
with cost(H) = ch, cost(A) = ca, cost(N) = 0 and q(H) = b, q(A) = b - m*s,
q(N) = 0, where s is the population share using AI on that task. The total
payoff is T * (task 1) + (1 - T) * (task 2), minus the switching cost c for a
strategy whose two modes differ.
"""
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from scipy.integrate import solve_ivp

# ----------------------------------------------------------- model parameters
b1 = 3.0
ch1 = 1.8
ca1 = 0.7
m1 = 1.5
b2 = 3.0
ch2 = 1
ca2 = 0.7
m2 = 1.5

SWITCHING_COST = 0.2
T_VALS = np.linspace(0, 1, 501)

# -------------------------------------------------------- numerical settings
METHOD = "LSODA"
RTOL = 1e-8
ATOL = 1e-10
T_END_MAX = 1e6
CONV_TOL = 1e-8          # max |dx_i/dt| at the stopping point
GROWTH_TOL = 1e-8        # max growth rate f_i - f_avg at the stopping point

X0_WITHOUT_AI = np.array([0.25, 0.25, 0.25, 0.25])
X0_WITH_AI = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2])

RUN_BASIN_CHECK = True
BASIN_EVERY = 1          # basin check at every BASIN_EVERY-th value of T
SEED = 20261008
N_IC = 10
BASIN_TOL = 1e-4

N_WORKERS = None         # None = all CPU cores

# ------------------------------------------------------------------ strategies
STRATEGIES = {
    "withAI": ("HH", "HN", "HA", "NN", "NH", "NA", "AA", "AH", "AN"),
    "withoutAI": ("HH", "HN", "NH", "NN"),
}
MODELS = ("withoutAI", "withAI")
_H, _A, _N = 0, 1, 2
_MODE = {"H": _H, "A": _A, "N": _N}
_MODES = {
    model: (np.array([_MODE[s[0]] for s in strats]),
            np.array([_MODE[s[1]] for s in strats]))
    for model, strats in STRATEGIES.items()
}
X0 = {"withoutAI": X0_WITHOUT_AI, "withAI": X0_WITH_AI}


# ----------------------------------------------------------------- payoffs
def _task_payoff(modes, b, ch, ca, m, ai_share):
    """Payoff matrix for one task, given each strategy's mode on that task."""
    q = np.array([b, b - m * ai_share, 0.0])[modes]
    cost = np.array([ch, ca, 0.0])[modes]
    return -cost[:, None] + (q[:, None] + q[None, :]) / 2


def payoff_matrix(model, T, x, c):
    mode1, mode2 = _MODES[model]
    s1 = x[mode1 == _A].sum()            # x7 + x8 + x9 with AI, 0 without
    s2 = x[mode2 == _A].sum()            # x3 + x6 + x7 with AI, 0 without
    switching = (mode1 != mode2).astype(float)
    return (T * _task_payoff(mode1, b1, ch1, ca1, m1, s1)
            + (1 - T) * _task_payoff(mode2, b2, ch2, ca2, m2, s2)
            - c * switching[:, None])


def payoff_matrix_withAI(T, x_vec, c=SWITCHING_COST):
    return payoff_matrix("withAI", T, np.asarray(x_vec, dtype=float), c)


def payoff_matrix_withoutAI(T, c=SWITCHING_COST):
    return payoff_matrix("withoutAI", T, np.zeros(4), c)


def welfare(P, x):
    return x @ P @ x


# ---------------------------------------------------------------- dynamics
def growth_rates(x, model, T, c):
    f = payoff_matrix(model, T, x, c) @ x
    return f - x @ f


def replicator_rhs(t, x, model, T, c):
    return x * growth_rates(x, model, T, c)


def _converged(t, x, model, T, c):
    """Crosses zero when both stopping conditions hold."""
    g = growth_rates(x, model, T, c)
    return max(np.max(np.abs(x * g)) - CONV_TOL, np.max(g) - GROWTH_TOL)


_converged.terminal = True
_converged.direction = -1


def integrate(model, T, c, x0):
    """Integrate to convergence (or T_END_MAX).

    Returns the end state, its welfare, max|dx/dt| there, max growth rate
    there, and the stopping time.
    """
    sol = solve_ivp(replicator_rhs, (0.0, T_END_MAX), x0, args=(model, T, c),
                    method=METHOD, rtol=RTOL, atol=ATOL, events=_converged)
    if sol.status == -1:
        raise RuntimeError(f"{model}, T={T}: {sol.message}")
    x_end = sol.y[:, -1]
    g = growth_rates(x_end, model, T, c)
    return (x_end, welfare(payoff_matrix(model, T, x_end, c), x_end),
            np.max(np.abs(x_end * g)), np.max(g), sol.t[-1])


def _is_converged(resid, growth):
    # small slack: the event stops at the crossing, up to solver tolerance
    return resid < 2 * CONV_TOL and growth < 2 * GROWTH_TOL


# ------------------------------------------------- analytical reference curves
def diff_welfare_independent_strategies(t_vals, ch2, b, m, ca, t):
    return (-2 * (ca - ch2) ** 2 + (2 * (ca - ch2) ** 2 + (b - ca) * m - m ** 2) * t_vals) / m


def diff_welfare_same_strategies_analytical(t_vals, ch1, ch2, b, m, ca, t):
    welfare_vals = np.zeros_like(t_vals)
    tl = (ca - ch2 + m / 2) / (ch1 - ch2)
    tc = (b / 2 - ch2) / (ch1 - ch2)
    for i, tv in enumerate(t_vals):
        if tv <= tl:
            welfare_vals[i] = -(2 * (ca + ch2 * (-1 + tv) - ch1 * tv) ** 2) / m
        elif tv < tc:
            welfare_vals[i] = -ca + ch2 - m + ch1 * tv - ch2 * tv
        else:
            welfare_vals[i] = b - ca - m
    return welfare_vals


# -------------------------------------------------------------------- main
def _solve_one_T(i, c):
    """Reported run and basin check for both models at T_VALS[i]."""
    T = T_VALS[i]
    do_basin = RUN_BASIN_CHECK and i % BASIN_EVERY == 0
    rng = np.random.default_rng([SEED, i])
    out = {}
    for model in MODELS:
        _, w, r, g, t_stop = integrate(model, T, c, X0[model])
        res = dict(welfare=w, resid=r, growth=g, t_stop=t_stop,
                   converged=_is_converged(r, g),
                   basin_spread=np.nan, basin_converged=True)
        if do_basin:
            starts = rng.dirichlet(np.ones(len(STRATEGIES[model])), size=N_IC)
            runs = [integrate(model, T, c, s) for s in starts]
            res["basin_spread"] = max(abs(run[1] - w) for run in runs)
            res["basin_converged"] = all(_is_converged(run[2], run[3]) for run in runs)
        out[model] = res
    return out


def run(c=SWITCHING_COST):
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        per_T = list(ex.map(partial(_solve_one_T, c=c), range(len(T_VALS)),
                            chunksize=4))

    def col(model, key):
        return np.array([r[model][key] for r in per_T])

    final_welfare = {m: list(col(m, "welfare")) for m in MODELS}

    np.savez('switching_cost{}.npz'.format(c), T_vals=T_VALS,
             final_welfare_withoutAI=final_welfare["withoutAI"],
             final_welfare_withAI=final_welfare["withAI"])

    keys = ("resid", "growth", "t_stop", "converged", "basin_spread", "basin_converged")
    np.savez('switching_cost{}_diagnostics.npz'.format(c), T_vals=T_VALS,
             switching_cost=c, method=METHOD, rtol=RTOL, atol=ATOL,
             t_end_max=T_END_MAX, conv_tol=CONV_TOL, growth_tol=GROWTH_TOL,
             x0_withoutAI=X0_WITHOUT_AI, x0_withAI=X0_WITH_AI, seed=SEED,
             n_ic=N_IC if RUN_BASIN_CHECK else 0, basin_every=BASIN_EVERY,
             **{f"{k}_{m}": col(m, k) for m in MODELS for k in keys})

    print(f"Switching cost {c}: {len(T_VALS)} values of T in {time.time() - t0:.0f}s; "
          f"{METHOD} rtol={RTOL} atol={ATOL}; stop at max|dx/dt|<{CONV_TOL:g} "
          f"and growth<{GROWTH_TOL:g}, t_end_max={T_END_MAX:g}")
    for model in MODELS:
        conv = col(model, "converged")
        t_stop = col(model, "t_stop")
        print(f"  {model:9s} reported run: {np.sum(~conv)} values of T not converged"
              + (f" (T = {np.round(T_VALS[~conv], 3).tolist()})" if np.any(~conv) else "")
              + f"; stopping time median {np.median(t_stop):.0f}, max {t_stop.max():.0f}")
        if RUN_BASIN_CHECK:
            s = col(model, "basin_spread")
            checked = ~np.isnan(s)
            bad = checked & (s >= BASIN_TOL)
            bc = col(model, "basin_converged")
            print(f"  {model:9s} basin check ({N_IC} random starts at {checked.sum()} values of T, "
                  f"seed {SEED}): max welfare difference {np.nanmax(s):.1e}"
                  f"; {bad.sum()} values of T above {BASIN_TOL:g}"
                  + (f" (T = {np.round(T_VALS[bad], 3).tolist()})" if np.any(bad) else "")
                  + f"; {np.sum(checked & ~bc)} with a non-converged start")

    return final_welfare


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    final_welfare = run(SWITCHING_COST)
    final_welfare_withoutAI = final_welfare["withoutAI"]
    final_welfare_withAI = final_welfare["withAI"]

    # --- Optional: Plot equilibrium value vs T ---
    plt.figure(figsize=(6, 4))
    plt.hlines(0, 0, 1, linestyles='--', colors='k')
    plt.plot(T_VALS, final_welfare_withoutAI, lw=2, label='welfare no-AI')
    plt.plot(T_VALS, final_welfare_withAI, lw=2, label='welfare AI')
    plt.plot(T_VALS, np.array(final_welfare_withAI) - np.array(final_welfare_withoutAI),
             lw=2, label='welfare difference')
    plt.xlabel('t')
    plt.ylabel('Welfare')
    plt.title('Switching cost = {}'.format(SWITCHING_COST))
    plt.legend()
    plt.show()
