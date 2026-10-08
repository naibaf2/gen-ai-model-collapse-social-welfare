"""
All four models of the SI, in six rows.

Row 1  main text            a = b - m x_AI
Row 2  lagged quality       a = b - m s,   ds/dt = lam (x_AI - s),          lam = LAM_MID
Row 3  lagged quality       same,                                             lam = LAM_SLOW
Row 4  fast corpus          a = b - m r x_AI / (r x_AI + x_H + ell)
Row 5  corpus dynamics      a = b - m s,   ds/dt = lam (r x_AI - s rho),     lam = LAM_MID
Row 6  corpus dynamics      same,                                             lam = LAM_SLOW

with rho = x_H + ell + r x_AI. Rows 1 and 4 contain no lambda.

Columns 1-2  "busy-work" (type Ia): low incentive, weak collapse
Columns 3-4  "poetry"    (type IIb): high incentive, strong collapse

Each domain is shown as a pair of panels: strategy frequencies (with the
contamination s dotted where it is a state variable, and the instantaneous
corpus share dotted in row 4) on the left, social welfare on the right with
the no-AI benchmark dashed.

Trajectories start from the no-AI equilibrium (all-N for busy-work, all-H for
poetry) invaded by a small AI fraction AI0, with a much smaller fraction MINOR
of the remaining strategy (H for busy-work, N for poetry), so that no strategy
starts at exactly zero; the corpus starts clean.

Requires numpy, scipy, matplotlib.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------------------------------------------------------
# numerical settings (reported in the caption for reproducibility)
# ----------------------------------------------------------------------
AI0 = 1e-3      # initial AI fraction (not a random seed)
MINOR = 1e-6    # initial fraction of the strategy absent from the no-AI equilibrium
RTOL, ATOL = 1e-10, 1e-12
SOLVER = "LSODA"

# ----------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------
BUSY = dict(key="busy", name="busy-work (Ia)", b=3.5, c_h=2.0, c_a=1.5, m=0.4)
POETRY = dict(key="poetry", name="poetry (IIb)", b=4.5, c_h=2.0, c_a=1.5, m=1.5)

R, ELL = 1.0, 0.2
LAM_MID, LAM_SLOW = 10.0, 0.015

# horizons per (model kind, domain); slow rows need longer for the cycle
T_END = {"busy": {"inst": 400.0, "mid": 400.0, "slow": 400.0},
         "poetry": {"inst": 120.0, "mid": 120.0, "slow": 1500.0}}


# ----------------------------------------------------------------------
# payoffs and welfare
# ----------------------------------------------------------------------
def payoffs(x_H, x_N, x_AI, a, p):
    b, c_h, c_a = p["b"], p["c_h"], p["c_a"]
    pi_H = x_H * (b - c_h) + x_N * (b / 2 - c_h) + x_AI * ((b + a) / 2 - c_h)
    pi_N = x_H * (b / 2) + x_AI * (a / 2)
    pi_AI = x_H * ((b + a) / 2 - c_a) + x_N * (a / 2 - c_a) + x_AI * (a - c_a)
    return pi_H, pi_N, pi_AI


def welfare(x_H, x_AI, a, p):
    return x_H * (p["b"] - p["c_h"]) + x_AI * (a - p["c_a"])


def welfare_no_ai(p):
    return p["b"] - p["c_h"] if p["b"] > 2 * p["c_h"] else 0.0


def corpus_share(x_H, x_AI, r, ell):
    """Instantaneous corpus share sigma = r x_AI / (r x_AI + x_H + ell)."""
    return r * x_AI / (r * x_AI + x_H + ell)


# ----------------------------------------------------------------------
# the four vector fields; `quality` returns a and the plotted contamination
# ----------------------------------------------------------------------
def rhs_main(t, y, p):
    x_H, x_N = y
    x_AI = 1.0 - x_H - x_N
    a = p["b"] - p["m"] * x_AI
    pi_H, pi_N, pi_AI = payoffs(x_H, x_N, x_AI, a, p)
    phi = x_H * pi_H + x_N * pi_N + x_AI * pi_AI
    return [x_H * (pi_H - phi), x_N * (pi_N - phi)]


def rhs_lag(t, y, p, lam):
    x_H, x_N, s = y
    x_AI = 1.0 - x_H - x_N
    a = p["b"] - p["m"] * s
    pi_H, pi_N, pi_AI = payoffs(x_H, x_N, x_AI, a, p)
    phi = x_H * pi_H + x_N * pi_N + x_AI * pi_AI
    return [x_H * (pi_H - phi), x_N * (pi_N - phi), lam * (x_AI - s)]


def rhs_fastcorpus(t, y, p, r, ell):
    x_H, x_N = y
    x_AI = 1.0 - x_H - x_N
    a = p["b"] - p["m"] * corpus_share(x_H, x_AI, r, ell)
    pi_H, pi_N, pi_AI = payoffs(x_H, x_N, x_AI, a, p)
    phi = x_H * pi_H + x_N * pi_N + x_AI * pi_AI
    return [x_H * (pi_H - phi), x_N * (pi_N - phi)]


def rhs_corpus(t, y, p, r, ell, lam):
    x_H, x_N, s = y
    x_AI = 1.0 - x_H - x_N
    a = p["b"] - p["m"] * s
    pi_H, pi_N, pi_AI = payoffs(x_H, x_N, x_AI, a, p)
    phi = x_H * pi_H + x_N * pi_N + x_AI * pi_AI
    rho = x_H + ell + r * x_AI
    return [x_H * (pi_H - phi), x_N * (pi_N - phi), lam * (r * x_AI - s * rho)]


MODELS = {
    # name      : (rhs, has_s, kind for horizon, extra args builder)
    "main":       (rhs_main,       False),
    "lag":        (rhs_lag,        True),
    "fastcorpus": (rhs_fastcorpus, False),
    "corpus":     (rhs_corpus,     True),
}


# ----------------------------------------------------------------------
# integration
# ----------------------------------------------------------------------
def initial_state(p, has_s):
    if p["b"] > 2 * p["c_h"]:
        x_H, x_N = 1.0 - AI0 - MINOR, MINOR
    else:
        x_H, x_N = MINOR, 1.0 - AI0 - MINOR
    return [x_H, x_N, 0.0] if has_s else [x_H, x_N]


def run(p, model, lam=None, r=R, ell=ELL):
    rhs, has_s = MODELS[model]
    y0 = initial_state(p, has_s)
    speed = "inst" if lam is None else ("mid" if lam >= 1 else "slow")
    t_end = T_END[p["key"]][speed]
    t_eval = np.linspace(0.0, t_end, 4000)
    args = {"main": (p,), "lag": (p, lam),
            "fastcorpus": (p, r, ell), "corpus": (p, r, ell, lam)}[model]
    sol = solve_ivp(rhs, (0, t_end), y0, args=args, t_eval=t_eval,
                    method=SOLVER, rtol=RTOL, atol=ATOL)
    if not sol.success:
        raise RuntimeError(sol.message)
    x_H, x_N = sol.y[0], sol.y[1]
    x_AI = 1.0 - x_H - x_N
    if has_s:
        s = sol.y[2]
    elif model == "fastcorpus":
        s = corpus_share(x_H, x_AI, r, ell)
    else:
        s = x_AI
    a = p["b"] - p["m"] * s
    return dict(t=sol.t, x_H=x_H, x_N=x_N, x_AI=x_AI, s=s,
                W=welfare(x_H, x_AI, a, p),
                show_s=(has_s or model == "fastcorpus"))


# ----------------------------------------------------------------------
# analytic equilibria, for checking
# ----------------------------------------------------------------------
def predict(p, corpus, r=R, ell=ELL):
    b, c_h, c_a, m = p["b"], p["c_h"], p["c_a"], p["m"]
    d, beta = c_h - c_a, b / 2 - c_a
    sig = r / (r + ell) if corpus else 1.0
    gamma = min(d, beta)
    if gamma < 0:
        return dict(state="all-N", x=0.0, W=0.0)
    if gamma > m * sig / 2:
        return dict(state="all-AI", x=1.0, W=b - c_a - m * sig)
    if c_h < b / 2:
        s_st = 2 * d / m
        x = s_st * (1 + ell) / (r - s_st * (r - 1)) if corpus else s_st
        return dict(state="H-AI", x=x, W=b - c_h - d * x)
    s_st = (b - 2 * c_a) / m
    x = s_st * ell / (r * (1 - s_st)) if corpus else s_st
    return dict(state="N-AI", x=x, W=c_a * x)


def report():
    print("=" * 76)
    print(f"r={R}, ell={ELL}, sigma_bar={R/(R+ELL):.4f};  lam_mid={LAM_MID}, "
          f"lam_slow={LAM_SLOW};  AI0={AI0}, MINOR={MINOR}, {SOLVER}, rtol={RTOL}, atol={ATOL}")
    print("=" * 76)
    for p in (BUSY, POETRY):
        print(f"\n{p['name']}:  b={p['b']} c_h={p['c_h']} c_a={p['c_a']} m={p['m']}"
              f"   W0={welfare_no_ai(p):.4f}")
        for model, lam, label in ROWS:
            q = predict(p, corpus=model in ("fastcorpus", "corpus"))
            res = run(p, model, lam=lam)
            ok = abs(res["x_AI"][-1] - q["x"]) < 2e-3 and abs(res["W"][-1] - q["W"]) < 2e-3
            print(f"  {label:<34} pred {q['state']:<6} x_AI={q['x']:.4f} W={q['W']:.4f}"
                  f" | sim x_AI={res['x_AI'][-1]:.4f} s={res['s'][-1]:.4f} "
                  f"W={res['W'][-1]:.4f}  {'OK' if ok else 'MISMATCH'}")


# ----------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------
COL = dict(H="#1f77b4", N="#7f7f7f", AI="#d62728", s="#2ca02c", W="#000000")

ROWS = [
    ("main",       None,     "linear + instantaneous (main text)"),
    ("lag",        LAM_MID,  rf"linear + lagged $(\lambda={LAM_MID:g})$"),
    ("lag",        LAM_SLOW, rf"linear + lagged $(\lambda={LAM_SLOW:g})$"),
    ("fastcorpus", None,     "saturating + instantaneous"),
    ("corpus",     LAM_MID,  rf"saturating + lagged $(\lambda={LAM_MID:g})$"),
    ("corpus",     LAM_SLOW, rf"saturating + lagged $(\lambda={LAM_SLOW:g})$"),
]


def make_figure(path="figS2.pdf"):
    fig, axes = plt.subplots(len(ROWS), 4, figsize=(15, 3.0 * len(ROWS)))

    for i, (model, lam, label) in enumerate(ROWS):
        for j, p in enumerate((BUSY, POETRY)):
            res = run(p, model, lam=lam)
            ax_s, ax_w = axes[i, 2 * j], axes[i, 2 * j + 1]

            ax_s.plot(res["t"], res["x_H"], color=COL["H"], lw=2)
            ax_s.plot(res["t"], res["x_N"], color=COL["N"], lw=2)
            ax_s.plot(res["t"], res["x_AI"], color=COL["AI"], lw=2)
            if res["show_s"]:
                ax_s.plot(res["t"], res["s"], color=COL["s"], lw=1.6, ls=":")
            ax_s.set_ylim(-0.04, 1.04)
            ax_s.set_ylabel("frequency")

            W0 = welfare_no_ai(p)
            ax_w.plot(res["t"], res["W"], color=COL["W"], lw=2)
            ax_w.axhline(W0, color=COL["W"], ls="--", lw=1.4)
            lo, hi = min(res["W"].min(), W0), max(res["W"].max(), W0)
            pad = 0.12 * max(hi - lo, 1e-6)
            ax_w.set_ylim(lo - pad, hi + pad)
            ax_w.set_ylabel("social welfare")

            for ax in (ax_s, ax_w):
                ax.set_xlim(0, res["t"][-1])
                ax.spines[["top", "right"]].set_visible(False)
                if i == len(ROWS) - 1:
                    ax.set_xlabel("time")
            if i == 0:
                ax_s.set_title(p["name"] + "\nstrategies", fontsize=10)
                ax_w.set_title(p["name"] + "\nwelfare", fontsize=10)

        axes[i, 0].text(-0.32, 0.5, label, transform=axes[i, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=10)

    handles = [Line2D([], [], color=COL[k], lw=2, label=lab)
               for k, lab in (("H", "human work (H)"), ("N", "no work (N)"),
                              ("AI", "AI work (AI)"))]
    handles += [Line2D([], [], color=COL["s"], lw=1.6, ls=":",
                       label="corpus contamination ($s$)"),
                Line2D([], [], color=COL["W"], lw=1.4, ls="--",
                       label="welfare without AI")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.003))

    fig.tight_layout(rect=(0.02, 0.03, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    print(f"\nwrote {path}")
    return fig


if __name__ == "__main__":
    report()
    make_figure()
