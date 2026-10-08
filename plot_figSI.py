import numpy as np
import matplotlib.pyplot as plt


def diff_welfare_independent_strategies(t_vals, ch2, b, m, ca):
    return (-2*(ca-ch2)**2 + (2*(ca-ch2)**2 + (b-ca)*m - m**2)*t_vals)/m


def diff_welfare_same_strategies_analytical(t_vals, ch1, ch2, b, m, ca):
    welfare_vals = np.zeros_like(t_vals)
    tl = (ca-ch2+m/2)/(ch1-ch2)
    tc = (b/2-ch2)/(ch1-ch2)
    for i, t in enumerate(t_vals):
        if t <= tl:
            welfare_vals[i] = -(2*(ca+ch2*(-1+t)-ch1*t)**2)/m
        elif tl < t < tc:
            welfare_vals[i] = -ca+ch2-m+ch1*t-ch2*t
        else:
            welfare_vals[i] = b-ca-m
    return welfare_vals


plt.rc('font', size=18)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
})
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)

# Parameters
b1 = 3.#0928427435671653
ch1 = 1.8#269128158915737
ca1 = 0.7#259555079998176
m1 = 1.5#1.492879247708983
b2 = 3.#0928427435671653
ch2 = 1.#0.9916203125654305
ca2 = 0.7#259555079998176
m2 = 1.5#1.492879247708983

# Colours
C_NOAI = '#1f77b4'   # welfare without AI  (blue)
C_AI   = '#2ca02c'   # welfare with AI     (green)
C_DIFF = '#ff7f0e'   # welfare difference  (orange)

# Analytical-curve legend labels.
# NOTE: label by content, not by a hard-coded equation number, so the legend
# never goes stale when the SI is renumbered. If you prefer explicit numbers,
# replace the two strings below with the *current* \eqref targets:
#   LABEL_INDEP  <-  Eq. for eq:deltaS-zero-switching-cost
#   LABEL_SAME   <-  Eq. for eq:deltaS-infinite-switching-costs
LABEL_INDEP = r'analytical, $c=0$'
LABEL_SAME  = r'analytical, $c\to\infty$'

# --- Create figure with 4 subplots next to each other ---
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)

xmin, xmax = 0, 1
ymin, ymax = -.8, 2.4

panels = [
    ('switching_cost0.02.npz', '$c = 0.02$', 'indep'),
    ('switching_cost0.05.npz', '$c = 0.05$', 'indep'),
    ('switching_cost0.15.npz', '$c = 0.15$', 'same'),
    ('switching_cost0.2.npz',  '$c = 0.2$',  'same'),
]

for k, (fname, title, which) in enumerate(panels):
    ax = axes[k]
    data = np.load(fname)
    T_vals = data['T_vals']
    w_noAI = data['final_welfare_withoutAI']
    w_AI = data['final_welfare_withAI']

    # analytical dashed reference
    if which == 'indep':
        ax.plot(T_vals, diff_welfare_independent_strategies(T_vals, ch2, b1, m1, ca1),
                '--', lw=3, c='k', label=LABEL_INDEP)
    else:
        ax.plot(T_vals, diff_welfare_same_strategies_analytical(T_vals, ch1, ch2, b1, m1, ca1),
                '--', lw=3, c='k', label=LABEL_SAME)

    # numerical welfare curves (labelled in every panel so each legend is complete)
    ax.plot(T_vals, w_noAI, lw=2, c=C_NOAI, label='welfare without AI')
    ax.plot(T_vals, w_AI,   lw=2, c=C_AI,   label='welfare with AI')
    ax.plot(T_vals, w_AI - w_noAI, lw=2, c=C_DIFF, label='welfare difference')

    ax.fill_between(T_vals, 0, -0.8, color="grey", alpha=0.2)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    ax.legend(fontsize=9)

# --- Shared labels ---
fig.text(0.5, 0.04, '$t$', ha='center', fontsize=20)
fig.text(0.04, 0.5, 'Welfare', va='center', rotation='vertical', fontsize=20)

plt.tight_layout(rect=[0.05, 0.05, 1, 1])
plt.savefig("finite-switching-costs.pdf", bbox_inches='tight', pad_inches=0)
plt.show()