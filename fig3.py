import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ── AI model (3 strategies) ──────────────────────────────────────────────────

def payoff_matrix_AI(x, y, b, ch, ca, m):
    z = 1 - x - y
    A = b - m * z
    P = np.array([
        [b - ch,         b/2 - ch,       (b + A)/2 - ch],
        [b/2,            0,              A/2            ],
        [(b + A)/2 - ca, A/2 - ca,       A - ca         ]
    ])
    return P


def Pix_AI(x, y, b, ch, ca, m):
    v = np.array([x, y, 1 - x - y])
    return payoff_matrix_AI(x, y, b, ch, ca, m)[0] @ v


def Piy_AI(x, y, b, ch, ca, m):
    v = np.array([x, y, 1 - x - y])
    return payoff_matrix_AI(x, y, b, ch, ca, m)[1] @ v


def Phi_AI(x, y, b, ch, ca, m):
    v = np.array([x, y, 1 - x - y])
    return v @ (payoff_matrix_AI(x, y, b, ch, ca, m) @ v)


def replicator_AI(t, state, b, ch, ca, m):
    x, y = state
    z = 1 - x - y
    if x < 0 or y < 0 or z < 0:
        return [0.0, 0.0]
    phi = Phi_AI(x, y, b, ch, ca, m)
    dx = x * (Pix_AI(x, y, b, ch, ca, m) - phi)
    dy = y * (Piy_AI(x, y, b, ch, ca, m) - phi)
    return [dx, dy]


# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

C_WORK     = '#56C1FF'
C_NOWORK   = '#929292'
C_AI_STRAT = '#ED6E57'
C_PHI_AI   = 'k'
C_PHI_NO   = 'k'

# Per-panel color overrides: list of dicts (one per param_set), keys optional.
# Keys: 'work', 'nowork', 'ai_strat', 'phi_ai', 'phi_no'
panel_colors = [
    dict(),   # panel 0: use defaults
    dict(),   # panel 1: use defaults
]


# ── Plotting function ─────────────────────────────────────────────────────────

def run_and_plot(b, ch, ca, m, x0, y0, phi_noAI, t_span, t_eval, ax, colors=None):
    c = colors or {}
    c_work     = c.get('work',     C_WORK)
    c_nowork   = c.get('nowork',   C_NOWORK)
    c_ai_strat = c.get('ai_strat', C_AI_STRAT)
    c_phi_ai   = c.get('phi_ai',   C_PHI_AI)
    c_phi_no   = c.get('phi_no',   C_PHI_NO)

    sol = solve_ivp(replicator_AI, t_span, [x0, y0], t_eval=t_eval,
                    args=(b, ch, ca, m), method='RK45', rtol=1e-8, atol=1e-10)

    x_sol   = sol.y[0]
    y_sol   = sol.y[1]
    z_sol   = 1 - x_sol - y_sol
    phi_sol = np.array([Phi_AI(x, y, b, ch, ca, m) for x, y in zip(x_sol, y_sol)])

    ax2 = ax.twinx()

    ax.plot(sol.t, x_sol, color=c_work,     lw=2.5, label='Human work (H)')
    ax.plot(sol.t, y_sol, color=c_nowork,   lw=2.5, label='No work (N)')
    ax.plot(sol.t, z_sol, color=c_ai_strat, lw=2.5, label='genAI work (AI)')

    ax2.plot(sol.t, phi_sol,                        color=c_phi_ai, lw=2,   ls='-',  label='Social welfare with AI')
    ax2.axhline(phi_noAI, color=c_phi_no, lw=2, ls='--', label='Social welfare without AI')

    ax.set_xlabel('Time', fontsize=13)
    ax.set_ylabel('Frequency', fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax2.set_ylabel('Social welfare', fontsize=13)

    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(c_phi_ai)
    ax2.yaxis.label.set_color(c_phi_ai)
    ax2.tick_params(axis='y', colors=c_phi_ai)

    ax.set_title(f'$b={b},\\ c_h={ch},\\ c_a={ca},\\ m={m}$', fontsize=13)

    return ax.get_legend_handles_labels(), ax2.get_legend_handles_labels()


# ── Parameter sets ────────────────────────────────────────────────────────────
# phi_noAI: expected no-AI payoff shown as horizontal dashed line
#   left panel:  Phi = 0      (defection equilibrium)
#   right panel: Phi = b - ch (cooperation equilibrium)

param_sets = [
    dict(b=3.5, ch=2, ca=1.5, m=0.4, t_span=(0, 100),
         x0=0.00001/2, y0=1-0.00001, phi_noAI=0),
    dict(b=4.5, ch=2, ca=1.5, m=1.5, t_span=(0, 70),
         x0=1-0.00001, y0=0.00001/2, phi_noAI=None),   # computed as b - ch below
]

# fill in phi_noAI = b - ch where None
for p in param_sets:
    if p['phi_noAI'] is None:
        p['phi_noAI'] = p['b'] - p['ch']

n = len(param_sets)


# ── Run ───────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, n, figsize=(7 * n, 4.))

legend_handles, legend_labels = None, None

for i, (ax, p) in enumerate(zip(axes, param_sets)):
    t_span    = p.pop('t_span')
    x0        = p.pop('x0')
    y0        = p.pop('y0')
    phi_noAI  = p.pop('phi_noAI')
    t_eval    = np.linspace(*t_span, 2000)
    (h1, l1), (h2, l2) = run_and_plot(**p, x0=x0, y0=y0, phi_noAI=phi_noAI,
                                       t_span=t_span, t_eval=t_eval, ax=ax,
                                       colors=panel_colors[i])
    if legend_handles is None:
        legend_handles = h1 + h2
        legend_labels  = l1 + l2

axes[0].legend(legend_handles, legend_labels, fontsize=10,
               loc='center left', framealpha=0.4, edgecolor='none')

fig.tight_layout(w_pad=4)
fig.savefig('fig3.pdf', bbox_inches='tight')

plt.show()
