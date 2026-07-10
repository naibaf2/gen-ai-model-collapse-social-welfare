import numpy as np
import matplotlib.pyplot as plt


plt.rc('font', size=18)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
})


def welfare_same_strategies_with_AI(ch1, ch2, b, m, ca, t):

    if t < (2*ca-2*ch2+m)/(2*ch1-2*ch2):
        return (-2*ca**2+b*m+(ch2*(-1+t)-ch1*t)*(m-2*ch2*(-1+t)+2*ch1*t)+4*ca*(ch2+ch1*t-ch2*t))/m
    else:
        return b-ca-m


def welfare_same_strategies_without_AI(ch1, ch2, b, m, ca, t):

    if t < (b/2-ch2)/(ch1-ch2):
        return t*(-ch1+b)+(1-t)*(-ch2+b)
    else:
        return 0


def diff_welfare_same_strategies(ch1, ch2, b, m, ca, t):

    tl = (ca-ch2+m/2)/(ch1-ch2)
    tc = (b/2-ch2)/(ch1-ch2)

    if t <= tl:
        return -(2*(ca+ch2*(-1+t)-ch1*t)**2)/m
    if tl < t < tc:
        return -ca+ch2-m+ch1*t-ch2*t
    else:
        return b-ca-m


# parameters
ch1 = 1.8
ch2 = 1.
b = 3.
m = 1.5
ca = 0.7

t_min, t_max, n_t = 0.0, 1.0, 1001
t_vals = np.linspace(t_min, t_max, n_t)

welfare_with_AI = np.array([welfare_same_strategies_with_AI(ch1, ch2, b, m, ca, t) for t in t_vals])
welfare_without_AI = np.array([welfare_same_strategies_without_AI(ch1, ch2, b, m, ca, t) for t in t_vals])
welfare_difference = np.array([diff_welfare_same_strategies(ch1, ch2, b, m, ca, t) for t in t_vals])

fig, ax = plt.subplots(figsize=(6, 5))

plt.plot(t_vals, welfare_difference, lw=4, c='orange', label='Welfare difference ($\Delta S$)')
plt.plot(t_vals, welfare_with_AI, lw=2, c='k', label='Social welfare with AI')
plt.plot(t_vals, welfare_without_AI, "--", lw=2, c='k', label='Social welfare without AI')
plt.fill_between(t_vals, 0, np.min(welfare_difference), color="grey", alpha=0.1)

plt.xlabel('$t$ (Task fraction)', fontsize=18)
plt.ylabel('Social welfare', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(loc="center left", fontsize=12)

plt.tight_layout()
fig.savefig("example_same.pdf")

plt.show()
