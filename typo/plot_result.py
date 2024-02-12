import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'sans-serif'

df = pd.read_excel("typo_experiment/results/typo_test_results.xlsx", sheet_name=None)
df.pop("PAWS-X")
# df.pop("NSMC")

color_settings = {'Jamo': 'mediumseagreen',
                  'Character': 'mediumpurple',
                  'Subword': 'goldenrod',
                  'MorSubword': 'cornflowerblue',
                  'KOMBO(Jamo)': 'palevioletred'
                  }
marker_settings = {'Jamo': '<',
                   'Character': '^',
                   'Subword': '>',
                   'MorSubword': 'D',
                   'KOMBO(Jamo)': 'o'
                   }

plt.style.use('seaborn')
# plt.style.use('seaborn-v0_8-whitegrid')

fig, axes = plt.subplots(1, len(df), figsize=(len(df)*6+2, 5))
fig.subplots_adjust(wspace=0.3)
for i, task_name in enumerate(df):
    cur_df = df[task_name]
    x = [str(int(n * 100)) for n in cur_df["noise"]]
    x_val = np.arange(len(x))
    ax = axes[i]
    ax.text(.5, .9, task_name,
            horizontalalignment='center',
            transform=ax.transAxes,
            fontsize=25,
            fontweight='bold')
    ax.set_xticks(x_val)
    ax.set_xticklabels(x, fontsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlabel("Typo Ratio (%)", fontsize=20, labelpad=10)
    if i == 0:
        ax.set_ylabel("Accuracy (%)", fontsize=20, labelpad=15)
    else:
        ax.set_ylabel(None)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    if task_name == 'PAWS-X':
        sota_baseline = "Subword"
    else:
        sota_baseline = "MorSubword"

    for tok_type in cur_df:
        if tok_type == "noise":
            continue
        else:
            y = [acc for acc in cur_df[tok_type]]
            if tok_type == "KOMBO(Jamo)":
                ax.plot(y, label="KOMBO$_{Jamo}$", linewidth=2, marker=marker_settings[tok_type], markersize=10, color=color_settings[tok_type])
            else:
                ax.plot(y, label=tok_type, linewidth=2, marker=marker_settings[tok_type], markersize=10, color=color_settings[tok_type])
    ax.legend(fontsize=10)
    baseline_y = [acc for acc in cur_df[sota_baseline]]
    hem_y = [acc for acc in cur_df["KOMBO(Jamo)"]]

    # fill the area between MorSubword and HALLA(Jamo), which are the baseline SOTA and our method SOTA, respectively.
    ax.fill_between(x, baseline_y, hem_y, alpha=0.15, color="grey")

    # Add the arrow to show and stress the difference between MorSubword and HALLA(Jamo)
    # for j in range(len(x)):
    #     ax.annotate("",
    #                 xy=(x_val[j], min(MorSubword_y[j], hem_y[j])),
    #                 xytext=(x_val[j], max(MorSubword_y[j], hem_y[j])),
    #                 arrowprops=dict(arrowstyle="<->", color='black', lw=1.5),
    #                 )
    ax.legend(loc="lower left", fontsize=14)

plt.tight_layout()
plt.savefig("typo_experiment/results/typo_test_results.pdf", format='pdf')
plt.show()