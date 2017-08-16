from collections import namedtuple

Figs = namedtuple('Figs', ['fig'])


def plot(sim):
    const = next(sim)
    size = const.size
    var = next(sim)
    L = var.L

    def fix_axis_labels(ax):
        ax.set_xticks(
            (4, 8, 16, 32)
        )
        ax.set_xticklabels(
            ('4', '8', '16', '32')
        )
        ax.set_yticks(
            (1e-0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
        )
        ax.set_yticklabels(
            ('1e-0', '1e-2', '1e-4', '1e-6', '1e-8', '1e-10', '1e-12')
        )

    def fig(ax):
        ax.loglog(size, L[0], '-o', color='r', label='Primal')
        ax.loglog(size, L[1], '-s', color='b', label='Dual')

        ax.grid(True)
        ax.legend()

        # plt.title(r'Periodic')
        ax.set_xlabel(r'$N$')
        ax.set_ylabel(r'$\Vert f - \Delta^{-1} q \Vert_\infty$')
        fix_axis_labels(ax)

    return Figs(fig=fig)
