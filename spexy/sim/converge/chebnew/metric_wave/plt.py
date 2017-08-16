from collections import namedtuple

Figs = namedtuple('Figs', ['fig0', 'fig1'])


def plot(sim):
    const = next(sim)
    size = const.size
    var = next(sim)

    def fig(L, Ld):
        def fig0(ax):
            ax.loglog(size, L, '-s', color='k', label='primal')
            ax.loglog(size, Ld, '-o', color='r', label='dual')
            ax.grid(True)
            ax.legend()

            ax.set_xlabel(r'$N$')
            ax.set_ylabel(r'$\Vert \Delta f - q \Vert_\infty$')
            ax.set_xticks(
                (5, 10, 20, 40, 80)
            )
            ax.set_xticklabels(
                (5, 10, 20, 40, 80)
            )
            ax.set_yticks(
                (1e-0, 1e-2, 1e-4, 1e-6, 1e-8)
            )
            ax.set_yticklabels(
                ('1e-0', '1e-2', '1e-4', '1e-6', '1e-8')
            )

        return fig0

    return Figs(fig0=fig(var.L0, var.L0d), fig1=fig(var.L1, var.L1d))
