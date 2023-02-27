#  Copyright (c) 2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import matplotlib
import numpy as np
from matplotlib import pyplot as plt


def heatmap(
    data,
    row_labels,
    col_labels,
    index,
    title,
    ax=None,
    cbar_kw=None,
    cbarlabel="",
    **kwargs,
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlabel("Agent")
    if index == 1:
        ax.set_ylabel("Agent")
    ax.xaxis.set_label_position("top")
    ax.set_title(title, y=-0.15)

    # Create colorbar
    if index == 4:
        cbar = None
        # divider = make_axes_locatable(cbar_ax)
        # cax = divider.append_axes("right", size="5%", pad=0.08)
        # cbar = plt.colorbar(
        #     im, fraction=0.047 * (data.shape[1] / data.shape[0]), **cbar_kw
        # )
        # cbar = fig.colorbar(im, ax=axs[3:], location="right", shrink=0.6)
        # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    else:
        cbar = None

    return im, cbar


def annotate_heatmap(
    im,
    std,
    data=None,
    valfmt="{x:.1f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(
                j,
                i,
                f"${valfmt(data[i, j], None)}$\n$\pm{valfmt(std[i,j],None)}$",
                **kw,
            )
            texts.append(text)

    return texts


tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "text.latex.preamble": "\\renewcommand{\\familydefault}{\\sfdefault}\n\\usepackage{helvet}",
    "font.family": "sans-serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 20,
    "font.size": 20,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 20,
    "legend.title_fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
}


plt.rcParams.update(tex_fonts)

n_goals = {
    "1": [
        np.array(
            [
                [0, 1.485, 1.411, 1.436],
                [1.485, 0.0, 1.475, 1.409],
                [1.411, 1.475, 0.0, 1.38],
                [1.436, 1.409, 1.38, 0.0],
            ]
        ),
        np.array(
            [
                [0, 0.3482, 0.2798, 0.1841],
                [0.3482, 0.0, 0.2221, 0.2835],
                [0.2798, 0.221, 0.0, 0.3087],
                [0.1841, 0.2835, 0.3087, 0.0],
            ]
        ),
    ],
    "2": [
        np.array(
            [
                [0, 0.14541, 1.347, 1.425],
                [0.1541, 0.0, 1.326, 1.391],
                [1.347, 1.326, 0.0, 1.443],
                [1.425, 1.391, 1.443, 0.0],
            ]
        ),
        np.array(
            [
                [0, 0.04734, 0.2554, 0.153],
                [0.04734, 0.0, 0.2799, 0.1056],
                [0.2554, 0.2799, 0.0, 0.2641],
                [0.153, 0.1056, 0.2641, 0.0],
            ]
        ),
    ],
    "3": [
        np.array(
            [
                [0, 0.1193, 0.1089, 1.475],
                [0.1193, 0, 0.1176, 1.468],
                [0.1089, 0.1176, 0, 1.462],
                [1.475, 1.468, 1.462, 0],
            ]
        ),
        np.array(
            [
                [0, 0.01895, 0.02506, 0.1512],
                [0.01895, 0, 0.02254, 0.1393],
                [0.02506, 0.02254, 0, 0.1512],
                [0.1512, 0.1393, 0.1512, 0],
            ]
        ),
    ],
    "4": [
        np.array(
            [
                [0, 0.08371, 0.07624, 0.07192],
                [0.08371, 0, 0.08203, 0.07158],
                [0.07624, 0.08203, 0, 0.07],
                [0.07192, 0.07158, 0.07, 0],
            ]
        ),
        np.array(
            [
                [0, 0.02042, 0.01195, 0.01528],
                [0.02042, 0, 0.01518, 0.01902],
                [0.01195, 0.01518, 0, 0.01216],
                [0.01528, 0.01902, 0.01216, 0],
            ]
        ),
    ],
}
mymax = max([m.max() for m, _ in n_goals.values()])
# fig, ax = plt.subplots(figsize=(6.5, 6.5))
fig, axs = plt.subplots(1, 4, figsize=(21, 8), constrained_layout=True)
for index in range(1, 5):
    ax = axs[index - 1]
    mean, std = n_goals[str(index)]

    im, _ = heatmap(
        mean,
        np.arange(mean.shape[0]),
        np.arange(mean.shape[1]),
        ax=ax,
        cmap="YlGn",
        cbarlabel="$d(i,j)$",
        vmax=mymax,
        index=index,
        title=f"{5-index} {'goals'if 5-index> 1 else 'goal'}",
    )
    texts = annotate_heatmap(im, std, valfmt="{x:.2f}", threshold=mymax / 2)

cbarlabel = "$d(i,j)$"
cbar = fig.colorbar(im, ax=axs[:], location="right", shrink=0.56, pad=0.01)
cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

plt.savefig(f"multi_goal.pdf", bbox_inches="tight", pad_inches=0)
plt.show()
