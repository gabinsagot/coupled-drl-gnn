from cycler import cycler

scientific_style = {
    # Figure
    "figure.figsize": (8, 6),  # Adjust the figure size
    # Font
    "font.family": "serif",  # Use a serif font
    "font.serif": ["cmr10", "Computer Modern Serif", "DejaVu Serif"],
    "font.size": 12,  # Set the font size
    # Axes
    "axes.labelsize": 12,  # Label font size
    "axes.titlesize": 14,  # Title font size
    "axes.prop_cycle": cycler(
        "color",
        [
            "darkblue",
            "darkgreen",
            "darkred",
            "darkorange",
            "darkorchid",
            "darksalmon",
            "darkmagenta",
            "darkcyan",
            "darkgray",
            "saddlebrown",
        ],
    ),
    "axes.grid": True,  # Show grid lines
    "axes.formatter.use_mathtext": True,
    # Set x axis
    "xtick.direction": "in",
    "xtick.major.size": 3,
    "xtick.major.width": 0.5,
    "xtick.minor.size": 1.5,
    "xtick.minor.width": 0.5,
    "xtick.minor.visible": True,
    "xtick.top": True,
    "xtick.labelsize": 8,
    # Set y axis
    "ytick.direction": "in",
    "ytick.major.size": 3,
    "ytick.major.width": 0.5,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.5,
    "ytick.minor.visible": True,
    "ytick.right": True,
    "ytick.labelsize": 8,
    # Lines
    "lines.linewidth": 1,  # Line width
    "lines.markersize": 5,  # Marker size
    # Grid
    "grid.linestyle": "--",  # Grid line style
    "grid.linewidth": 0.3,
    "grid.alpha": 0.4,
    "grid.color": "gray",
    # Legend
    "legend.fontsize": 8,  # Legend font size
    "legend.frameon": True,  # Show legend frame
    "legend.framealpha": 0.6,  # Legend frame opacity
    "legend.edgecolor": "gray",  # Legend frame color
    "legend.loc": "best",  # Legend location
    # Savefig
    "savefig.dpi": 300,  # Set DPI for saved figures
    "savefig.bbox": "tight",  # Adjust bounding box when saving
    "savefig.pad_inches": 0.05,
    # Text
    "text.usetex": False,  # Use LaTeX for text rendering
    # 'text.latex.preamble' : r'\usepackage{amsmath}',  # LaTeX preamble
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
}
