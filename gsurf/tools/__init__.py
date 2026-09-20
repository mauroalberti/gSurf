"""
The tools, one module each, and the list the launcher offers them from.

A tool is a window plus what it has to be given, and the two are kept apart:
`WANTS` says which sources it takes and which of them it cannot do without,
`build` gets a session that is already open together with the choices it was
opened on. That is what lets the launcher ask for the right things for the tool
that was picked, instead of asking for everything any tool might want.

What is here is only what can be shown without importing anything: a name, a
line of description, and where the module is. Importing a tool pulls in
matplotlib, and the fold axes pull in geogst and mplstereonet on top of that,
so the import waits until a tool is actually chosen -- otherwise the launcher
would pay for every tool in the tree before it drew its first window.
"""

from __future__ import annotations

from importlib import import_module

TOOLS = (
    dict(
        name="Plane on a DEM",
        module="gsurf.tools.intersection",
        summary="Lay an unbounded plane on the topography and watch where it crops out.",
    ),
    dict(
        name="Fold axes",
        module="gsurf.tools.fold_axes",
        summary="Drag a window over bedding attitudes and read the girdle they spread on.",
    ),
    dict(
        name="Sections",
        module="gsurf.tools.profiles",
        summary="Drag a section trace over the map and watch the geology under it.",
    ),
)


def load(entry):
    """The module of a tool, imported now rather than at start-up."""

    return import_module(entry["module"])
