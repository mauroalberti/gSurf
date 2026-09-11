"""
The parts of gSurf that outlive any one tool.

A session says where we are, a map view draws it, and the readers under them
turn files into the arrays a kernel wants. The tools in `gsurf.tools` are what
is about one calculation and nothing else; everything they have in common is
here, so that the next one inherits it rather than copying it.

The package was called `app` while the pre-2026 `gSurf/` package was still in
the tree: on a case-insensitive filesystem -- macOS, Windows, both of which the
GitHub mirror serves -- `gsurf/` beside `gSurf/` is the same directory. That
package is gone, so the name is free.
"""
