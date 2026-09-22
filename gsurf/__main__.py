"""
gSurf, started from the front door.

    python -m gsurf

Offers the tools, and asks what to open only once one has been picked -- a tool
knows which sources it takes, and which of them it cannot run without, so the
question is a short one with the required parts marked. What you answer is put
back the next time, so moving between tools does not mean naming the same files
again -- and it is written down, so neither does coming back tomorrow.

A tool can still be started on its own -- `python -m gsurf.tools.fold_axes`,
with or without arguments -- which is what a repeated run wants.
"""

from __future__ import annotations

import sys

# PyQt6 has to be imported before the backend: matplotlib picks the binding by
# looking at what is already in sys.modules.
import PyQt6.QtCore  # noqa: F401
from PyQt6 import QtWidgets


def main():
    # The QApplication before any window, or Qt exits without saying why.
    app = QtWidgets.QApplication(sys.argv)

    from .launcher import Launcher

    launcher = Launcher()
    launcher.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
