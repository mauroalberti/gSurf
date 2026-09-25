"""
Run every check, and say what failed.

    python checks/run.py

Qt is put in its off-screen mode here rather than left to the caller, because
these open real windows and drive real mouse events through them: on a desktop
they would flash up and steal focus, and on a machine with no display they
would not start at all.

    python checks/run.py --show     to watch them instead

Each check is a script that can also be run on its own, which is how they are
used while working on the thing they check.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent

CHECKS = (
    ("check_interaction.py", "the map: clicks, drags, zoom, legend, screenshots"),
    ("check_session.py", "the session, and a map with no DEM under it"),
    ("check_folds.py", "fold axes against an axis that is known"),
    ("check_traces.py", "attitudes along a trace, against a plane that is known"),
    ("check_bare_traces.py", "a line layer with no attitude columns, and the fit that gives it some"),
    ("check_attitude_export.py", "a computed attitude leaving the tool, and landing where it was read"),
    ("check_curation.py", "the gstruct boundary, and which way a normal points across it"),
    ("check_gstruct.py", "a .gstruct as a source, and a curation laid over one"),
    ("check_editor.py", "the trace editor: what holds along a trace, and a save that loses nothing"),
    ("check_rotations.py", "how an axis field turns, against a rotation that is known"),
    ("check_qgis_project.py", "a QGIS project, read for its layers and their colours"),
    ("check_imports.py", "a line layer transcribed into gstruct, and what is refused"),
    ("check_launcher.py", "the launcher: the question each tool asks, and the handover"),
    ("check_recent.py", "the answers, kept from one run to the next"),
    ("check_sections.py", "sections: one profile dragged, the bundle on release"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="let the windows appear")
    args = parser.parse_args()

    environment = dict(os.environ)
    if not args.show:
        environment.setdefault("QT_QPA_PLATFORM", "offscreen")

    failed = []

    for script, what in CHECKS:
        print(f"\n{'=' * 72}\n{script}  --  {what}\n{'=' * 72}")

        started = perf_counter()
        result = subprocess.run(
            [sys.executable, str(HERE / script)],
            env=environment,
        )
        elapsed = perf_counter() - started

        print(f"({elapsed:.1f} s)")

        if result.returncode != 0:
            failed.append(script)

    print()

    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1

    print(f"all {len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
