"""
The trace editor: what holds along a trace, and what saving leaves alone.

Two things are worth checking here and they pull in opposite directions.

The first is that the editor writes to the file it opened, which nothing else in
this project does. What makes that safe is that it never rewrites the file: it
replaces the lines of the structure that was edited and leaves every other byte
where it was. That is measured against the real `curation.gstruct`, whose ten
lines of comment are the argument for why five thrusts are `exposed` -- and
which the format's own writer deletes, having nowhere to keep them. A Save that
loses those is the failure this whole arrangement exists to prevent, so the
check asserts both halves: that a whole-file rewrite would lose them, and that
saving an edited block does not.

The second is the picture. Precedence in this format is a computation over
several lines at once, so the synthetic file is built to make all five of its
answers come out at places that can be named. One trace runs due east for a
kilometre with a vertex every hundred metres; a fit covers the whole of it, a
stretch between 200 m and 400 m is rejected, and a compass reading sits at
800 m. So at 100 m the fit answers, at 300 m the refusal does, at 800 m the
measurement does, and at 500 m -- three hundred metres from the reading, past
the reach -- the fit takes it back. Any other answer is a number to explain.

    python check_editor.py
"""

import difflib
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(os.environ.get("GSURF_REPO", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO))

FAILURES = []

AOI = Path("/home/mauro/Documenti/Ricerca/AppenninoMeridionale/gstruct")

# The trace everything exact is measured along: due east, a vertex every 100 m.
X0, Y0 = 600000.0, 4420000.0

SOURCE = """gstruct 0.2
crs EPSG:25833
project "a synthetic file, built so the five answers are known"

# This comment belongs to no block. It sits between the header and the first
# structure, and the point of it is that editing either neighbour leaves it
# exactly here.

structure F001 "Alpha" fid=1
  kind fault
  span certainty * * certain
  span use @600200.00,4420000.00 @600400.00,4420000.00 rejected reason="prova"
  attitude @600800.00,4420000.00 plane 90/30 station=S1 src=field
  fit plane * * 100/40 from=trace-dem verdict=indipendente-dal-versante
  path 11
{alpha}

structure F002 "Beta" fid=2
  kind fault
  fit plane * * 12/88 from=trace-dem verdict=traccia-rettilinea
  path 2
    602000.00 4420000.00
    602500.00 4420000.00

structure F003 "Gamma" fid=3
  kind fault
  attitude @603000.00,4420000.00 plane 270/60 station=S2 src=field
  path 2
    603000.00 4420000.00
    603000.00 4421000.00

structure F004 "Delta" fid=4
  kind fault
  span certainty * * inferred
  path 2
    605000.00 4420000.00
    605400.00 4420000.00
"""


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}{'   ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(label)


def written(directory, name, text):
    path = Path(directory) / name
    path.write_text(text, encoding="utf8")
    return path


def source_text():
    alpha = "\n".join(f"    {X0 + n * 100.0:.2f} {Y0:.2f}" for n in range(11))

    return SOURCE.format(alpha=alpha)


def comment_lines(text):
    return [line for line in text.splitlines() if line.lstrip().startswith("#")]


def changed_lines(before, after):
    """The lines a save added or took away, and nothing about the rest."""

    return [
        line
        for line in difflib.unified_diff(
            before.splitlines(), after.splitlines(), lineterm="", n=0
        )
        if line[:1] in "+-" and line[:3] not in ("+++", "---")
    ]


def main():
    from PyQt6 import QtWidgets

    import gstruct
    from gsurf.curation import (
        Document,
        nearest_structure,
        place_on,
        point_on,
        provenance_of,
        stretch,
    )
    from gsurf.session import Session
    from gsurf.tools import editor as tool

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841

    with tempfile.TemporaryDirectory() as tmp:
        path = written(tmp, "synthetic.gstruct", source_text())
        original = path.read_text(encoding="utf-8")

        # -- the document, and what it does not touch ---------------------

        print("\n-- the document --\n")

        document = Document(path)

        check("every structure in the text is a structure in the model",
              len(document.blocks) == len(document.dataset.structures) == 4,
              f"{len(document.blocks)} block(s)")

        check("a block is the lines of one structure, geometry and all",
              document.text_of(0).startswith('structure F001 "Alpha"')
              and document.text_of(0).rstrip().endswith("601000.00 4420000.00")
              and "F002" not in document.text_of(0))

        check("and the comment above the first one belongs to neither",
              "This comment belongs to no block" not in document.text_of(0))

        document.save()

        check("saving a file nobody edited gives back the same bytes",
              path.read_text(encoding="utf-8") == original)

        # -- a block that will not go in ----------------------------------

        print("\n-- refusals --\n")

        was = document.text_of(0)

        refusals = (
            ("an empty block", ""),
            ("a block that starts with a span", "  span use * * rejected"),
            ("a `structure` line that is indented", "  structure F001"),
            ("two structures in one block", was + "\n\nstructure F009"),
            ("an observation inside one",
             was + "\nobservation S9 @1,2 plane 55/70"),
            ("an anchor that is not one", 'structure F001\n  span use @nope * rejected'),
        )

        for label, text in refusals:
            try:
                document.replace(0, text)
                check(f"{label} is refused", False, "it went in")
            except ValueError as err:
                check(f"{label} is refused", True, str(err)[:58])

        check("and a refused block leaves the document as it was",
              document.text_of(0) == was and not document.dirty)

        # -- what holds along a trace -------------------------------------

        print("\n-- what holds --\n")

        alpha, beta, gamma, _delta = document.dataset.structures

        def said_at(structure, s):
            return structure.attitude_at(s, 250.0)[1]

        check("a fit covering the trace answers where nothing else does",
              said_at(alpha, 100.0) == "fit:indipendente-dal-versante",
              said_at(alpha, 100.0))

        check("a rejected stretch answers with the reason it was rejected for",
              said_at(alpha, 300.0) == "rifiutata:prova", said_at(alpha, 300.0))

        check("a measurement within reach beats the fit under it",
              said_at(alpha, 800.0) == "misurata:S1@0m", said_at(alpha, 800.0))

        check("and past its reach the fit takes it back",
              said_at(alpha, 500.0).startswith("fit:"), said_at(alpha, 500.0))

        check("a fit off a straight trace holds nothing at all",
              said_at(beta, 250.0) == "assente", said_at(beta, 250.0))

        check("a measurement with no fit under it still answers, further off",
              said_at(gamma, 1000.0) == "misurata-lontana:1000m",
              said_at(gamma, 1000.0))

        # The band the panel paints, over the same trace: one sample a metre,
        # so a run's edges are the stretch that was written down.
        runs = tool.runs_of(provenance_of(alpha, samples=1001, max_gap=250.0))
        rejected = [run for run in runs if run[2] == "rifiutata"]

        check("the band's own run of refusal is the stretch in the file",
              len(rejected) == 1
              and abs(rejected[0][0] - 200.0) <= 1.0
              and abs(rejected[0][1] - 400.0) <= 1.0,
              str([(round(a), round(b), kind) for a, b, kind, _ in runs]))

        check("and every class it paints is one the format answers with",
              {run[2] for run in runs} <= set(tool.PROVENANCE_TINT),
              str({run[2] for run in runs}))

        # -- a click, and an anchor ---------------------------------------

        print("\n-- the click --\n")

        found = nearest_structure(document.dataset, X0 + 300.0, Y0 + 50.0)

        check("a click finds the trace under it, and where along it",
              found is not None and found[0] == 0
              and abs(found[1] - 300.0) < 1e-6 and abs(found[2] - 50.0) < 1e-6,
              str(found))

        check("and finds nothing where nothing is within reach",
              nearest_structure(
                  document.dataset, X0 + 300.0, Y0 + 50.0, within=10.0
              ) is None)

        # The anchor is snapped onto the trace and not written where the mouse
        # was: `resolve` projects it back to get `s`, so a point fifty metres
        # off the line would read the same and say something false about where
        # anybody stood.
        s, _ = place_on(alpha.path, X0 + 300.0, Y0 + 50.0)
        snapped = point_on(alpha.path, s)

        check("a picked anchor is snapped onto the trace it names",
              abs(snapped[0] - (X0 + 300.0)) < 1e-6 and abs(snapped[1] - Y0) < 1e-6,
              str(snapped))

        check("a rejected stretch draws over the ground it covers",
              [round(x) for x, _ in stretch(alpha.path, 200.0, 400.0)]
              == [600200, 600300, 600400],
              str(stretch(alpha.path, 200.0, 400.0)))

        # -- the tool -----------------------------------------------------

        print("\n-- the tool --\n")

        check("it declares the file it edits as the thing it cannot run without",
              tool.WANTS["traces"] == "required" and tool.WANTS["dem"] == "optional")

        spec = dict(path=str(path), role="traces")
        session = Session.open(frame_layers=[spec])
        window = tool.build(session, {"traces": spec})

        check("the file opens as a window", window is not None)

        # The precondition the unapplied-work guard rests on: what is in the box
        # and what is in the document are the same string until somebody types.
        check("the box holds the block exactly as the document has it",
              window.panel.text.toPlainText() == window.document.text_of(0))

        check("every structure is offered, not only the ones carrying a plane",
              window.panel.chooser.count() == 4,
              f"{window.panel.chooser.count()} offered")

        window.panel.carrying.setChecked(True)

        # Delta drops out: a mapped contact nobody has read a plane off yet,
        # which is 348 of the 393 traces of the real fault layer. It is still
        # editable -- the filter shortens the list, it does not lock anything.
        check("and the filter leaves the ones something was read on",
              window.panel.chooser.count() == 3,
              f"{window.panel.chooser.count()} carrying a plane")

        window.panel.carrying.setChecked(False)

        # A click on the northward trace, half way up it.
        window.pick(603000.0, 4420500.0)

        check("a click on the map selects the trace it landed on",
              window.index == 2, f"index {window.index}")

        check("and says what holds there, in the format's own words",
              "misurata-lontana:500m" in window.statusBar().currentMessage(),
              window.statusBar().currentMessage())

        # The template puts the cursor somewhere known; the two shift-clicks
        # fill the two anchors in turn.
        window.panel.add_line('  span use * * rejected reason="check"')
        window.pick(603000.0, 4420200.0, anchor=True)
        window.pick(603000.0, 4420600.0, anchor=True)

        typed = [
            line for line in window.panel.text.toPlainText().splitlines()
            if "span use" in line
        ]

        check("two picked anchors fill the two the template left open",
              typed == [
                  '  span use @603000.00,4420200.00 @603000.00,4420600.00 '
                  'rejected reason="check"'
              ],
              str(typed))

        check("and what was typed is not in the document until it is applied",
              window.panel.text.toPlainText()
              != window.document.text_of(window.index)
              and not window.document.dirty)

        check("applying it reads it in", window.panel.apply_block())

        after_edit = window.document.dataset.structures[2]
        answers = {
            kind for _, _, _, kind in provenance_of(after_edit, samples=201)
        }

        check("and the refusal is in force along the ground it named",
              after_edit.attitude_at(400.0, 250.0)[1] == "rifiutata:check"
              and "rifiutata" in answers,
              str(sorted(answers)))

        check("the map has the rejected stretch to draw now",
              len(window.refused.get_xdata()) > 0,
              f"{len(window.refused.get_xdata())} point(s)")

        # -- and the one gesture that could throw work away ----------------

        print("\n-- leaving a block --\n")

        asked = []
        QtWidgets.QMessageBox.question = staticmethod(
            lambda *args, **rest: asked.append(args[2] if len(args) > 2 else "")
            or QtWidgets.QMessageBox.StandardButton.Cancel
        )

        window.panel.text.appendPlainText("  span exposure * * covered")
        window.pick(X0 + 300.0, Y0)

        check("clicking another trace with a block typed and not applied asks",
              len(asked) == 1 and "not applied" in asked[0],
              f"asked {len(asked)} time(s)")

        check("and saying no leaves the map and the text on the same structure",
              window.index == 2 and window.panel.index == 2
              and "span exposure * * covered" in window.panel.text.toPlainText(),
              f"index {window.index}")

        QtWidgets.QMessageBox.question = staticmethod(
            lambda *args, **rest: QtWidgets.QMessageBox.StandardButton.Discard
        )

        window.pick(X0 + 300.0, Y0)

        check("and saying yes moves on, with the document none the wiser",
              window.index == 0
              and "span exposure * * covered" not in window.document.text_of(2),
              f"index {window.index}")

        # The reach decides what holds, not what is written, so turning it must
        # redraw the picture and leave the box alone -- the same loss as above,
        # from a dial instead of a click.
        window.panel.text.appendPlainText("  span exposure * * covered")
        window.panel.gap_spin.setValue(600.0)

        check("turning the reach dial redraws the picture and not the text",
              "span exposure * * covered" in window.panel.text.toPlainText())

        def measured(runs):
            return sum(b - a for a, b, kind, _ in runs if kind == "misurata")

        narrow = tool.runs_of(provenance_of(alpha, samples=201, max_gap=250.0))
        wide = tool.runs_of(provenance_of(alpha, samples=201, max_gap=600.0))

        check("and a wider reach is one measurement answering for more trace",
              measured(wide) > measured(narrow),
              f"{measured(narrow):.0f} m -> {measured(wide):.0f} m")

        window.panel.gap_spin.setValue(250.0)
        window.panel._redraw()

        # -- two projections, and the one place each crosses ---------------

        print("\n-- projections --\n")

        check("one projection for the file and the map means no transform",
              window._forward is None and window._back is None)

        from pyproj import CRS, Transformer

        theirs, ours = CRS.from_epsg(25833), CRS.from_epsg(32632)
        window._forward = Transformer.from_crs(theirs, ours, always_xy=True)
        window._back = Transformer.from_crs(ours, theirs, always_xy=True)

        here = (X0 + 300.0, Y0)
        there = window.on_map([here])[0]
        back = window.in_file(*there)

        check("two projections put the map somewhere else entirely",
              abs(there[0] - here[0]) > 100000.0, f"{there[0] - here[0]:.0f} m")

        check("and a point out to the map and back lands where it started",
              abs(back[0] - here[0]) < 0.001 and abs(back[1] - here[1]) < 0.001,
              str(back))

        window._forward = window._back = None

        # -- saving -------------------------------------------------------

        print("\n-- saving --\n")

        before = path.read_text(encoding="utf-8")
        window.save()
        saved = path.read_text(encoding="utf-8")

        check("saving writes the line that was added and nothing else",
              changed_lines(before, saved) == [
                  '+  span use @603000.00,4420200.00 @603000.00,4420600.00 '
                  'rejected reason="check"'
              ],
              str(changed_lines(before, saved))[:70])

        check("and the file still reads as what it was",
              len(gstruct.loads(saved).structures) == 4)

        # -- the real file, and the reason any of this is done this way ----

        print("\n-- the reasoning that would have been lost --\n")

        real = AOI / "curation.gstruct"

        if not real.exists():
            print(f"(not here: {real})")
        else:
            copy = Path(tmp) / "curation.gstruct"
            shutil.copy(real, copy)

            kept = copy.read_text(encoding="utf-8")
            rebuilt = gstruct.dumps(gstruct.loads(kept))

            check("the format's own writer has nowhere to keep a comment",
                  len(comment_lines(kept)) == 10 and not comment_lines(rebuilt),
                  f"{len(comment_lines(kept))} lines in, "
                  f"{len(comment_lines(rebuilt))} out")

            curation = Document(copy)
            curation.replace(
                0,
                curation.text_of(0)
                + '\n  span use * * rejected src=check reason="prova"',
            )
            curation.save()

            written_back = copy.read_text(encoding="utf-8")

            check("editing a block of the real file changes that one line",
                  changed_lines(kept, written_back) == [
                      '+  span use * * rejected src=check reason="prova"'
                  ],
                  str(changed_lines(kept, written_back))[:70])

            check("and the ten lines saying why it says what it says stay put",
                  comment_lines(written_back) == comment_lines(kept))

        # -- what it will not open ----------------------------------------

        print("\n-- what it will not open --\n")

        # `build` says why it will not run by putting a box on screen, and a
        # modal box in a check is a check that hangs. The refusal is the thing
        # being tested, so the box is replaced by something that writes it down.
        shown = []
        QtWidgets.QMessageBox.critical = staticmethod(
            lambda *args, **rest: shown.append(args[2] if len(args) > 2 else "")
        )

        layer = written(tmp, "not-a-format.gpkg", "")

        check("a layer is not a thing this edits, and it says why",
              tool.build(session, {"traces": dict(path=str(layer))}) is None
              and shown and "export_gsurf.py" in shown[-1],
              (shown[-1] if shown else "nothing said").split("\n")[-1][:60])

        empty = written(tmp, "empty.gstruct", "gstruct 0.2\ncrs EPSG:25833\n")

        check("nor is a file with no structure in it",
              tool.build(session, {"traces": dict(path=str(empty))}) is None
              and "no structure" in shown[-1],
              shown[-1].split("\n")[-1][:60])

        ahead = written(
            tmp, "ahead.gstruct", "gstruct 9.9\ncrs EPSG:25833\n\nstructure F1\n"
        )

        check("nor one written by a gstruct that does not exist yet",
              tool.build(session, {"traces": dict(path=str(ahead))}) is None
              and "9.9" in shown[-1],
              shown[-1].split("\n")[-1][:60])

    print()

    if FAILURES:
        print(f"FAILED: {', '.join(FAILURES)}")
        return 1

    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
