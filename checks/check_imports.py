"""
A line layer transcribed into gstruct: what travels, and what is refused outright.

The import has no arithmetic in it worth checking, and that is exactly why it
needs checking. What it does is decide, for every column of a table, what the
format is allowed to say about it -- and the two ways that goes wrong are both
silent. A value written into a field the parser reads as one token comes back
truncated, so the file says something other than what it looks like; and a plane
written as an observation at a point it was never observed at is a claim nobody
made, indistinguishable afterwards from one somebody did.

So the geometry here is built to be known -- a trace due east a kilometre long,
a two-part trace whose parts total a known length, a row with no geometry at all
-- and the assertions are about what the file *says*: that a name the source
repeated comes out as two names and a `set=`, that nothing was joined across a
gap, that an axis nobody interpreted says so in the format's own grammar, and
that a strike is still recognisable as a strike after being turned into a dip
direction.

The four refusals are the other half, and each one is a file that would have been
wrong rather than unreadable: a kind of two words, a projection with no EPSG
code, a layer in degrees, and one angle column named without the other.

    python check_imports.py
"""

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(os.environ.get("GSURF_REPO", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO))

FAILURES = []

# Where everything is built, and the one trace every length is measured against:
# due east from the corner, exactly a kilometre.
X0, Y0 = 600000.0, 4420000.0

# The strike in the table, and the dip direction it has to become. A right-hand
# rule strike of 50 has the plane dipping to 140: known from the convention, not
# from the code, which is the point of asserting it.
STRIKE, DIP = 50.0, 35.0
DIP_DIRECTION = 140.0


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}{'   ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(label)


def store_in(directory, name="recent.ini"):
    """A settings file of this check's own, so the machine's is never touched."""

    from PyQt6 import QtCore

    return QtCore.QSettings(
        str(Path(directory) / name), QtCore.QSettings.Format.IniFormat
    )


def layer_at(directory, name="traces.gpkg", crs="EPSG:25833"):
    """
    The fixture: six rows, each one a case the writer has to decide about.

    F001 is the plain one. F002 is two parts totalling 600 m, which is where
    joining would show. F003 and F003 again are one name on two unrelated
    features. One row has no name, one has no geometry at all, and one carries a
    dip of 99 -- which is how a CARG sheet records contorted bedding, and which
    has to cost the row its plane and not its trace.
    """

    import geopandas as gpd
    from shapely.geometry import LineString, MultiLineString

    east = LineString([(X0, Y0), (X0 + 1000.0, Y0)])

    split = MultiLineString([
        [(X0, Y0 + 500.0), (X0 + 400.0, Y0 + 500.0)],
        [(X0 + 600.0, Y0 + 500.0), (X0 + 800.0, Y0 + 500.0)],
    ])

    rows = [
        ("F001", "Alpha", STRIKE, DIP, east),
        ("F002", "", STRIKE, DIP, split),
        ("F003", "", 10.0, 20.0, LineString([(X0, Y0 + 900.0), (X0 + 300.0, Y0 + 900.0)])),
        ("F003", "", 10.0, 20.0, LineString([(X0, Y0 + 1100.0), (X0 + 300.0, Y0 + 1100.0)])),
        (None, "", 10.0, 20.0, LineString([(X0, Y0 + 1300.0), (X0 + 300.0, Y0 + 1300.0)])),
        ("F009", "", 10.0, 99.0, LineString([(X0, Y0 + 1500.0), (X0 + 300.0, Y0 + 1500.0)])),
        ("F010", "", 10.0, 20.0, None),
    ]

    path = Path(directory) / name

    gpd.GeoDataFrame(
        {
            "code": [r[0] for r in rows],
            "nome": [r[1] for r in rows],
            "Tipologia": ["certo"] * len(rows),
            "strike": [r[2] for r in rows],
            "dip": [r[3] for r in rows],
        },
        geometry=[r[4] for r in rows],
        crs=crs,
    ).to_file(path, layer="faglie", driver="GPKG")

    return path


def one_line(directory, name, crs, ident="F001"):
    """One trace, for the checks that are about the header rather than the rows."""

    import geopandas as gpd
    from shapely.geometry import LineString

    path = Path(directory) / name

    gpd.GeoDataFrame(
        {"code": [ident]},
        geometry=[LineString([(11.0, 39.0), (11.1, 39.1)])]
        if crs == "EPSG:4326"
        else [LineString([(X0, Y0), (X0 + 100.0, Y0)])],
        crs=crs,
    ).to_file(path, layer="faglie", driver="GPKG")

    return path


def main():
    import gstruct
    from PyQt6 import QtWidgets

    from gsurf.imports import FROM, ImportDialog, Mapping, guess, transcript_of

    print("-- a layer read for what its columns mean --\n")

    with tempfile.TemporaryDirectory() as tmp:
        path = layer_at(tmp)

        proposed = guess(path, "faglie")

        check("the columns this recognises are guessed off their names",
              proposed.ident_field == "code" and proposed.label_field == "nome"
              and proposed.dip_dir_field == "strike" and proposed.dip_field == "dip",
              f"{proposed.ident_field}/{proposed.label_field}/"
              f"{proposed.dip_dir_field}/{proposed.dip_field}")

        check("and every column is offered to be carried through",
              set(proposed.keep_fields) >= {"code", "nome", "Tipologia"},
              str(proposed.keep_fields))

        mapping = Mapping(
            layer="faglie", ident_field="code", label_field="nome",
            dip_dir_field="strike", dip_field="dip", is_rhr_strike=True,
            keep_fields=("code", "Tipologia"),
        )

        text, report = transcript_of(path, mapping, project="una prova")

        dataset = gstruct.loads(text)

        # -- what came out ------------------------------------------------

        check("the file loads, and declares the layer's own projection",
              dataset.crs == "EPSG:25833"
              and dataset.meta["version"] == gstruct.VERSION,
              f"{dataset.crs} {dataset.meta.get('version')}")

        check("six features with a line became seven structures, the split one twice",
              report.features == 6 and report.structures == 7
              and len(dataset.structures) == 7,
              f"{report.features} features, {report.structures} structures")

        check("the row with no geometry is dropped, and counted as dropped",
              report.dropped.get("no geometry") == 1, str(report.dropped))

        idents = [structure.ident for structure in dataset.structures]

        check("every ident is unique, which a curation has to be able to rely on",
              len(set(idents)) == len(idents), str(idents))

        # -- a name the source used twice ---------------------------------

        check("a name on two unrelated features is suffixed, and says what it was",
              {"F003.1", "F003.2"} <= set(idents)
              and dataset.by_ident("F003.1").attrs.get("set") == "F003"
              and report.collided == 2,
              f"collided={report.collided}")

        check("and a name the source did not give is minted, not left blank",
              any(i.startswith("L") for i in idents) and report.minted == 1,
              str([i for i in idents if i.startswith("L")]))

        # -- the multipart, which is where joining would show -------------

        parts = [dataset.by_ident("F002.1"), dataset.by_ident("F002.2")]

        check("a multipart trace becomes two structures, sharing a set",
              all(p is not None for p in parts)
              and all(p.attrs.get("set") == "F002" for p in parts)
              and report.split == 1)

        # Measured on the parts if they are there, and on whatever carries the
        # name if they are not: a run that joined the two would put 800 m under
        # one name, and the number is what says so.
        spans = [p.length for p in parts if p is not None] or [
            p.length for p in dataset.structures if p.attrs.get("raw.code") == "F002"
        ]

        check("and nothing was joined across the gap: 400 m and 200 m, not 800",
              len(spans) == 2
              and abs(spans[0] - 400.0) < 0.1 and abs(spans[1] - 200.0) < 0.1,
              " + ".join(f"{m:.1f}" for m in spans) + " m")

        # -- rule 1: the source string survives the normalisation ---------

        alpha = dataset.by_ident("F001")

        check("the label travels, and the carried columns arrive as raw.*",
              alpha.label == "Alpha"
              and alpha.attrs.get("raw.Tipologia") == "certo"
              and alpha.attrs.get("raw.code") == "F001",
              str(alpha.attrs))

        check("a column nobody asked to carry is not in the file",
              "raw.nome" not in alpha.attrs and "raw.strike" not in alpha.attrs,
              str([k for k in alpha.attrs if k.startswith("raw.")]))

        # Read out rather than indexed at every use, so that a run where the
        # plane was written as something other than a fit reports each assertion
        # it breaks instead of stopping at the first with an IndexError.
        fit = alpha.fits[0] if alpha.fits else None

        check("a strike became a dip direction, once",
              fit is not None and fit.plane.dip_dir == DIP_DIRECTION
              and fit.plane.dip == DIP,
              str(fit.plane) if fit else "no fit")

        # The reason rule 1 exists, stated as the thing it prevents: without the
        # raw, a strike of 50 read as a strike and a dip direction of 140 read as
        # one are the same two numbers in the file, and no later reader can tell
        # which convention the survey used.
        check("and the table's own two numbers are still in the file beside it",
              fit is not None
              and fit.attrs.get("raw") == f"strike={STRIKE:.0f} dip={DIP:.0f}",
              fit.attrs.get("raw", "nothing") if fit else "no fit")

        # -- the plane is a fit over the whole trace, not an observation ---

        check(f"the plane is a fit carrying from={FROM}, with no diagnostics invented",
              fit is not None and fit.attrs.get("from") == FROM
              and not any(k in fit.attrs
                          for k in ("nvert", "snr", "flat", "jack", "verdict")),
              str(fit.attrs) if fit else "no fit")

        check("no attitude was written, there being no point anybody measured at",
              not any(s.attitudes for s in dataset.structures))

        # The end of the argument for `fit`: the plane answers everywhere along
        # the trace. An attitude anchored at the midpoint would answer near the
        # middle and fall out of reach at both ends.
        answers = [alpha.attitude_at(s) for s in (0.0, 500.0, 1000.0)]

        check("so it answers at both ends of the trace and in the middle",
              all(plane is not None and plane.dip_dir == DIP_DIRECTION
                  for plane, _ in answers)
              and all(said.startswith("fit:") for _, said in answers),
              ", ".join(said for _, said in answers))

        # -- rule 2: what the source does not say ------------------------

        axes = {span.axis: span for span in alpha.spans}

        # An axis left out of the file and an axis nobody has decided read the
        # same to `value_at` -- both `unknown` -- so the assertion is on the line
        # being there, which is the only difference and the whole of rule 2.
        check("both axes are written, unknown, with the reason for it",
              {"certainty", "exposure"} <= set(axes)
              and all(axes[a].value == "unknown" and axes[a].attrs.get("reason")
                      for a in ("certainty", "exposure")),
              str({a: (s.value, s.attrs) for a, s in axes.items()}))

        check("and no kind is claimed, the source having no column in the vocabulary",
              alpha.kind == "unknown" and "\n  kind " not in text,
              alpha.kind)

        # -- a bad angle costs the plane, not the trace -------------------

        contorted = dataset.by_ident("F009")

        check("a dip of 99 loses its plane and keeps its trace",
              contorted is not None and not contorted.fits
              and abs(contorted.length - 300.0) < 0.1
              and any("dip outside" in reason for reason in report.dropped),
              str(report.dropped))

        # Six rows held a readable plane and seven structures came out, so the
        # count is not the rows: the split trace carries its plane into both of
        # its parts, which is the only right answer -- both halves of one fault
        # dip the way the table says the fault dips.
        check("five rows with a readable plane become six fits, the split one twice",
              report.planes == 6
              and all(dataset.by_ident(i).fits for i in ("F002.1", "F002.2")),
              str(report.planes))

        # -- the file as a file -------------------------------------------

        check("what the file says about its own making is in it",
              dataset.meta.get("project") == "una prova"
              and "layer=faglie" in dataset.meta.get("source", "")
              and report.notes and dataset.meta.get("note"),
              str(report.notes[:1]))

        check("and it survives its own round trip, byte for byte",
              gstruct.dumps(dataset) == text)

        print("\n-- what is refused, and why each one would have been wrong --\n")

        dialog = ImportDialog()

        check("the dialog fills itself from the file it was pointed at",
              dialog.set_path(str(path)) is True
              and dialog.layer_combo.currentText() == "faglie"
              and dialog.ident_combo.currentText() == "code"
              and dialog.write_button.isEnabled()
              and "7" in dialog.report_label.text(),
              dialog.report_label.text())

        filled = dialog.mapping()

        check("and what it hands back is the mapping the writer takes",
              filled.layer == "faglie" and filled.ident_field == "code"
              and filled.label_field == "nome" and filled.dip_field == "dip"
              and set(filled.keep_fields) == {"code", "nome", "Tipologia",
                                              "strike", "dip"},
              str(filled.keep_fields))

        # A file with lines in it is the whole of what this can be pointed at,
        # and saying so is better than a mapping with nothing to map: there is
        # no path in a polygon or a point, and a structure is one path.
        polygons = Path(tmp) / "no-lines.gpkg"

        import geopandas as gpd
        from shapely.geometry import Point

        gpd.GeoDataFrame({"code": ["P1"]}, geometry=[Point(X0, Y0)],
                         crs="EPSG:25833").to_file(
                             polygons, layer="punti", driver="GPKG")

        said = []
        original_info = QtWidgets.QMessageBox.information
        QtWidgets.QMessageBox.information = staticmethod(
            lambda *args, **rest: said.append(args[2] if len(args) > 2 else "")
        )

        try:
            refused_layer = dialog.set_path(str(polygons))
        finally:
            QtWidgets.QMessageBox.information = original_info

        check("a file with no lines in it is turned away, with the reason",
              refused_layer is False and said and "no layer of lines" in said[-1]
              and dialog.layer_combo.currentText() == "faglie",
              (said[-1] if said else "nothing said").split("\n")[-1][:56])

        two_words = dialog.refusal(Mapping(kind="faglia diretta"))

        # The trap this closes, verified rather than asserted from the docstring:
        # the writer does not quote a kind and the parser takes one token, so a
        # kind of two words is not an error anywhere -- it is a different kind.
        truncated = gstruct.loads(
            "gstruct 0.2\n\nstructure F1 \"\"\n  kind faglia diretta\n"
        ).structures[0].kind

        check("a kind of two words is refused, because it would be silently cut",
              two_words is not None and "one word" in two_words
              and truncated == "faglia",
              f"round trip gives `{truncated}`")

        check("one angle column without the other is refused",
              dialog.refusal(Mapping(dip_dir_field="strike")) is not None
              and dialog.refusal(Mapping()) is None
              and dialog.refusal(Mapping(dip_dir_field="strike", dip_field="dip")) is None)

        # A projection with no EPSG code, for the same reason as the kind: `crs`
        # is read back as one token, so a WKT on that line loses all but its
        # first word and the file is then wrong about where it is.
        local = one_line(
            tmp, "local.gpkg",
            'LOCAL_CS["senza codice",UNIT["metre",1.0],AXIS["X",EAST],AXIS["Y",NORTH]]',
        )

        refused = ""

        try:
            transcript_of(local, Mapping(layer="faglie", ident_field="code"))
        except Exception as err:
            refused = str(err)

        check("a projection with no EPSG code is refused, not written as WKT",
              "EPSG" in refused and "truncated" in refused, refused[:70])

        degrees = one_line(tmp, "degrees.gpkg", "EPSG:4326")

        refused = ""

        try:
            transcript_of(degrees, Mapping(layer="faglie", ident_field="code"))
        except Exception as err:
            refused = str(err)

        check("and a layer in degrees is refused here rather than at the editor's door",
              "geographic" in refused and "metres" in refused, refused[:70])

        print("\n-- and the file opens in the tool it was written for --\n")

        from gsurf.session import Session
        from gsurf.tools import editor

        written = Path(tmp) / "imported.gstruct"
        written.write_text(text, encoding="utf-8")

        spec = dict(path=str(written), role="lines", layer="structures",
                    dip_dir_field=None, dip_field=None, is_rhr_strike=False)

        session = Session.open(frame_layers=[spec])
        window = editor.build(session, {"traces": spec})

        check("the editor opens it, with every structure on the chooser",
              window is not None and window.panel.chooser.count() == 7,
              f"{window.panel.chooser.count()} items" if window else "no window")

        if window is not None:
            window.select(0)

            check("and the band along the top reads the plane off the table",
                  any("fit" == kind for _, _, kind, _ in window.panel.view.runs)
                  if hasattr(window.panel.view, "runs")
                  else window.document.dataset.structures[0].attitude_at(1.0)[1]
                  .startswith("fit:"),
                  window.document.dataset.structures[0].attitude_at(1.0)[1])

            window.close()

        session.close()

        print("\n-- and the launcher puts it where the editor will ask for it --\n")

        from gsurf.launcher import Launcher
        from gsurf.recent import Recent

        recent = Recent(store_in(tmp, "recent.ini"))
        launcher = Launcher(recent=recent)

        # The dialog is not driven here -- what is being checked is the half
        # after it, which is the only part with a consequence that outlives the
        # run: a file nobody remembered is a file the next dialog does not offer.
        import gsurf.imports

        original = gsurf.imports.run
        gsurf.imports.run = lambda parent=None: str(written)

        try:
            back = launcher.import_lines()
        finally:
            gsurf.imports.run = original

        proposed = recent.proposed().get("traces") or {}

        check("the written file is remembered in the traces slot, as a spec",
              back == str(written)
              and proposed.get("path") == str(written)
              and proposed.get("role") == "lines"
              and launcher.chosen["traces"]["path"] == str(written),
              str(proposed.get("path")))

        cancelled = Launcher(recent=Recent(store_in(tmp, "empty.ini")))

        gsurf.imports.run = lambda parent=None: None

        try:
            nothing = cancelled.import_lines()
        finally:
            gsurf.imports.run = original

        check("and a dialog backed out of remembers nothing",
              nothing is None and "traces" not in cancelled.chosen)

        launcher.close()
        cancelled.close()

    print()

    if FAILURES:
        print(f"FAILED: {', '.join(FAILURES)}")
        return 1

    print("all checks passed")
    return 0


if __name__ == "__main__":
    from PyQt6 import QtWidgets

    app = QtWidgets.QApplication(sys.argv)

    sys.exit(main())
