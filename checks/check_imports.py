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


# The synthetic topography: one inclined plane, so the answer to "what does the
# best fit through a trace draped on it come out as" is known from the arithmetic
# that built it rather than from the code being checked. Any curve lying on a
# plane lies on it exactly, so every window that has a lever arm must return
# this, and the only scatter is the DEM's own cell quantisation -- half a cell of
# gradient, 1.4 m here.
RELIEF_DIP, RELIEF_DIP_DIR = 30.0, 90.0
CELL = 5.0

# Where the V turns, and therefore the only stretch of it that can carry an
# attitude. The first limb is hypot(1200, 600) long, which is where the apex
# falls along the trace; twice that is the whole of it.
APEX_X = X0 + 1300.0
APEX_S = (1200.0 ** 2 + 600.0 ** 2) ** 0.5


def relief_at(directory, name="plane.tif", crs="EPSG:25833"):
    """A DEM that is one plane, dipping 30 degrees due east and nothing else."""

    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    cols, rows = int(2700 / CELL), int(1600 / CELL)

    # Cell centres, so that a sample anywhere inside a cell is off by at most
    # half a cell of gradient. Taking the corner instead would put a systematic
    # half-cell shift into every elevation and tilt the answer.
    x = X0 + (np.arange(cols) + 0.5) * CELL
    z = 2000.0 - np.tan(np.radians(RELIEF_DIP)) * (x - X0)

    path = Path(directory) / name

    with rasterio.open(
        path, "w", driver="GTiff", width=cols, height=rows, count=1,
        dtype="float32", crs=crs,
        transform=from_origin(X0, Y0 + 1400.0, CELL, CELL),
    ) as out:
        out.write(np.repeat(z[None, :], rows, axis=0).astype("float32"), 1)

    return path


def draped_at(directory, name="draped.gpkg", angles=False):
    """
    Four traces on that plane, each one a different thing for the gate to say.

    VEE turns once, so exactly the stretch around the bend can carry a plane and
    the rest cannot -- which is what makes it the test that a fit is written over
    the stretch that held and not over the trace. ZIG turns everywhere, so the
    whole path holds and the extent has no neighbour to stop at. RING is a
    closed circle, which is the case that was silently broken: its two path ends
    are one point. FLAT is dead straight, and has to come out with nothing.
    """

    import geopandas as gpd
    import numpy as np
    from shapely.geometry import LineString

    def densify(points, step=20.0):
        out = []
        for (ax, ay), (bx, by) in zip(points, points[1:]):
            count = max(2, int(((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 / step))
            out.extend(
                (ax + t * (bx - ax), ay + t * (by - ay))
                for t in (i / count for i in range(count))
            )
        out.append(points[-1])
        return LineString(out)

    vee = densify([(X0 + 100.0, Y0), (APEX_X, Y0 + 600.0), (X0 + 2500.0, Y0)])

    teeth = [(X0 + 100.0, Y0 + 700.0)]
    for tooth in range(6):
        teeth.append((X0 + 200.0 + tooth * 200.0, Y0 + 700.0 + 100.0 * (tooth % 2)))
    zig = densify(teeth)

    turn = np.radians(np.arange(0.0, 360.0 + 1e-9, 6.0))
    ring = LineString([
        (X0 + 700.0 + 200.0 * float(np.sin(a)), Y0 + 1100.0 + 200.0 * float(np.cos(a)))
        for a in turn
    ])

    flat = densify([(X0 + 1400.0, Y0 + 1000.0), (X0 + 2500.0, Y0 + 1000.0)])

    columns = {"code": ["VEE", "ZIG", "RING", "FLAT"]}

    if angles:
        # A plane in the table as well, so that one structure carries two kinds
        # of fit and their order can be asked about. Deliberately nothing like
        # the topography's: 200/70 against 90/30 is unmistakable.
        columns["immersione"] = [200.0] * 4
        columns["inclinazione"] = [70.0] * 4

    path = Path(directory) / name

    gpd.GeoDataFrame(
        columns, geometry=[vee, zig, ring, flat], crs="EPSG:25833"
    ).to_file(path, layer="tracce", driver="GPKG")

    return path


def stations_at(directory, name="stazioni.gpkg", crs="EPSG:25833"):
    """
    Four measured points against F001, which runs due east a kilometre from Y0.

    S1 stands on it, S2 is fifty metres off it, S3 is four hundred, and S4 has a
    position and no plane. Each one is a different answer the writer has to give,
    and none of them is "dropped".
    """

    import geopandas as gpd
    from shapely.geometry import Point

    rows = [
        ("S1", 200.0, 40.0, Point(X0 + 100.0, Y0)),
        ("S2", 210.0, 45.0, Point(X0 + 600.0, Y0 + 50.0)),
        ("S3", 220.0, 50.0, Point(X0 + 500.0, Y0 - 400.0)),
        ("S4", None, None, Point(X0 + 900.0, Y0 + 5.0)),
    ]

    path = Path(directory) / name

    gpd.GeoDataFrame(
        {
            "sigla": [r[0] for r in rows],
            "immersione": [r[1] for r in rows],
            "inclinazione": [r[2] for r in rows],
        },
        geometry=[r[3] for r in rows], crs=crs,
    ).to_file(path, layer="punti", driver="GPKG")

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

        # Kept under its own name because `text` is reused by the sections
        # below, and it is this file -- seven structures off the faglie layer --
        # that the editor is opened on at the end.
        transcript = text

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

        print("\n-- a plane read off the topography, one per stretch that holds --\n")

        from gsurf.imports import FROM_DEM

        relief = relief_at(tmp)
        draped = draped_at(tmp)

        text, fitted = transcript_of(
            draped, Mapping(layer="tracce", ident_field="code",
                            dem_path=str(relief)),
        )

        read = gstruct.loads(text)

        vee = read.by_ident("VEE")
        zig = read.by_ident("ZIG")
        ring = read.by_ident("RING")
        flat = read.by_ident("FLAT")

        planes = [
            (structure.ident, structure.fits[0].plane)
            for structure in (vee, zig, ring)
            if structure is not None and structure.fits
        ]

        # The DEM is one plane and every one of these traces lies on it, so the
        # answer is the arithmetic that built the raster and not anything in the
        # code. Two degrees of slack for the DEM's own cell quantisation, which
        # is half a cell of gradient -- 1.4 m of height over 5 m of ground.
        check("the plane the DEM was built from is what comes back off every trace",
              len(planes) == 3
              and all(abs(plane.dip - RELIEF_DIP) <= 2.0
                      and abs(plane.dip_dir - RELIEF_DIP_DIR) <= 2.0
                      for _, plane in planes),
              ", ".join(f"{ident} {plane}" for ident, plane in planes))

        # The load-bearing negative, and the reason the gate is in this at all: a
        # plane through a straight trace is arbitrary rather than imprecise, and
        # writing one would be a number nobody could tell from a measurement.
        check("a dead straight trace carries no fit at all, having no plane to carry",
              flat is not None and not flat.fits and fitted.silent == 1,
              f"{len(flat.fits) if flat else '?'} fits, silent={fitted.silent}")

        stretch = vee.fits[0] if vee is not None and vee.fits else None

        check("the V's one fit is the stretch that turns, not the trace it is on",
              stretch is not None
              and stretch.s0 < APEX_S < stretch.s1
              and (stretch.s1 - stretch.s0) < vee.length / 10.0,
              f"{stretch.s0:.0f}..{stretch.s1:.0f} of {vee.length:.0f} m, "
              f"apex at {APEX_S:.0f}" if stretch else "no fit")

        # The widening, and where it stops. `runs` reports the interval the
        # window *centres* covered, which for one position is a single step -- so
        # a fit read over 250 m of trace would claim 25 m of it, and `attitude_at`
        # would answer `assente` over ground the plane was computed from. It
        # reaches half a window further, and no further than a neighbouring
        # verdict: on either side of the V's bend the gate said `line`.
        reach = (stretch.s1 - stretch.s0) if stretch else 0.0
        window = float(stretch.attrs["window"]) if stretch else 0.0

        check("and it does not widen across a verdict: the gate said line either side",
              stretch is not None and reach < window,
              f"{reach:.0f} m claimed, read over {window:.0f} m")

        whole = zig.fits[0] if zig is not None and zig.fits else None

        check("a trace that holds throughout is claimed throughout, `*` at the path end",
              whole is not None and whole.start is None
              and (whole.s1 - whole.s0) > 0.95 * zig.length,
              f"{whole.s0:.0f}..{whole.s1:.0f} of {zig.length:.0f} m" if whole
              else "no fit")

        # The regression, and it was a silent one: on a closed trace `path[-1]`
        # is `path[0]`, so a fit reaching both ends written as two anchors is one
        # coordinate twice, and reads back covering nothing. Eight traces of
        # `elementi_tettonici` are rings, and every fit on them was lost this way.
        loop = ring.fits[0] if ring is not None and ring.fits else None

        check("a closed ring keeps its fit: two anchors at one point would cover nothing",
              loop is not None
              and ring.path[0] == ring.path[-1]
              and loop.start is None and loop.end is None
              and (loop.s1 - loop.s0) > 0.99 * ring.length,
              f"{loop.s1 - loop.s0:.0f} of {ring.length:.0f} m" if loop
              else "no fit")

        check("the fit says what read it, over how much, and to what verdict",
              stretch is not None
              and stretch.attrs.get("from") == FROM_DEM
              and stretch.attrs.get("span_verdict") == "held"
              and stretch.attrs.get("window") and stretch.attrs.get("windows"),
              str(stretch.attrs) if stretch else "no fit")

        # FORMAT.md reserves `nvert` for digitised vertices and says in as many
        # words that gSurf is not to write it, sampling the DEM instead. The rest
        # are `export_geology.py`'s error budget, which this has no eps for.
        check("and writes none of the other producer's numbers, having none of them",
              stretch is not None
              and not any(key in stretch.attrs
                          for key in ("nvert", "dof", "snr", "flat", "jack",
                                      "eps", "plan", "drape", "verdict")),
              str(sorted(stretch.attrs)) if stretch else "no fit")

        check("and the file still survives its own round trip",
              gstruct.dumps(read) == text)

        # Three facts, counted apart. Together they would say the topography
        # refused 1200 traces of `elementi_tettonici` when most were never asked:
        # 106 of the first 1200 over the DEM are off it, and the sheet's median
        # trace is 165 m against a 250 m window.
        check("the three ways of carrying no fit are counted apart, not summed",
              all(hasattr(fitted, name)
                  for name in ("silent", "too_short", "unreached"))
              and fitted.too_short == 0 and fitted.unreached == 0,
              f"silent={fitted.silent} short={fitted.too_short} "
              f"off={fitted.unreached}")

        elsewhere = relief_at(tmp, "utm32.tif", crs="EPSG:25832")

        refused = ""

        try:
            transcript_of(draped, Mapping(layer="tracce", ident_field="code",
                                         dem_path=str(elsewhere)))
        except Exception as err:
            refused = str(err)

        check("a DEM in another projection is refused, not silently sampled",
              "25832" in refused and "25833" in refused, refused[:70])

        # Given up on rather than finished, which the file has to be able to say.
        # A fit on the first N structures and none on the rest would leave a
        # structure with no fit meaning either "refused" or "never reached", and
        # those are opposite facts.
        _, given_up = transcript_of(
            draped,
            Mapping(layer="tracce", ident_field="code", dem_path=str(relief)),
            progress=lambda done, total: done == 0,
        )

        check("a fitting given up on writes no fit at all, and says so in the file",
              given_up.stopped and given_up.fits == 0
              and any("interrotto" in note for note in given_up.notes),
              f"stopped={given_up.stopped} fits={given_up.fits}")

        print("\n-- the measured points, attached or kept with the reason --\n")

        from gsurf.imports import NO_ATTITUDE, UNATTACHED

        points = stations_at(tmp)

        with_points = Mapping(
            layer="faglie", ident_field="code", label_field="nome",
            dip_dir_field="strike", dip_field="dip", is_rhr_strike=True,
            keep_fields=("code",),
            points_path=str(points), points_layer="punti",
            points_ident_field="sigla", points_dip_dir_field="immersione",
            points_dip_field="inclinazione",
        )

        text, joined = transcript_of(path, with_points)

        near = gstruct.loads(text)
        alpha = near.by_ident("F001")

        stations = {a.attrs.get("station"): a for a in alpha.attitudes}

        check("a point standing on the trace attaches to it, and says it stood on it",
              joined.attached == 2 and "S1" in stations
              and abs(stations["S1"].s - 100.0) < 0.1
              and stations["S1"].attrs.get("off") == "0.0",
              f"attached={joined.attached} "
              + (f"s={stations['S1'].s:.1f}" if "S1" in stations else "no S1"))

        # `off` is the one number that says how much of an attachment this was:
        # `s` is derived and looks equally exact at any distance from the trace.
        check("and one fifty metres off it attaches too, with the fifty metres written",
              "S2" in stations
              and abs(float(stations["S2"].attrs.get("off", "0")) - 50.0) < 0.1
              and abs(stations["S2"].s - 600.0) < 0.1,
              stations["S2"].attrs.get("off") if "S2" in stations else "no S2")

        # The whole point of attaching anything: FORMAT.md's precedence puts a
        # measurement inside `max_gap` above every fit, so the stretch nearest the
        # outcrop stops reporting the column and starts reporting the compass --
        # and further along, where no measurement reaches, it goes back.
        at_outcrop = alpha.attitude_at(100.0)
        far_off = alpha.attitude_at(1000.0)

        check("a measurement outranks the plane from the table, and only where it reaches",
              at_outcrop[1].startswith("misurata:S1")
              and at_outcrop[0].dip_dir == 200.0
              and far_off[1].startswith("fit:")
              and far_off[0].dip_dir == DIP_DIRECTION,
              f"{at_outcrop[1]} / {far_off[1]}")

        kept = {ob.ident: ob for ob in near.observations}

        check("a point past the threshold is kept, with the distance and the threshold",
              joined.observations == 2 and "S3" in kept
              and kept["S3"].attrs.get("unattached") == UNATTACHED
              and kept["S3"].attrs.get("nearest") == "F001"
              and kept["S3"].attrs.get("threshold") == "100"
              and abs(float(kept["S3"].attrs.get("distance", 0)) - 400.0) < 0.1,
              str(kept["S3"].attrs) if "S3" in kept else str(list(kept)))

        # Rule 3 has two causes and one answer. A point with no plane is not a
        # measurement to attach, and dropping it would lose a station that was
        # visited -- so it is written with a plane of nothing and the reason.
        check("and one with no plane is kept for the other reason, not thrown away",
              "S4" in kept and kept["S4"].plane is None
              and kept["S4"].attrs.get("no_attitude") == NO_ATTITUDE
              and "unattached" not in kept["S4"].attrs,
              str(kept["S4"].attrs) if "S4" in kept else str(list(kept)))

        check("and a file with observations in it still round trips",
              gstruct.dumps(near) == text)

        # A station's code column is not called what a trace's ident is called,
        # and the guessing is the whole of why this dialog asks so little: what
        # it picks becomes the name `attitude_at` reports every measurement by.
        check("the dialog guesses a station layer's own columns, code included",
              dialog.set_points(str(points)) is True
              and dialog.station_combo.currentText() == "sigla"
              and dialog.points_dip_dir_combo.currentText() == "immersione"
              and dialog.points_dip_combo.currentText() == "inclinazione",
              f"{dialog.station_combo.currentText()}/"
              f"{dialog.points_dip_dir_combo.currentText()}")

        dialog._clear_points()

        check("and backing out of the point layer leaves the mapping without one",
              not dialog.mapping().has_points
              and dialog.mapping().points_path is None
              and not dialog.points_layer_combo.isEnabled())

        # Two ways of writing the same positions, and one meaning, so this is not
        # a decision worth refusing over -- unlike the DEM, where the dip
        # direction itself would have come out of the wrong north.
        import geopandas as gpd

        elsewhere_points = Path(tmp) / "in4326.gpkg"
        gpd.read_file(points, layer="punti").to_crs("EPSG:4326").to_file(
            elsewhere_points, layer="punti", driver="GPKG")

        moved = Mapping(**{**vars(with_points), "points_path": str(elsewhere_points)})

        _, reprojected = transcript_of(path, moved)

        check("a point layer in another projection is reprojected, and the note says so",
              reprojected.attached == joined.attached
              and any("riproiettati" in note for note in reprojected.notes),
              f"attached={reprojected.attached}")

        naked = gpd.read_file(points, layer="punti").set_crs(None, allow_override=True)
        nowhere = Path(tmp) / "nocrs.gpkg"
        naked.to_file(nowhere, layer="punti", driver="GPKG")

        refused = ""

        try:
            transcript_of(path, Mapping(
                **{**vars(with_points), "points_path": str(nowhere)}))
        except Exception as err:
            refused = str(err)

        check("and one with no projection at all is refused: a distance needs units",
              "no projection" in refused and "metres" in refused, refused[-60:])

        print("\n-- two kinds of fit on one structure, and which one answers --\n")

        both = draped_at(tmp, "draped2.gpkg", angles=True)

        text, mixed = transcript_of(both, Mapping(
            layer="tracce", ident_field="code",
            dip_dir_field="immersione", dip_field="inclinazione",
            dem_path=str(relief),
        ))

        two = gstruct.loads(text).by_ident("VEE")

        check("a structure can carry both, and the derived one is written first",
              two is not None and len(two.fits) == 2
              and two.fits[0].attrs.get("from") == FROM_DEM
              and two.fits[1].attrs.get("from") == FROM,
              ", ".join(f.attrs.get("from", "?") for f in two.fits) if two else "none")

        # Which matters because `attitude_at` takes the *first* fit that covers a
        # progressive where `span_at` takes the *last* span. So the specific
        # statement goes first among fits and last among spans, and having it the
        # wrong way round is not an error anywhere -- it is the wrong plane.
        on_the_bend = two.attitude_at(APEX_S) if two else (None, "")
        away = two.attitude_at(10.0) if two else (None, "")

        check("so the stretch read off the ground answers on it, and the column elsewhere",
              on_the_bend[0] is not None
              and abs(on_the_bend[0].dip - RELIEF_DIP) <= 2.0
              and away[0] is not None and away[0].dip == 70.0,
              f"at the bend {on_the_bend[0]}, away {away[0]}")

        check("the point layer's two angle columns are both named or neither",
              dialog.refusal(Mapping(points_path="x", points_dip_dir_field="a"))
              is not None
              and dialog.refusal(Mapping(points_path="x", points_dip_dir_field="a",
                                         points_dip_field="b")) is None)

        # A step that is most of the window makes a held stretch one window wide
        # and its extent the step, which is the thing `_reach` exists to stop
        # being the normal case rather than the corner.
        check("and a step that barely overlaps its window is refused",
              dialog.refusal(Mapping(fit_step=200.0, fit_fallback=250.0))
              is not None
              and dialog.refusal(Mapping(fit_step=25.0, fit_fallback=250.0)) is None)

        print("\n-- and the file opens in the tool it was written for --\n")

        from gsurf.session import Session
        from gsurf.tools import editor

        written = Path(tmp) / "imported.gstruct"
        written.write_text(transcript, encoding="utf-8")

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
