"""
A `.gstruct` out to a GeoPackage, edited as QGIS would edit it, and read back.

`gsurf/exports.py` claims to be lossless. That claim is only worth the thing
that checks it, so this is the thing: the real corpus goes out, comes back, and
the two texts are compared byte for byte -- not record counts, which agree
happily while a plane loses half a degree and a comment disappears.

What is under test:

  - **The round trip is byte-identical on the real files.** `curation.gstruct`
    with its ten comments, `montealpi_01.gstruct` with its 393 traces and 13
    fits, `merid_faults.gstruct` with its 791 spans. The comparison is against
    what the format's own writer makes of the original, because `dumps` is
    canonical and a line it would have rewritten anyway is not a line this
    export broke -- and separately against the bytes on disk, which is the
    stronger claim and the one the first two files actually meet.

  - **What the corpus has none of.** No file in the AOI carries a `lineation`,
    an unlabelled structure, a planeless record or a quoted `#`. Those go
    through a fixture instead, because a path with no test on it is a path that
    works until somebody writes the first one.

  - **The five edits a geologist actually makes.** Retype a dip, drag an anchor,
    reshape a trace, add a row, delete a row. Each has to arrive, and the ones
    that move geometry have to re-project the records anchored to it -- `s` is
    derived, and the whole reason the trace is editable is that it re-derives.

  - **A row added by hand does not assert itself over the whole trace.** A span
    drawn in QGIS leaves `open_start` NULL, and NULL read as a boolean is true:
    the naive reading throws away the geometry just drawn and spreads the value
    along the entire structure. This is the regression most worth nailing down,
    because it is silent and it changes an answer.

  - **The one knowing loss is bounded and reported.** A comment naming a vertex
    does not survive a reshape of the path it names -- and must not be slid onto
    a neighbouring vertex instead, which is what an index-only check allows.

  - **The layers are consumable by qgSurf.** The attitude layers come out as
    points carrying `dip_dir` and `dip` in their own columns, which is what
    StereoplotTool and GeoProfiler ask a layer for.

    QT_QPA_PLATFORM=offscreen python checks/check_exports.py
"""

import os
import sys
import tempfile
from pathlib import Path

REPO = Path(os.environ.get("GSURF_REPO", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(REPO))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gstruct as gs
from gsurf import exports

AOI = Path("/home/mauro/Documenti/Ricerca/AppenninoMeridionale/gstruct")

FAILURES = []


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}{'   ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(label)


def first_difference(a, b):
    """The first line that differs, which is what a byte-identity failure owes."""
    one, two = a.splitlines(), b.splitlines()
    for ndx, (x, y) in enumerate(zip(one, two)):
        if x != y:
            return f"line {ndx + 1}: {x!r} -> {y!r}"
    if len(one) != len(two):
        return f"{len(one)} lines -> {len(two)}"
    return "identical"


# ---------------------------------------------------------------- the real files

def check_corpus(tmp):
    print("\n-- the corpus, byte for byte --")

    if not AOI.is_dir():
        print(f"SKIP  {AOI} is not here")
        return

    for name in ("curation", "montealpi_01", "merid_faults"):
        src = AOI / f"{name}.gstruct"
        if not src.exists():
            print(f"SKIP  {name}.gstruct is not here")
            continue

        with open(src, encoding="utf-8", newline="") as fh:
            disk = fh.read()

        ok, canonical, after, report = exports.round_trip(
            str(src), str(Path(tmp) / f"{name}.gpkg"))

        check(f"{name}: the round trip is the canonical text", ok,
              "" if ok else first_difference(canonical, after))
        check(f"{name}: nothing was reshaped or dropped on the way",
              not report.reshaped and not report.revertexed,
              f"reshaped {len(report.reshaped)}, revertexed {len(report.revertexed)}")

        # The bytes on disk are a stronger claim, and only the files that are
        # already fixed points of `dumps` meet it. `merid_faults` is declared
        # 0.1 and carries 172 `kind unknown` lines that the writer does not
        # re-emit -- a difference that predates this export and belongs to the
        # decision still parked on that file, so it is stated rather than
        # asserted either way.
        if name == "merid_faults":
            check(f"{name}: differs from disk only where `dumps` already did",
                  after == canonical and disk != canonical,
                  "the 172 `kind unknown` lines, as before")
        else:
            check(f"{name}: identical to the bytes on disk", after == disk,
                  "" if after == disk else first_difference(disk, after))

    # The counts are not the proof, but a round trip that silently exported
    # nothing would pass a byte comparison of two empty files.
    src = AOI / "montealpi_01.gstruct"
    if src.exists():
        ds = gs.load(str(src))
        gpkg = str(Path(tmp) / "counts.gpkg")
        written = exports.to_geopackage(ds, gpkg, source_text=src.read_text())
        check("montealpi_01: every record reached a row",
              written.counts[exports.TRACES] == len(ds.structures)
              and written.counts[exports.FITS] == sum(len(s.fits) for s in ds.structures)
              and written.counts[exports.SPANS] == sum(len(s.spans) for s in ds.structures)
              and written.counts[exports.OBSERVATIONS] == len(ds.observations),
              str(written.counts))

        text, digest = exports.source_of(gpkg)
        check("montealpi_01: the source text travelled with its hash",
              text == src.read_text() and digest is not None and len(digest) == 64)


# ------------------------------------------------------- what the corpus lacks

# A lineation, a structure with no label token, a planeless attitude, a quoted
# `#`, a comment in every position, and an open-ended span: none of these is in
# any file in the AOI.
FIXTURE = '''gstruct 0.2
crs EPSG:25833
note "una prova"
# la testa del file

structure L0001 ""
  kind fault
  span certainty * * certain
  span use @100.00,200.00 * accepted reason="il foglio #534 non e' mai uscito"
  attitude @105.00,205.00 plane 231.4/60.0 src=field  # in coda alla riga
  # e questo sta sopra la lineazione
  lineation @110.00,210.00 45.5 12.0 sense=normal
  fit plane * @150.00,250.00 108/7 verdict=ok
  path 3
    100.00 200.00
    # nomina il secondo vertice
    150.00 250.00
    200.00 300.00

structure L0002
  kind thrust
  attitude @310.00,410.00
  path 2
    300.00 400.00
    400.00 500.00

observation S9 @700.00,800.00 plane 90/30 note="libera"
# la coda del file
'''


def check_fixture(tmp):
    print("\n-- the records no real file in the AOI has --")

    ds = gs.loads(FIXTURE)
    gpkg = str(Path(tmp) / "fixture.gpkg")
    exports.to_geopackage(ds, gpkg, source_text=FIXTURE)
    back, report = exports.from_geopackage(gpkg)

    after, canonical = gs.dumps(back), gs.dumps(ds)
    check("the fixture round-trips to the canonical text", after == canonical,
          "" if after == canonical else first_difference(canonical, after))

    # Canonical against canonical cannot see a text that neither side can read:
    # until 04.10.2026 `gstruct.dumps` wrote the planeless attitude below as
    # `plane None`, which its own parser refuses, and the assertion above passed
    # all along -- both texts were malformed identically. What an import promises
    # is not that two writers agree, it is that the file opens next session.
    try:
        reopened, refused = gs.loads(after), ""
    except ValueError as err:
        reopened, refused = None, f"{type(err).__name__}: {err}"
    check("and what it wrote is a text that opens", reopened is not None, refused)

    one = back.structures[0]
    check("the lineation kept its trend and plunge",
          one.lineations and one.lineations[0].trend == 45.5
          and one.lineations[0].plunge == 12.0)
    check("a `#` inside quotes stayed a value",
          one.spans[1].attrs.get("reason") == "il foglio #534 non e' mai uscito")
    check("the comment above the lineation came back on the lineation",
          one.lineations[0].lead == ["  # e questo sta sopra la lineazione"],
          repr(one.lineations[0].lead))
    check("the inline comment came back inline",
          one.attitudes[0].tail.strip() == "# in coda alla riga",
          repr(one.attitudes[0].tail))
    check("the open span end stayed open, not a coordinate",
          one.spans[1].start is not None and one.spans[1].end is None)
    check("the vertex comment came back on its vertex",
          one.line_lead.get("v1") == ["    # nomina il secondo vertice"],
          repr(one.line_lead.get("v1")))

    two = back.structures[1]
    check("the structure with no label token still has none", two.label is None,
          repr(two.label))
    check("the first structure's empty label token is still empty, not absent",
          one.label == "", repr(one.label))
    check("a planeless attitude survived as a row", two.attitudes
          and two.attitudes[0].plane is None
          and two.attitudes[0].anchor == (310.0, 410.0))
    check("the file head, the trailer and the note are all back",
          back.meta.get("note") == "una prova"
          and back.line_lead.get("structure") is None
          and any("coda del file" in line for line in back.trailer),
          repr(back.trailer))
    check("nothing was reported lost", not report.revertexed and not report.reshaped)


# -------------------------------------------------------------- the five edits

def check_edits(tmp):
    print("\n-- the five edits a geologist makes in QGIS --")

    import pandas as pd
    import geopandas as gpd
    import pyogrio
    from shapely.geometry import LineString, Point

    ds = gs.loads(FIXTURE)
    gpkg = str(Path(tmp) / "edited.gpkg")
    exports.to_geopackage(ds, gpkg, source_text=FIXTURE)

    before = gs.loads(FIXTURE).structures[0]

    # 1 and 2: a dip retyped, an anchor dragged.
    frame = pyogrio.read_dataframe(gpkg, layer=exports.ATTITUDES)
    frame.loc[0, "dip"] = 72.5
    frame.loc[0, "geometry"] = Point(130.00, 230.00)
    pyogrio.write_dataframe(frame, gpkg, layer=exports.ATTITUDES,
                            driver="GPKG", append=False)

    # 3: a trace extended past its old end.
    frame = pyogrio.read_dataframe(gpkg, layer=exports.TRACES)
    frame.loc[0, "geometry"] = LineString(
        [(100, 200), (150, 250), (200, 300), (260, 360)])
    pyogrio.write_dataframe(frame, gpkg, layer=exports.TRACES,
                            driver="GPKG", append=False)

    # 4 and 5: a span drawn by hand, and one deleted.
    frame = pyogrio.read_dataframe(gpkg, layer=exports.SPANS)
    drawn = frame.iloc[[0]].copy()
    drawn["axis"], drawn["value"], drawn["seq"] = "exposure", "covered", 9
    # Exactly what QGIS leaves in the columns the geologist did not fill in.
    for column in ("open_start", "open_end", "attrs", "comment", "lead_json",
                   "s0_m", "s1_m"):
        drawn[column] = None
    drawn["geometry"] = [LineString([(150, 250), (200, 300)])]
    frame = gpd.GeoDataFrame(pd.concat([frame, drawn], ignore_index=True),
                             crs=frame.crs)
    frame = frame[frame["axis"] != "certainty"]
    pyogrio.write_dataframe(frame, gpkg, layer=exports.SPANS,
                            driver="GPKG", append=False)

    back, report = exports.from_geopackage(gpkg)
    one = back.structures[0]

    check("1. the retyped dip arrived",
          one.attitudes[0].plane.dip == 72.5
          and one.attitudes[0].plane.dip_dir == 231.4,
          str(one.attitudes[0].plane))
    check("2. the dragged anchor arrived",
          one.attitudes[0].anchor == (130.0, 230.0), str(one.attitudes[0].anchor))
    check("3. the reshaped trace arrived", len(one.path) == 4
          and one.path[-1] == (260.0, 360.0))
    check("4. the span drawn by hand arrived",
          any(sp.axis == "exposure" and sp.value == "covered" for sp in one.spans))
    check("5. the deleted span is gone",
          not any(sp.axis == "certainty" for sp in one.spans))

    drawn_span = next(sp for sp in one.spans if sp.axis == "exposure")
    check("the hand-drawn span kept the ground it was drawn on, "
          "instead of claiming the whole trace",
          drawn_span.start == (150.0, 250.0) and drawn_span.end == (200.0, 300.0),
          f"{drawn_span.start} .. {drawn_span.end}")

    # The derived numbers re-derive, which is the point of editable geometry.
    check("the trace got longer and the open span end followed it",
          one.length > before.length and abs(one.spans[0].s1 - one.length) < 1e-6,
          f"{before.length:.2f} m -> {one.length:.2f} m")
    check("the dragged anchor re-projected to a new progressive",
          abs(one.attitudes[0].s - before.attitudes[0].s) > 1.0,
          f"s {before.attitudes[0].s:.2f} -> {one.attitudes[0].s:.2f}")
    check("the reshaped trace is named in the report",
          report.reshaped == ["L0001"], str(report.reshaped))


# -------------------------------------------------------- the one knowing loss

def check_vertex_comments(tmp):
    print("\n-- the comment on a vertex, and the reshape it cannot survive --")

    import pyogrio
    from shapely.geometry import LineString

    ds = gs.loads(FIXTURE)
    gpkg = str(Path(tmp) / "vertices.gpkg")
    exports.to_geopackage(ds, gpkg, source_text=FIXTURE)

    back, report = exports.from_geopackage(gpkg)
    check("untouched, the vertex comment is kept and nothing is reported",
          back.structures[0].line_lead.get("v1") and not report.revertexed)

    # A vertex inserted BEFORE the commented one. The index still exists, so an
    # index-only check keeps the comment -- and keeps it on the wrong vertex.
    frame = pyogrio.read_dataframe(gpkg, layer=exports.TRACES)
    frame.loc[0, "geometry"] = LineString(
        [(100, 200), (120, 220), (150, 250), (200, 300)])
    pyogrio.write_dataframe(frame, gpkg, layer=exports.TRACES,
                            driver="GPKG", append=False)

    back, report = exports.from_geopackage(gpkg)
    moved = back.structures[0]
    check("a vertex inserted before it drops the comment rather than sliding it",
          not any(k.startswith("v") for k in moved.line_lead),
          str(moved.line_lead))
    check("and the structure that paid is named",
          report.revertexed == ["L0001"] and report.reshaped == ["L0001"])

    # The fingerprint is taken at the precision the format writes, so a layer
    # opened and saved again is not a reshape.
    exports.to_geopackage(gs.loads(FIXTURE), gpkg, source_text=FIXTURE)
    frame = pyogrio.read_dataframe(gpkg, layer=exports.TRACES)
    pyogrio.write_dataframe(frame, gpkg, layer=exports.TRACES,
                            driver="GPKG", append=False)
    back, report = exports.from_geopackage(gpkg)
    check("re-saving a layer untouched is not a reshape",
          not report.reshaped and back.structures[0].line_lead.get("v1"),
          str(report.reshaped))


# --------------------------------------------- what a hand in the table writes

def check_typed_in(tmp):
    """The two columns that punish being filled in the obvious way.

    Both of these were found by driving the real handlers on `montealpi_01`
    rather than by reading: the corpus could not have caught either, because it
    carries no inline comment at all -- every `#` in it is a whole line above a
    record -- and it has no reason to rename anything.
    """

    print("\n-- the columns a hand typing into them could break --")

    import pandas as pd
    import geopandas as gpd
    import pyogrio
    from shapely.geometry import Point

    ds = gs.loads(FIXTURE)
    gpkg = str(Path(tmp) / "typed.gpkg")
    exports.to_geopackage(ds, gpkg, source_text=FIXTURE)

    # Two blank `comment` cells filled in the obvious way, and the difference
    # between them is the whole severity of this: a tail is appended to the end
    # of the line, so what it welds itself to depends on what the record ends
    # with. The added attitude carries no attributes, so its line ends in a
    # number -- the case found on `montealpi_01`; the lineation keeps
    # `sense=normal`, so there the weld lands inside an attribute value instead.
    frame = pyogrio.read_dataframe(gpkg, layer=exports.ATTITUDES)
    marked = frame.loc[0, "comment"]          # `  # in coda alla riga`, untouched

    # A measurement added in QGIS, with a comment typed beside it.
    added = frame.iloc[[0]].copy()
    added["seq"], added["attrs"], added["lead_json"] = 9, None, None
    added["comment"] = "misurato col compasso"
    added["geometry"] = [Point(115.00, 215.00)]
    frame = gpd.GeoDataFrame(pd.concat([frame, added], ignore_index=True),
                             crs=frame.crs)
    pyogrio.write_dataframe(frame, gpkg, layer=exports.ATTITUDES,
                            driver="GPKG", append=False)

    frame = pyogrio.read_dataframe(gpkg, layer=exports.LINEATIONS)
    frame.loc[0, "comment"] = "anche questo"
    pyogrio.write_dataframe(frame, gpkg, layer=exports.LINEATIONS,
                            driver="GPKG", append=False)

    back, report = exports.from_geopackage(gpkg)
    rebuilt = gs.dumps(back)
    typed = next((line for line in rebuilt.splitlines() if "compasso" in line), "")

    # The loud half, and the assertion that matters is not how the line looks:
    # it is that the file opens. Welded to the record -- `plane 231.4/60.0misurato
    # col compasso` -- it was written, reported as imported and remembered in the
    # `traces` slot, then refused a session later with
    # `could not convert string to float: '60.0misurato'`.
    opened = None
    try:
        gs.loads(rebuilt)
    except Exception as err:
        opened = f"{type(err).__name__}: {err}"

    check("a comment typed where the line ends in a number still opens",
          opened is None, opened or "")
    check("and it is a comment rather than welded to the number",
          "# misurato col compasso" in typed and "60.0misurato" not in typed,
          typed or "not found")

    # The quiet half, and the worse of the two: this line parses either way, so
    # before the fix the attribute silently swallowed the sentence --
    # `sense=normalanche questo` -- and the comment stopped being a comment with
    # nothing anywhere saying so. Asserted on the line and not on `back`, which
    # is the dataset *before* `dumps` and so cannot show a weld at all: written
    # that way first, this check read `sense` as "normal" from the old reader too.
    lineated = next((line for line in rebuilt.splitlines()
                     if "anche questo" in line), "")
    check("where the line ends in an attribute, the value is not eaten",
          "sense=normal  # anche questo" in lineated, lineated or "not found")

    check("a comment that already had its `#` is untouched, spacing included",
          any(at.tail == marked
              for st in back.structures for at in st.attitudes if at.tail),
          repr(marked))

    # The quietest of the three, and it needs no `#` to go wrong. A text cell in
    # QGIS takes newlines; a tail is one line. Left as typed, the second line
    # arrives in the file as a line of its own and is read as something else
    # entirely -- `seconda riga` came back as `structure X seconda=riga`, an
    # attribute invented out of prose, with nothing raised anywhere.
    frame = pyogrio.read_dataframe(gpkg, layer=exports.ATTITUDES)
    frame.loc[0, "comment"] = "due righe\nseconda riga"
    pyogrio.write_dataframe(frame, gpkg, layer=exports.ATTITUDES,
                            driver="GPKG", append=False)

    twice, _ = exports.from_geopackage(gpkg)
    text = gs.dumps(twice)

    # Guarded, or a regression in any of the cases above arrives here as a
    # traceback that takes the rest of the section with it instead of a failure.
    try:
        settled = gs.dumps(gs.loads(text))
    except Exception as err:
        settled = f"{type(err).__name__}: {err}"

    check("a comment typed over two lines stays one comment",
          "# due righe seconda riga" in text
          and not any(line.strip() == "seconda riga" for line in text.splitlines()),
          next((line for line in text.splitlines() if "righe" in line), "not found"))
    # Re-reading its own output is the assertion that catches an invented
    # attribute: the stray line only becomes `seconda=riga` on the way back in.
    check("and the text it writes means the same thing when read again",
          settled == text and "seconda=riga" not in settled,
          next((line for line in settled.splitlines()
                if "structure" in line), "not found"))

    # And the rename, which is the one edit the join key cannot survive.
    exports.to_geopackage(gs.loads(FIXTURE), gpkg, source_text=FIXTURE)
    frame = pyogrio.read_dataframe(gpkg, layer=exports.TRACES)
    frame.loc[0, "ident"] = "RENAMED"
    pyogrio.write_dataframe(frame, gpkg, layer=exports.TRACES,
                            driver="GPKG", append=False)

    rows = {name: len(pyogrio.read_dataframe(gpkg, layer=name))
            for name in exports.LAYERS}
    back, report = exports.from_geopackage(gpkg)
    kept = sum(len(st.attitudes) for st in back.structures)

    check("renaming a trace orphans its records, and that is said",
          any("unknown structure" in note for note in report.notes),
          f"{len(report.notes)} note(s)")
    # The precondition, so this cannot pass by the two numbers both being wrong.
    check("the layer still holds the rows, so counting them would miss the loss",
          rows[exports.ATTITUDES] > kept,
          f"{rows[exports.ATTITUDES]} row(s) in the layer, {kept} rebuilt")
    check("so the count reports what was rebuilt, not what was read",
          report.counts[exports.ATTITUDES] == kept
          and report.counts[exports.TRACES] == len(back.structures),
          f"{report.counts[exports.ATTITUDES]} reported, {kept} rebuilt")


# ------------------------------------------------------- what qgSurf asks for

def check_consumable(tmp):
    print("\n-- the shape qgSurf's tools ask a layer for --")

    import pyogrio

    ds = gs.loads(FIXTURE)
    gpkg = str(Path(tmp) / "consumable.gpkg")
    exports.to_geopackage(ds, gpkg, source_text=FIXTURE)

    listed = {info[0] if isinstance(info, (list, tuple)) else info
              for info in pyogrio.list_layers(gpkg)[:, 0]}
    check("every layer is there, including the empty ones",
          set(exports.LAYERS) <= listed,
          str(sorted(set(exports.LAYERS) - listed)) if not set(exports.LAYERS) <= listed
          else f"{len(listed)} layers")

    frame = pyogrio.read_dataframe(gpkg, layer=exports.ATTITUDES)
    check("attitudes are points with dip_dir and dip in their own columns",
          {"dip_dir", "dip"} <= set(frame.columns)
          and frame.geometry.geom_type.eq("Point").all(),
          str(list(frame.columns)))

    fits = pyogrio.read_dataframe(gpkg, layer=exports.FITS)
    check("a fit is drawn along the stretch it was computed on",
          fits.geometry.geom_type.eq("LineString").all()
          and {"dip_dir", "dip"} <= set(fits.columns))

    check("the CRS travelled onto the layers",
          frame.crs is not None and "25833" in str(frame.crs.to_string()),
          str(frame.crs))

    # The two plain tables are not content of the GeoPackage. GDAL lists them
    # anyway -- it lists every table in the file -- so the claim worth checking
    # is the declaration and not the listing: they are absent from
    # `gpkg_contents` and they carry no geometry.
    import sqlite3
    con = sqlite3.connect(gpkg)
    try:
        declared = {row[0] for row in con.execute(
            "SELECT table_name FROM gpkg_contents;")}
    finally:
        con.close()
    check("the header and the provenance are not declared as content",
          exports.DATASET_TABLE not in declared
          and exports.SOURCE_TABLE not in declared
          and declared == set(exports.LAYERS),
          str(sorted(declared)))

    aspatial = {name for name, kind in pyogrio.list_layers(gpkg) if kind is None}
    check("and GDAL sees them as tables with no geometry, which is what they are",
          aspatial == {exports.DATASET_TABLE, exports.SOURCE_TABLE},
          str(sorted(aspatial)))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        check_corpus(tmp)
        check_fixture(tmp)
        check_edits(tmp)
        check_vertex_comments(tmp)
        check_typed_in(tmp)
        check_consumable(tmp)

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: " + "; ".join(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
