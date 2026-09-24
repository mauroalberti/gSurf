"""
Does a `.gstruct` open as a source, and does a curation lay over what is open?

Two readings of one format. A file that carries paths is a trace layer like any
other; a file that carries none is a curation, and names structures that are
already in hand. Both end as `TraceRecord`s, which is the point -- nothing past
the reader is told which door the records came in by.

What is actually measured here is the anchor. A GeoPackage cannot hold `@x,y`,
so the progressive is stored instead, and a progressive is a reading off a ruler
that the projection and the digitising both move. So the file is opened twice in
two projections, and once more with its trace redrawn, and each time the
question is the same: did the measurement stay on the ground where it was taken.

The synthetic dataset is built so the answers are exact. Its first trace runs due
east for a kilometre with a vertex every hundred metres, and the attitude on it
is anchored fifty metres off the line at six hundred thousand three hundred --
so the progressive is 300 m and the offset is 50, to the millimetre, and any
number that comes back instead is a number to explain.

    python check_gstruct.py
"""

import os
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
project "a synthetic dataset, built so the answers are known"

structure F001 "Alpha" fid=1 set="the one with numbers on it"
  kind fault
  span certainty * * certain raw=certain
  span exposure @600000.00,4420000.00 @600400.00,4420000.00 covered src=survey
  attitude @600300.00,4420050.00 plane 90/30 station=S1 src=field
  fit plane * * 100/40 from=trace-dem nvert=11 snr=25 verdict=indipendente-dal-versante
  path 11
{alpha}

structure F002 "Beta" fid=2
  kind thrust
  span certainty * * inferred
  span exposure * * unknown reason=assente-in-sorgente
  path 3
    602000.00 4420000.00
    602000.00 4420300.00
    602000.00 4420600.00

structure F003 "Gamma" fid=3
  kind fault
  span use * * rejected src=curatela reason="ricalca una rottura di pendio"
  attitude @603000.00,4420000.00 plane 270/60 station=S2 src=field
  path 2
    603000.00 4420000.00
    603400.00 4420000.00

structure F004 "Delta" fid=4
  kind fault
  fit plane * * 12/88 from=trace-dem verdict=traccia-rettilinea plan=2
  path 2
    604000.00 4420000.00
    604500.00 4420000.00

observation S9 @605000.00,4425000.00 plane 55/70 station=S9 src=field
    unattached=oltre-soglia nearest=F001 distance=5099.0
"""

# A curation: six claims, no geometry at all. Two of them are about where along
# the trace they hold, which is the case that could not even be parsed before --
# `s` is derived by projection, and a file with no path has nothing to project
# onto until it is laid over one that has.
CURATION = """gstruct 0.2
crs EPSG:25833
note "Si applica sopra il sintetico. Ogni riga e' un'asserzione umana."

structure F001
  span exposure @600250.00,4420000.00 @600350.00,4420000.00 exposed src=field
  span exposure @600600.00,4420000.00 @601000.00,4420000.00 covered src=field
  span use * * rejected src=gsurf reason="non va in questa sezione"
  span vergence * * east src=field
  attitude @600800.00,4420000.00 plane 120/45 station=S7 src=field

structure F999
  span exposure * * exposed src=field
"""


def check(label, condition, detail=""):
    print(f"{'PASS' if condition else 'FAIL'}  {label}{'   ' + detail if detail else ''}")
    if not condition:
        FAILURES.append(label)


def written(directory, name, text):
    path = Path(directory) / name
    path.write_text(text, encoding="utf8")
    return path


def alpha_path(step=100.0, keep=1):
    """The first trace's vertices, optionally redrawn keeping one in `keep`."""

    vertices = [(X0 + n * step, Y0) for n in range(11)][::keep]

    if vertices[-1] != (X0 + 1000.0, Y0):
        vertices.append((X0 + 1000.0, Y0))

    return vertices


def as_text(vertices):
    return "\n".join(f"    {x:.2f} {y:.2f}" for x, y in vertices)


def source_text(vertices=None):
    vertices = alpha_path() if vertices is None else vertices

    return SOURCE.format(alpha=as_text(vertices)).replace(
        "  path 11", f"  path {len(vertices)}"
    )


def by_ident(records, ident, fitted=None):
    """The records of one structure, optionally only the fitted or the measured."""

    from gsurf.curation import ident_of

    return [
        record for record in records
        if ident_of(record) == ident
        and (fitted is None or bool(record.attrs.get("fitted")) == fitted)
    ]


def main():
    try:
        import gstruct  # noqa: F401
    except ImportError:
        print("gstruct is not installed, and this check is about reading it.\n"
              "  pip install -e <gstruct repo>")
        return 1

    from pyproj import CRS, Transformer

    from gsurf.attitudes import TraceAttitudeSource
    from gsurf.curation import apply_to, is_gstruct

    work = Path(tempfile.mkdtemp())
    native = CRS.from_epsg(25833)

    print("-- a file with paths, opened as a trace layer --\n")

    check("a .gstruct is recognised as one to read here",
          is_gstruct(work / "x.gstruct") and not is_gstruct(work / "x.gpkg"))

    path = written(work, "source.gstruct", source_text())
    source = TraceAttitudeSource(path, native)

    check("it opens", source.problem is None and source.is_loaded,
          source.problem or source.summary())

    # Four structures: one carrying two planes, one carrying none, two carrying
    # one each. The bare one is a record too -- it is a mapped contact nobody
    # has measured, which is what the fit exists for.
    check("one record per plane, and one for the trace that has none",
          len(source.traces) == 5 and source.unread == 1,
          f"{len(source.traces)} records, {source.unread} with no plane")

    check("the category is the kind, not the ident",
          sorted({record.category for record in source.traces}) == ["fault", "thrust"],
          str(sorted({record.category for record in source.traces})))

    measured = by_ident(source.traces, "F001", fitted=False)[0]
    fitted = by_ident(source.traces, "F001", fitted=True)[0]

    print("\n-- the anchor, which is the whole reason for the format --\n")

    check("an anchor arrives as a progressive along the trace",
          abs(measured.anchor - 300.0) < 1e-6, f"{measured.anchor:.6f} m of 1000")

    check("and its distance from the trace comes with it",
          abs(float(measured.attrs["off_m"]) - 50.0) < 1e-6,
          f"{measured.attrs['off_m']} m off the line")

    check("a measurement gets no reach: that decision is not in the file",
          measured.span is None)

    check("a fit gets one, because it was computed over an interval",
          fitted.span is not None and abs(fitted.span[1] - 1000.0) < 1e-6,
          f"{fitted.span[0]:.0f}-{fitted.span[1]:.0f} m")

    check("and is marked as derived, so the curation writer will not quote it",
          fitted.attrs.get("fitted") is True and fitted.attrs.get("src") == "trace-dem")

    # The two records of one structure sit at different places along it, and the
    # span covers only the first 400 m. Read at one place for the whole
    # structure, both would say the same thing and one of them would be wrong.
    check("an axis is read at each record's own place",
          measured.attrs["exposure"] == "covered"
          and fitted.attrs["exposure"] == "unknown",
          f"measured at 300 m: {measured.attrs['exposure']}, "
          f"fit at 500 m: {fitted.attrs['exposure']}")

    check("and the axis that says nothing says so",
          measured.attrs["certainty"] == "certain"
          and by_ident(source.traces, "F002")[0].attrs["exposure"] == "unknown")

    print("\n-- what arrives switched off --\n")

    rejected = by_ident(source.traces, "F003")[0]

    check("a stretch the curator rejected arrives held out, with the reason on it",
          rejected.enabled is False and rejected.attrs.get("use") == "rejected",
          rejected.attrs.get("use", "nothing said"))

    straight = by_ident(source.traces, "F004")[0]

    check("so does a fit whose trace was too straight to constrain the dip",
          straight.enabled is False and straight.plane is not None,
          f"verdict {straight.attrs.get('verdict')}, and the plane is still there "
          f"to be argued with")

    check("the section is offered the planes that are left, and no others",
          list(source.records) == ["fault"] and len(source.records["fault"]) == 2,
          str({name: len(items) for name, items in source.records.items()}))

    print("\n-- what has no trace to sit on --\n")

    check("an unattached measurement is carried, not dropped",
          len(source.loose) == 1 and source.loose[0].ident == "S9",
          str(source.loose[0].ident if source.loose else "none"))

    check("with its plane converted like every other",
          abs(source.loose[0].plane.dipazim - 55.0) < 1e-9
          and abs(source.loose[0].plane.dipang - 70.0) < 1e-9)

    check("and the summary says it out loud",
          "no trace to sit on" in source.summary(), source.summary())

    print("\n-- the same file in another projection --\n")

    # UTM zone 32 for data that belongs in zone 33: a real projection, badly
    # chosen, which is the interesting case. The ground does not move and the
    # ruler does.
    other = CRS.from_epsg(32632)
    abroad = TraceAttitudeSource(path, other)
    there = by_ident(abroad.traces, "F001", fitted=False)[0]

    back = Transformer.from_crs(other, native, always_xy=True)
    x, y = back.transform(*there.anchor_point()[:2])

    check("the anchor lands on the same ground",
          abs(x - 600300.0) < 0.05 and abs(y - 4420000.0) < 0.05,
          f"({x:.3f}, {y:.3f}) against (600300.000, 4420000.000)")

    check("while the progressive moves with the projection, as a measured length must",
          abs(there.anchor - measured.anchor) > 0.5,
          f"{there.anchor:.2f} m in zone 32 against {measured.anchor:.2f} in zone 33 "
          f"-- which is why a progressive is not a place")

    print("\n-- the same file, redigitised --\n")

    coarse = written(work, "coarse.gstruct", source_text(alpha_path(keep=5)))
    redrawn = by_ident(TraceAttitudeSource(coarse, native).traces, "F001", fitted=False)[0]

    moved = (
        (redrawn.anchor_point()[0] - measured.anchor_point()[0]) ** 2
        + (redrawn.anchor_point()[1] - measured.anchor_point()[1]) ** 2
    ) ** 0.5

    check("keeping one vertex in five, the measurement stays where it was taken",
          moved < 0.01, f"moved {moved:.4f} m")

    print("\n-- the way in through the dialog --\n")

    from PyQt6 import QtWidgets

    from gsurf.session import Session
    from gsurf.sources import TracePicker
    from gsurf.vectors import VectorSource

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841

    check("the file offers itself where lines are wanted, and nowhere else",
          VectorSource.candidate_layers(path, "lines") == ["structures"]
          and VectorSource.candidate_layers(path, "polygons") == [])

    picker = TracePicker()
    filled = picker.set_path(str(path))
    spec = picker.value()

    # Neither angle column is named, and that is the finished answer rather than
    # half of one: the format writes `90/30` and says in its own grammar which
    # number is which, so there is nothing here for a dialog to ask.
    check("a picker fills on it with no column named",
          filled and picker.is_filled and spec is not None
          and spec["dip_dir_field"] is None and spec["dip_field"] is None,
          str(spec))

    framed = Session.open(frame_layers=[spec])

    check("and a session frames itself on it with no DEM",
          framed.epsg == 25833
          and abs(framed.bounds[0] - X0) < 1.0
          and abs(framed.bounds[2] - 605000.0) < 1.0,
          f"EPSG:{framed.epsg}, {framed.summary()}")

    print("\n-- a curation, laid over what is open --\n")

    curation = written(work, "curation.gstruct", CURATION)

    from gsurf.curation import read

    records, applied = apply_to(source.traces, read(curation))

    print(f"   {applied.summary()}\n")

    check("a structure the records never heard of is named, not swallowed",
          applied.missing == ["F999"] and applied.matched == 1, str(applied.missing))

    # `vergence` is a plausible proposal and not a typo, which is the case worth
    # covering: `value_at` would read it perfectly well, and acting on it
    # silently is how an invented axis becomes real without anybody deciding it
    # should. That is what `use` itself was until it went into the format.
    check("an axis this tool does not act on is reported instead of acted on",
          applied.ignored == {"vergence": 1}, str(applied.ignored))

    check("an interval that covers no record of its structure is counted",
          applied.nowhere == 1,
          "the 600-1000 m span, on a structure whose records sit at 300 and 500")

    check("and the one that does cover a record changes it",
          measured.attrs["exposure"] == "exposed"
          and fitted.attrs["exposure"] == "unknown",
          f"measured at 300 m: {measured.attrs['exposure']}, "
          f"fit at 500 m: {fitted.attrs['exposure']}")

    check("a decision about the whole structure reaches all its records",
          measured.enabled is False and fitted.enabled is False)

    added = [record for record in records if record.attrs.get("station") == "S7"]

    check("a plane in a curation adds a record", len(added) == 1
          and len(records) == len(source.traces) + 1, f"{len(records)} records now")

    check("it does not overwrite the one that was there",
          abs(measured.plane.dipazim - 90.0) < 1e-9
          and abs(added[0].plane.dipazim - 120.0) < 1e-9,
          f"survey {measured.plane.dipazim:.0f}/{measured.plane.dipang:.0f}, "
          f"curation {added[0].plane.dipazim:.0f}/{added[0].plane.dipang:.0f}")

    # The file it came from carries no geometry at all, so this number can only
    # have been computed by projecting `@600800,4420000` onto the trace the
    # records brought with them.
    check("and its anchor was resolved against the records' own trace",
          abs(added[0].anchor - 800.0) < 1e-6, f"{added[0].anchor:.6f} m")

    check("it inherits the structure's identity and nothing of the measurement beside it",
          added[0].attrs.get("ident") == "F001"
          and "nvert" not in added[0].attrs and added[0].attrs.get("station") == "S7")

    print("\n-- and the same thing from the panel --\n")

    from gsurf.tools.profiles import TracePanel

    panel = TracePanel(TraceAttitudeSource(path, native))
    rows = panel.table.rowCount()
    from_panel = panel.apply_curation(str(curation))

    check("the button reads the file and the table grows by what it added",
          panel.table.rowCount() == rows + len(from_panel.added)
          and from_panel.assertions == applied.assertions,
          f"{rows} rows -> {panel.table.rowCount()}")

    # `Back to the layer` is about undoing a fit. An assertion is not a fit, and
    # pressing that button must not take one away.
    panel.restore_surveyed()

    check("and what it added survives going back to the layer",
          len(by_ident(panel.source.traces, "F001")) == 3,
          f"{len(panel.source.traces)} records after the undo")

    print("\n-- and back out, as a curation --\n")

    from gsurf.curation import curation_of
    from gsurf.curation import module as gstruct_module

    # A fresh reading, so that what is written out is a decision made here and
    # not the overlay from the block above still sitting on the records.
    again = TraceAttitudeSource(path, native)
    mine = by_ident(again.traces, "F001", fitted=False)[0]
    computed = by_ident(again.traces, "F001", fitted=True)[0]

    mine.span = (mine.anchor - 120.0, mine.anchor + 120.0)   # a reach, set by hand
    by_ident(again.traces, "F002")[0].enabled = False        # and a refusal

    text, report = curation_of(again.traces, again.half_span,
                               crs="EPSG:25833", source="source.gstruct")

    print(text.rstrip() + "\n")

    check("a refusal goes out on the axis the format defines, not an invented one",
          "span use" in text and "rejected" in text and "excluded" not in text,
          f"{report['refusals']} refusal(s)")

    check("a reach goes out as a fit, because that is what a plane over an "
          "interval is",
          "from=reach" in text and "span reach" not in text,
          f"{report['fits']} fit(s)")

    # A fit computed *here* goes out with `window=` and the rest of what it was
    # computed by -- which the old writer dropped, having declared every line in
    # the file a human assertion. That one needs the fitter and a DEM, so it is
    # checked where both are: `check_sections.py`, over Monte Alpi.

    check("structures are named by ident, not by category",
          "structure F001" in text and "structure fault" not in text)

    # F003 arrived switched off because the file says so, and F004 because its
    # trace is straight. Neither is a decision made here, and a curation that
    # signed them would be handing the geologist a refusal the gate had made.
    check("a refusal the file already accounts for is not restated as a new one",
          "F003" not in text and report["refusals"] == 1,
          f"{report['refusals']} refusal(s) written, of "
          f"{sum(1 for r in again.traces if not r.enabled)} records switched off")

    check("nor is a fit that arrived with the file",
          "from=trace-dem" not in text and report["fits"] == 1,
          "it is already in the thing this is laid over")

    # The real test of a writer: the parser it was written against.
    back = gstruct_module().loads(text)

    check("what comes out parses, and says the same thing",
          {s.ident for s in back.structures} == {"F001", "F002"}
          and len(back.by_ident("F001").fits) == 1,
          f"{len(back.structures)} structures, "
          f"{sum(len(s.fits) for s in back.structures)} fits, "
          f"{sum(len(s.spans) for s in back.structures)} spans")

    check("and a curation carries no geometry, not even an empty one",
          "path" not in text and "kind" not in text,
          "it names the structures; the path is in the file it is laid over")

    # Round trip, all the way: the refusal written here has to come back as the
    # same record switched off. A file that writes what it cannot read is a file
    # that looks like a record of a decision and is not one.
    third = TraceAttitudeSource(path, native)
    _, over = apply_to(third.traces, gstruct_module().loads(text))

    check("laid back over the records, the refusal comes home",
          by_ident(third.traces, "F002")[0].enabled is False
          and by_ident(third.traces, "F001", fitted=False)[0].enabled is True,
          over.summary())

    check("and the reach comes back as its own derived record, beside the "
          "measurement it was extended from",
          any(record.attrs.get("from") == "reach" for record in over.added),
          f"{len(over.added)} added, "
          f"{sorted({r.attrs.get('from', '?') for r in over.added})}")

    # The reason the writer holds back what it read: this file is laid over the
    # source, so anything it restated would arrive twice. One record added for
    # one decision made, and the file's own five untouched.
    check("and nothing the file already said arrives a second time",
          len(third.traces) + len(over.added) == len(again.traces) + 1,
          f"{len(again.traces)} records, {len(over.added)} added by the curation")

    # Nothing is guessed for a record the layer never named. The measurement
    # below is real and its decision is real, and both of them stay in the
    # window, which the box that writes the file has to say out loud.
    orphan = TraceAttitudeSource(path, native)

    for record in orphan.traces:
        record.attrs.pop("ident", None)
        record.attrs.pop("code", None)
        record.enabled = False

    empty, silent = curation_of(orphan.traces, orphan.half_span, crs="EPSG:25833")

    check("a record the layer gives no ident to is counted, not invented a name for",
          silent["structures"] == 0 and silent["unnamed"] > 0
          and "structure" not in empty,
          f"{silent['unnamed']} line(s) with nothing to address them to")

    print("\n-- the real thing --\n")

    faults, curated = AOI / "merid_faults.gstruct", AOI / "curation.gstruct"

    if not faults.exists() or not curated.exists():
        print(f"   not here: {AOI}")
    else:
        real = TraceAttitudeSource(faults, native)
        print(f"   {real.summary()}\n")

        check("the fault layer of the AOI opens",
              real.problem is None and len(real.traces) >= 393,
              f"{len(real.traces)} records over 393 structures")

        check("and carries the measurements that never attached",
              len(real.loose) > 0, f"{len(real.loose)} unattached")

        named = ("F0058", "F0074", "F0055", "F0385", "F0380")
        before = {
            ident: by_ident(real.traces, ident)[0].attrs.get("exposure")
            for ident in named
        }
        from gsurf.curation import ident_of

        # Counted by structure and not by record: those five are the sites that
        # were walked, so they carry a compass reading and a facet fit each, and
        # thirteen records answer to five idents.
        exposed = {
            ident_of(record) for record in real.traces
            if record.attrs.get("exposure") == "exposed"
        }

        # That file is already in this one. `export_geology.py` merges the
        # curation on the way out, so what is read back is the survey with five
        # human assertions in it -- which is what makes the next check worth
        # making: the same claim laid on twice has to leave the same answer.
        check("the curation is visible in the export, and only where it was made",
              set(before.values()) == {"exposed"}
              and exposed == set(named)
              and by_ident(real.traces, "F0000")[0].attrs["exposure"] == "unknown",
              f"{len(exposed)} structures exposed of {real.num_lines}, the rest unknown")

        _, over = apply_to(real.traces, read(curated))

        after = {
            ident: by_ident(real.traces, ident)[0].attrs.get("exposure")
            for ident in named
        }

        print(f"   {over.summary()}\n")

        check("the real curation matches every structure it names",
              over.matched == over.named and not over.missing,
              f"{over.matched} of {over.named}")

        # Append-only shadowing, from the other end: the last span covering a
        # place wins, so a claim that is already in force is not a second claim.
        # FORMAT.md licenses the exhumed-facet fit on `exposure=exposed`, and
        # the survey said `unknown` on all 393 traces because no field in it
        # could say otherwise -- so this axis is the whole of what that file is.
        check("and laying it on again says the same thing",
              after == before and over.assertions == len(named), str(after))

    print()

    if FAILURES:
        print(f"FAILED: {', '.join(FAILURES)}")
        return 1

    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
