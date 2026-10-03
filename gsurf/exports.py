"""
A `.gstruct` as a GeoPackage you can edit in QGIS, and read back as a `.gstruct`.

The export that already existed -- `export_gsurf.py`, in the gstruct repo -- goes
the other way on purpose. It flattens: `value_at` sampled at mid-trace, one row
per (trace, attitude), spans and lineations and planeless records dropped. That
is the right shape for *drawing* a section, and it cannot come back, because
most of the file is not in it.

This one is the complete projection. Every record of the model becomes a row,
including the ones the consumption export has no column for, so that a dataset
written out here can be rebuilt **from the layers alone** -- not patched onto the
file it came from. `from_geopackage` is the inverse and lives in this module
rather than in an importer, because "lossless" is otherwise a claim about
columns rather than a property anybody has checked: `checks/check_exports.py`
exports the real corpus, reads it back, and compares the text byte for byte.

**The anchors are the truth and `s` is not, which is what makes the trace
geometry safe to edit.** The model already works this way -- `Anchored.resolve`
projects an anchor onto the path, `Span.resolve` derives `s0`/`s1` -- so moving a
vertex in QGIS re-projects the records of that structure on the way back in,
which is the correct answer and not a loss. `s_m`, `offset_m`, `s0_m`, `s1_m`,
`length_m` and `n_vertices` are exported because a geologist filtering a layer
needs them, and ignored on the way back in: they are derived, and a derived
column that is read as truth is how two numbers that must agree stop agreeing.

**`*` is kept as a flag, not as a coordinate.** A span or fit end written `*`
means "the end of the path, whatever the path becomes" -- so with the geometry
editable the distinction is load-bearing: extend a trace and an open end should
follow it, while an anchored one should stay where the geologist put it. The
geometry of a span is drawn along the trace it applies to, so it is visible on
the map, but on the way back in only its two endpoints are read, and only for
the ends `open_start`/`open_end` say are anchored.

**What a vector layer cannot hold, and what is done about it.** The comments are
the geologist's reasoning and the only place it is written down, so they travel:
`comment` is the inline one, `lead_json` the lines above a record. They are
carried, not edited -- QGIS has no reason to maintain them, and nothing here
pretends it will. Two things are weaker than that:

  - **Per-vertex comments.** Those are keyed by vertex index, and editing a
    geometry destroys vertex identity -- inserting a vertex before a commented
    one slides the comment onto a different vertex, which is worse than losing
    it, and checking that the index still exists does not catch that. So the
    export carries a fingerprint of the path (`path_hash`), and they are
    re-applied only to a trace that came back unreshaped; everywhere else they
    are dropped and the report names the structure. The fingerprint is taken at
    the precision the format writes, so re-saving a layer untouched does not
    count as a reshape.
  - **An attribute whose value is empty.** `gstruct._kw` skips those, so they do
    not survive a `dumps` either. The parity is deliberate: this export loses
    exactly what the format's own writer loses, and not one thing more.

The source text travels too, in `gs_source`, with its hash. It is provenance and
not a baseline -- nothing in `from_geopackage` reads it -- so an export stays
self-sufficient and an edit in QGIS is not a diff against a past the geologist
has to keep. It is there so that what a file *was* can still be answered after
it has been rewritten, and so a check can compare without being handed the
original separately.

The two plain tables stay out of `gpkg_contents`, which is what qgSurf does with
its own side tables (`utils/sectioneditor/db.py`): they are not content of the
GeoPackage and no conformant reader will treat them as layers to be drawn. That
buys less than it sounds like -- GDAL lists every table in the file whatever
`gpkg_contents` says, so QGIS will still offer them, with no geometry -- but the
declaration is the part that is ours to get right, and the alternative is a
header that arrives in the layer tree looking like something to style.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from gsurf.curation import module

if TYPE_CHECKING:  # annotations only -- nothing is imported at run time
    import gstruct as gs

_LIBRARY = None


def _gs():
    """The gstruct library, through curation's gate rather than around it.

    `curation.module()` is the one door, and not out of tidiness: it turns the
    ImportError into something a dialog can show -- gstruct is on no package
    index, so "pip install gstruct" installs an unrelated project -- and it
    refuses a library older than 0.2. That floor is this module's business as
    much as anybody's: the `use` axis arrived in 0.2, this export carries `use`
    spans, and a 0.1 library would hand back a refusal it had not acted on.
    """
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = module()
    return _LIBRARY


# Geometry layers, in the order they are written. The names carry the `gs_`
# prefix so that a project holding a geological map as well does not end up with
# two layers called `traces` whose columns mean different things.
TRACES = "gs_traces"
ATTITUDES = "gs_attitudes"
LINEATIONS = "gs_lineations"
FITS = "gs_fits"
SPANS = "gs_spans"
OBSERVATIONS = "gs_observations"

LAYERS = (TRACES, ATTITUDES, LINEATIONS, FITS, SPANS, OBSERVATIONS)

# Plain tables: the header needed to rebuild, and the provenance that is not.
DATASET_TABLE = "gs_dataset"
SOURCE_TABLE = "gs_source"

# Bumped when a reader of an older export would read it *wrongly* rather than
# read less of it -- the same rule the format itself uses for its own version.
SCHEMA = 1


@dataclass
class Report:
    """What went out, and what could not be carried whole."""
    path: str = ""
    crs: str = ""
    counts: Dict[str, int] = field(default_factory=dict)
    # Structures whose per-vertex comments were dropped on the way back in,
    # because the geometry no longer has the vertices they name.
    revertexed: List[str] = field(default_factory=list)
    # Structures whose trace geometry came back changed. Not a loss -- it is the
    # point of exporting to QGIS -- but the geologist is owed the list, because
    # every anchored record of those structures has been re-projected.
    reshaped: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.counts.values())


# ------------------------------------------------------------------- attributes

def _attrs_text(attrs: Dict[str, str]) -> str:
    """The attribute dict as the file spells it.

    Not JSON. The format's own key/value syntax goes in the column, through the
    format's own quoting, so the text in QGIS is the text on the line and the
    way back is `gstruct._split` rather than a second parser that can disagree
    with the first one.
    """
    return _gs()._kw(attrs).strip()


def _attrs_of(text: Optional[str]) -> Dict[str, str]:
    if not text:
        return {}
    pos, kw = _gs()._split(text)
    # A bare word in that column is not a key/value pair and the format has
    # nowhere to put it on a record line; keep it as a valueless attribute
    # rather than dropping it, which is what `loads` does with an unknown word.
    for word in pos:
        kw.setdefault(word, "")
    return kw


def _lead_text(lead: Optional[Sequence[str]]) -> Optional[str]:
    """The lines above a record, as JSON.

    JSON and not a joined string because `[]` and `[""]` are different files --
    no blank line, and one blank line -- and joining makes them the same text.
    `None` stays NULL, which for a structure means "the writer decides".
    """
    return None if lead is None else json.dumps(list(lead))


def _lead_of(text: Optional[str]) -> Optional[List[str]]:
    if text is None or text == "":
        return None
    try:
        got = json.loads(text)
    except ValueError:
        return None
    return [str(x) for x in got] if isinstance(got, list) else None


def _json_or_none(obj) -> Optional[str]:
    return json.dumps(obj) if obj else None


def _loaded(text: Optional[str]):
    if not text:
        return {}
    try:
        return json.loads(text)
    except ValueError:
        return {}


# --------------------------------------------------------------------- geometry

def _sub_path(path: Sequence[gs.XY], s0: Optional[float], s1: Optional[float],
              start: Optional[gs.XY], end: Optional[gs.XY]):
    """The stretch of a trace a span or fit applies to.

    Drawn along the path, so that on the map it lies on the ground it is about;
    read back by its endpoints only. The endpoints are the anchors themselves
    where there are anchors, so a round trip reproduces `@x,y` exactly instead
    of a point resampled onto a segment.
    """
    if len(path) < 2:
        # A curation names structures and carries no geometry. The anchors are
        # still written, so they are what there is to draw.
        if start is not None and end is not None:
            return [start, end]
        return None
    if s0 is None or s1 is None:
        return None
    lo, hi = (s0, s1) if s0 <= s1 else (s1, s0)
    head = start if start is not None else _gs().point_at(path, 0.0)
    tail = end if end is not None else _gs().point_at(path, _gs().path_length(path))
    if s0 > s1:
        head, tail = tail, head
    out = [head]
    run = 0.0
    for a, b in zip(path, path[1:]):
        run += ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        if lo < run < hi:
            out.append(b)
    out.append(tail)
    if len(out) < 2 or (len(out) == 2 and out[0] == out[1]):
        # A zero-length span is a legal thing to have written; a one-point
        # LineString is not a legal thing to store.
        return [out[0], out[0]]
    return out


def _ends_of(geom, open_start: bool, open_end: bool
             ) -> Tuple[Optional[gs.XY], Optional[gs.XY]]:
    """The two anchors of a span or fit, from its geometry and its two flags."""
    if open_start and open_end:
        return None, None
    coords = [] if geom is None else list(geom.coords)
    if len(coords) < 2:
        return None, None
    head = None if open_start else (round(coords[0][0], 2), round(coords[0][1], 2))
    tail = None if open_end else (round(coords[-1][0], 2), round(coords[-1][1], 2))
    return head, tail


def _point(anchor: Optional[gs.XY]):
    from shapely.geometry import Point
    return None if anchor is None else Point(anchor)


def _line(coords):
    from shapely.geometry import LineString
    return None if not coords or len(coords) < 2 else LineString(coords)


def _anchor_of(geom) -> Optional[gs.XY]:
    if geom is None or geom.is_empty:
        return None
    return (round(geom.x, 2), round(geom.y, 2))


def _rounded(value) -> Optional[float]:
    return None if value is None else round(float(value), 2)


# ------------------------------------------------------------------------ export

def _rows(ds: gs.Dataset) -> Dict[str, List[dict]]:
    traces, attitudes, lineations, fits, spans, observations = [], [], [], [], [], []

    for ndx, st in enumerate(ds.structures):
        traces.append({
            "ident": st.ident,
            # NULL = the line carried no label token, '' = it carried an empty
            # one. The two say the same thing and are different spellings, and
            # a GeoPackage keeps them apart as QGIS does.
            "label": st.label,
            "kind": st.kind,
            "attrs": _attrs_text(st.attrs),
            "comment": st.tail or None,
            "lead_json": _lead_text(st.lead),
            "line_lead_json": _json_or_none(st.line_lead),
            "line_tail_json": _json_or_none(st.line_tail),
            "seq": ndx,
            "n_vertices": len(st.path),
            "length_m": round(st.length, 2) if len(st.path) > 1 else None,
            # Not decoration: it is the only way the reader can tell a reshaped
            # trace from an untouched one, and the vertex comments turn on it.
            "path_hash": path_fingerprint(st.path),
            "geometry": _line(st.path),
        })

        for pos, at in enumerate(st.attitudes):
            attitudes.append({
                "ident": st.ident,
                "seq": pos,
                "dip_dir": None if at.plane is None else at.plane.dip_dir,
                "dip": None if at.plane is None else at.plane.dip,
                "attrs": _attrs_text(at.attrs),
                "comment": at.tail or None,
                "lead_json": _lead_text(at.lead),
                "s_m": _rounded(at.s),
                "offset_m": _rounded(at.offset),
                "geometry": _point(at.anchor),
            })

        for pos, ln in enumerate(st.lineations):
            lineations.append({
                "ident": st.ident,
                "seq": pos,
                "trend": ln.trend,
                "plunge": ln.plunge,
                "attrs": _attrs_text(ln.attrs),
                "comment": ln.tail or None,
                "lead_json": _lead_text(ln.lead),
                "s_m": _rounded(ln.s),
                "offset_m": _rounded(ln.offset),
                "geometry": _point(ln.anchor),
            })

        for pos, ft in enumerate(st.fits):
            fits.append({
                "ident": st.ident,
                # Order is not decoration here: `attitude_at` takes the FIRST
                # fit that covers a progressive, so which row came first is part
                # of the answer, and a GeoPackage promises no row order.
                "seq": pos,
                "dip_dir": None if ft.plane is None else ft.plane.dip_dir,
                "dip": None if ft.plane is None else ft.plane.dip,
                "open_start": int(ft.start is None),
                "open_end": int(ft.end is None),
                "attrs": _attrs_text(ft.attrs),
                "comment": ft.tail or None,
                "lead_json": _lead_text(ft.lead),
                "s0_m": _rounded(ft.s0),
                "s1_m": _rounded(ft.s1),
                "geometry": _line(_sub_path(st.path, ft.s0, ft.s1, ft.start, ft.end)),
            })

        for pos, sp in enumerate(st.spans):
            spans.append({
                "ident": st.ident,
                # And here it is precedence outright: the LAST span covering a
                # progressive wins, which is how a local correction is written
                # without touching the general one.
                "seq": pos,
                "axis": sp.axis,
                "value": sp.value,
                "open_start": int(sp.start is None),
                "open_end": int(sp.end is None),
                "attrs": _attrs_text(sp.attrs),
                "comment": sp.tail or None,
                "lead_json": _lead_text(sp.lead),
                "s0_m": _rounded(sp.s0),
                "s1_m": _rounded(sp.s1),
                "geometry": _line(_sub_path(st.path, sp.s0, sp.s1, sp.start, sp.end)),
            })

    for ndx, ob in enumerate(ds.observations):
        observations.append({
            "ident": ob.ident,
            "seq": ndx,
            "dip_dir": None if ob.plane is None else ob.plane.dip_dir,
            "dip": None if ob.plane is None else ob.plane.dip,
            "attrs": _attrs_text(ob.attrs),
            "comment": ob.tail or None,
            "lead_json": _lead_text(ob.lead),
            "geometry": _point(ob.anchor),
        })

    return {TRACES: traces, ATTITUDES: attitudes, LINEATIONS: lineations,
            FITS: fits, SPANS: spans, OBSERVATIONS: observations}


_GEOM_TYPE = {
    TRACES: "LineString", FITS: "LineString", SPANS: "LineString",
    ATTITUDES: "Point", LINEATIONS: "Point", OBSERVATIONS: "Point",
}

# A column that is NULL in every row of a short layer would otherwise be typed
# from nothing, and a text column guessed as float comes back as NaN.
_DTYPES = {
    "ident": "string", "label": "string", "kind": "string", "axis": "string",
    "value": "string", "attrs": "string", "comment": "string",
    "lead_json": "string", "line_lead_json": "string", "line_tail_json": "string",
    "path_hash": "string",
}


def to_geopackage(ds: gs.Dataset, path: str, *, source_text: Optional[str] = None,
                  source_path: Optional[str] = None) -> Report:
    """Write a dataset out as a GeoPackage. Returns what went.

    `source_text` is kept verbatim for provenance and never read back; pass the
    text the dataset was loaded from when there is one.
    """
    import geopandas as gpd
    import pandas as pd
    import pyogrio

    rows = _rows(ds)
    report = Report(path=path, crs=ds.crs,
                    counts={name: len(rows[name]) for name in LAYERS})

    if os.path.exists(path):
        # Appending layer by layer into a file that already holds a previous
        # export would leave the old rows beside the new ones under the same
        # names. The export is whole or it is not an export.
        os.remove(path)

    first = True
    for name in LAYERS:
        if rows[name]:
            frame = gpd.GeoDataFrame(rows[name], geometry="geometry",
                                     crs=ds.crs or None)
            for column, kind in _DTYPES.items():
                if column in frame.columns:
                    frame[column] = frame[column].astype(kind)
        else:
            # An empty layer still goes out: a reader must be able to tell
            # "this dataset has no lineations" from "this export forgot them".
            frame = gpd.GeoDataFrame(
                {column: pd.Series(dtype=_DTYPES.get(column, "float64"))
                 for column in _columns_of(name)},
                geometry=gpd.GeoSeries([], crs=ds.crs or None), crs=ds.crs or None)
        pyogrio.write_dataframe(frame, path, layer=name, driver="GPKG",
                                geometry_type=_GEOM_TYPE[name],
                                promote_to_multi=False, append=not first)
        first = False

    _write_header(path, ds, source_text, source_path, report)
    return report


def _columns_of(name: str) -> List[str]:
    common = ["ident", "seq", "attrs", "comment", "lead_json"]
    if name == TRACES:
        return ["ident", "label", "kind", "attrs", "comment", "lead_json",
                "line_lead_json", "line_tail_json", "seq", "n_vertices",
                "length_m", "path_hash"]
    if name == ATTITUDES:
        return common + ["dip_dir", "dip", "s_m", "offset_m"]
    if name == LINEATIONS:
        return common + ["trend", "plunge", "s_m", "offset_m"]
    if name == FITS:
        return common + ["dip_dir", "dip", "open_start", "open_end", "s0_m", "s1_m"]
    if name == SPANS:
        return common + ["axis", "value", "open_start", "open_end", "s0_m", "s1_m"]
    return common + ["dip_dir", "dip"]


def _write_header(path: str, ds: gs.Dataset, source_text: Optional[str],
                  source_path: Optional[str], report: Report) -> None:
    """The dataset-level lines, and the provenance, as plain tables."""
    con = sqlite3.connect(path)
    try:
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {DATASET_TABLE} (
                schema_version   INTEGER NOT NULL,
                gstruct_version  TEXT,
                crs              TEXT,
                meta_json        TEXT,
                line_lead_json   TEXT,
                line_tail_json   TEXT,
                trailer_json     TEXT
            );""")
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {SOURCE_TABLE} (
                source_path      TEXT,
                source_sha256    TEXT,
                source_bytes     INTEGER,
                source_text      TEXT,
                exported_at      TEXT,
                exported_by      TEXT
            );""")
        con.execute(f"DELETE FROM {DATASET_TABLE};")
        con.execute(f"DELETE FROM {SOURCE_TABLE};")
        con.execute(
            f"INSERT INTO {DATASET_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?);",
            (SCHEMA, ds.meta.get("version", _gs().VERSION), ds.crs,
             _json_or_none(ds.meta), _json_or_none(ds.line_lead),
             _json_or_none(ds.line_tail), _json_or_none(ds.trailer)))
        blob = (source_text or "").encode("utf-8")
        con.execute(
            f"INSERT INTO {SOURCE_TABLE} VALUES (?, ?, ?, ?, ?, ?);",
            (source_path,
             hashlib.sha256(blob).hexdigest() if source_text is not None else None,
             len(blob) if source_text is not None else None,
             source_text,
             datetime.now(timezone.utc).isoformat(timespec="seconds"),
             f"gsurf.exports schema {SCHEMA}, gstruct {_gs().VERSION}"))
        con.commit()
    finally:
        con.close()
    if source_text is None:
        report.notes.append("no source text carried: provenance table is empty")


# ------------------------------------------------------------------------ import

def from_geopackage(path: str) -> Tuple[gs.Dataset, Report]:
    """Rebuild a dataset from the layers. Nothing here reads the source text.

    The order of the rows is taken from `seq` and not from the file: a
    GeoPackage promises no row order, and for spans `seq` *is* the precedence.
    """
    import pyogrio

    report = Report(path=path)
    ds = _gs().Dataset()

    con = sqlite3.connect(path)
    try:
        got = con.execute(
            f"SELECT schema_version, gstruct_version, crs, meta_json, "
            f"line_lead_json, line_tail_json, trailer_json FROM {DATASET_TABLE};"
        ).fetchone()
    finally:
        con.close()
    if got is None:
        raise ValueError(f"{path} carries no {DATASET_TABLE}: not a gstruct export")
    schema, version, crs, meta, line_lead, line_tail, trailer = got
    if schema > SCHEMA:
        raise ValueError(
            f"{path} was written by schema {schema}; this reader knows {SCHEMA}")
    ds.crs = crs or ""
    ds.meta = dict(_loaded(meta))
    ds.meta.setdefault("version", version or _gs().VERSION)
    ds.line_lead = {k: list(v) for k, v in _loaded(line_lead).items()}
    ds.line_tail = dict(_loaded(line_tail))
    ds.trailer = list(_loaded(trailer) or [])

    frames = {}
    for name in LAYERS:
        frame = pyogrio.read_dataframe(path, layer=name)
        frames[name] = frame.sort_values("seq") if "seq" in frame.columns else frame

    by_ident: Dict[str, gs.Structure] = {}
    for _, row in frames[TRACES].iterrows():
        st = _gs().Structure(
            ident=_text(row["ident"]) or "",
            label=_text(row["label"], blank_is_empty=True),
            kind=_text(row["kind"]) or "unknown",
            attrs=_attrs_of(_text(row["attrs"])),
            lead=_lead_of(_text(row["lead_json"])),
            tail=_text(row["comment"]) or "",
        )
        st.path = [] if row["geometry"] is None else [
            (round(x, 2), round(y, 2)) for x, y in row["geometry"].coords]
        st.line_lead = {k: list(v)
                        for k, v in _loaded(_text(row["line_lead_json"])).items()}
        st.line_tail = dict(_loaded(_text(row["line_tail_json"])))
        _vertex_comments(st, _text(row["path_hash"]) if "path_hash" in row else None,
                         report)
        ds.structures.append(st)
        by_ident[st.ident] = st

    def owner(row):
        return by_ident.get(_text(row["ident"]) or "")

    for _, row in frames[ATTITUDES].iterrows():
        st = owner(row)
        if st is None:
            report.notes.append(f"attitude on unknown structure {row['ident']!r}")
            continue
        st.attitudes.append(_gs().Attitude(
            anchor=_anchor_of(row["geometry"]),
            plane=_plane_of(row["dip_dir"], row["dip"]),
            attrs=_attrs_of(_text(row["attrs"])),
            lead=_lead_of(_text(row["lead_json"])) or [],
            tail=_text(row["comment"]) or "",
        ))

    for _, row in frames[LINEATIONS].iterrows():
        st = owner(row)
        if st is None:
            report.notes.append(f"lineation on unknown structure {row['ident']!r}")
            continue
        st.lineations.append(_gs().Lineation(
            anchor=_anchor_of(row["geometry"]),
            trend=_number(row["trend"]), plunge=_number(row["plunge"]),
            attrs=_attrs_of(_text(row["attrs"])),
            lead=_lead_of(_text(row["lead_json"])) or [],
            tail=_text(row["comment"]) or "",
        ))

    for _, row in frames[FITS].iterrows():
        st = owner(row)
        if st is None:
            report.notes.append(f"fit on unknown structure {row['ident']!r}")
            continue
        head, tail = _ends_of(row["geometry"], _flag(row["open_start"]),
                              _flag(row["open_end"]))
        st.fits.append(_gs().Fit(
            plane=_plane_of(row["dip_dir"], row["dip"]),
            start=head, end=tail,
            attrs=_attrs_of(_text(row["attrs"])),
            lead=_lead_of(_text(row["lead_json"])) or [],
            tail=_text(row["comment"]) or "",
        ))

    for _, row in frames[SPANS].iterrows():
        st = owner(row)
        if st is None:
            report.notes.append(f"span on unknown structure {row['ident']!r}")
            continue
        head, tail = _ends_of(row["geometry"], _flag(row["open_start"]),
                              _flag(row["open_end"]))
        st.spans.append(_gs().Span(
            axis=_text(row["axis"]) or "", value=_text(row["value"]) or "",
            start=head, end=tail,
            attrs=_attrs_of(_text(row["attrs"])),
            lead=_lead_of(_text(row["lead_json"])) or [],
            tail=_text(row["comment"]) or "",
        ))

    for _, row in frames[OBSERVATIONS].iterrows():
        ds.observations.append(_gs().Observation(
            ident=_text(row["ident"]) or "",
            anchor=_anchor_of(row["geometry"]),
            plane=_plane_of(row["dip_dir"], row["dip"]),
            attrs=_attrs_of(_text(row["attrs"])),
            lead=_lead_of(_text(row["lead_json"])),
            tail=_text(row["comment"]) or "",
        ))

    report.counts = {name: len(frames[name]) for name in LAYERS}
    report.crs = ds.crs
    return ds.resolve(), report


def path_fingerprint(path: Sequence[gs.XY]) -> str:
    """The path as the format writes it, hashed.

    Hashed at the precision the format writes (`.2f`), so that a QGIS session
    that opens a layer and saves it again -- perturbing doubles below the
    centimetre, or round-tripping them through another library -- is not reported
    as having reshaped a trace. A reshape is a change the file would show.
    """
    body = ";".join(f"{x:.2f},{y:.2f}" for x, y in path)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


def _vertex_comments(st: gs.Structure, stored: Optional[str], report: Report) -> None:
    """A comment on "the third vertex" of a path that has been reshaped.

    Vertex comments are keyed by index, and an index stops being an identity
    the moment a geometry is edited: inserting a vertex *before* a commented one
    silently slides the comment onto a different vertex, which is worse than
    losing it. Checking that the index still exists does not catch that -- only
    comparing the path itself does, which is why the export carries a
    fingerprint of it.

    So they are kept whole where the trace came back unreshaped, and dropped
    where it did not, with the report naming which structure paid. This is the
    one place the export is knowingly lossy, and it is lossy in the direction of
    saying nothing rather than saying something wrong.
    """
    keys = {k for k in (*st.line_lead, *st.line_tail) if k.startswith("v")}
    if stored is not None and stored != path_fingerprint(st.path):
        report.reshaped.append(st.ident)
        if not keys:
            return
        for k in keys:
            st.line_lead.pop(k, None)
            st.line_tail.pop(k, None)
        report.revertexed.append(st.ident)
        return
    # Unreshaped, but a comment may still name a vertex that is not there -- a
    # hand-edited export, or a path written shorter than its comments.
    for k in keys:
        if not k[1:].isdigit() or int(k[1:]) >= len(st.path):
            st.line_lead.pop(k, None)
            st.line_tail.pop(k, None)
            if st.ident not in report.revertexed:
                report.revertexed.append(st.ident)


def _text(value, blank_is_empty: bool = False) -> Optional[str]:
    """A string column, with pandas' several ways of saying nothing."""
    if value is None:
        return None
    try:
        import pandas as pd
        if pd.isna(value):
            return None
    except (ImportError, TypeError, ValueError):
        pass
    got = str(value)
    if got == "" and not blank_is_empty:
        return None
    return got


def _flag(value) -> bool:
    """One of the two open-end flags, as a row written by hand may leave it.

    NULL means anchored and not open. A row added in QGIS carries NULL in every
    column the geologist did not fill in, and `bool(nan)` is `True` -- so read
    naively, a span just drawn would come back with both ends `*`, throwing the
    geometry away and asserting the value over the whole trace. That is the
    shape of wrong answer this whole module exists to avoid, so the default
    here is the one that keeps what was drawn.
    """
    if value is None:
        return False
    try:
        import pandas as pd
        if pd.isna(value):
            return False
    except (ImportError, TypeError, ValueError):
        pass
    return bool(int(value)) if str(value).strip() not in ("", "nan") else False


def _number(value) -> Optional[float]:
    if value is None:
        return None
    try:
        import pandas as pd
        if pd.isna(value):
            return None
    except (ImportError, TypeError, ValueError):
        pass
    return float(value)


def _plane_of(dip_dir, dip) -> Optional[gs.Plane]:
    a, b = _number(dip_dir), _number(dip)
    return None if a is None or b is None else _gs().Plane(a, b)


# -------------------------------------------------------------------- round trip

def source_of(path: str) -> Tuple[Optional[str], Optional[str]]:
    """The text an export was made from, and its hash. Provenance, not input."""
    con = sqlite3.connect(path)
    try:
        got = con.execute(
            f"SELECT source_text, source_sha256 FROM {SOURCE_TABLE};").fetchone()
    finally:
        con.close()
    return (None, None) if got is None else (got[0], got[1])


def round_trip(src: str, gpkg: str) -> Tuple[bool, str, str, Report]:
    """Export a `.gstruct`, read it back, and compare the two texts.

    This is the only thing that makes "lossless" a fact rather than a column
    listing, so it lives beside the export and not only in the checks.
    """
    with open(src, encoding="utf-8", newline="") as fh:
        before = fh.read()
    ds = _gs().loads(before)
    to_geopackage(ds, gpkg, source_text=before, source_path=src)
    back, report = from_geopackage(gpkg)
    after = _gs().dumps(back)
    # The comparison is against what the format's own writer makes of the
    # original, not against the bytes on disk: `dumps` is canonical, and a file
    # it would have rewritten anyway is not something this export broke.
    canonical = _gs().dumps(ds)
    return after == canonical, canonical, after, report
