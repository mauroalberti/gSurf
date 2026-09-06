# gSurf

Lay an unbounded geological plane on a DEM and watch where it crops out, while
you turn the dial. The intersection is recomputed on every frame, not behind a
"Calculate" button, so a dip direction is something you sweep through rather
than something you guess and check.

![gSurf, real-time plane/DEM intersection](ims/realtime_intersection.png)

The heavy lifting is done by [misah](https://gitlab.com/mauroalberti/misah), a
Rust crate with Python bindings: its marching-squares kernel returns the chords
where the plane cuts the grid, and this application is the minimum needed to
steer that kernel by hand and see the answer move. Kernel, drawing and frame
rate are reported in the status bar on every frame, so the cost stays visible
while you work.

### Status

`realtime_intersection.py`, at the repository root, is the part that runs.

The `gSurf/` package is the older application and **does not currently run**:
it is kept for the code worth porting, not for use. `python -m gSurf` fails
outright — there is no `__main__.py` — and beyond that `gSurf/gSurf.py` imports
`pygsf`, `gst` and `pygmt`, `gSurf/intersections/` still wants PyQt5, and
`gSurf/stereoplot/` looks for a vendored `apsg` that is not in the tree. Only
`gSurf/profiles/profiles_tools.py` imports cleanly.

Alpha stage. The repository dates from 2012-04-08, was worked on through 2019,
lay dormant for three years, was restarted 2022-11-28, and was rebuilt around
the misah kernel in 2026. Development is on `master` here on GitLab; the
GitHub repository is archived, and the old `profiles` branch survives only
there.

### Requirements

Python 3.9+, and:

```bash
pip install misah numpy rasterio PyQt6 matplotlib pyproj
```

`misah` is on PyPI, at an alpha version — expect it to move. Developed and run
on Python 3.13 against a local misah 0.2.0a0, numpy 2.5.2, rasterio 1.5.1,
PyQt6 6.11.0, matplotlib 3.10.9 and pyproj 3.7.2; the syntax itself stays
within 3.9.

Vector backdrops and shapefile export additionally need:

```bash
pip install geopandas shapely
```

They are imported only where they are actually used, so without them the tool
still starts, draws the DEM and recomputes the intersection as you drag; you
get no layers underneath, and `Export trace` raises on the way out.

### Usage

Launched bare, it asks for what it needs:

```bash
python realtime_intersection.py
```

Or state it up front:

```bash
python realtime_intersection.py dem.tif \
    --polygons geology.gpkg:carbonates \
    --lines    geology.gpkg:faults \
    --points   stations.shp \
    --x 611240 --y 4409700 --window 1000
```

The DEM is the only thing required. The three vector slots — polygons, lines,
points — answer a different question from the calculation: where the plane is
being laid down. In the dialog, the layers offered in each slot are filtered on
geometry read from the file's metadata, so faults never appear among the
polygons and a geometry-less attribute table appears nowhere.

Polygons are coloured per unit, on the field given by `--categories` (default
`code`, `none` to switch it off). The colours are drawn from the layer's
complete list of values rather than from whatever is currently in view, so a
formation keeps its colour as you pan. The legend is cut by outcropping area,
not alphabetically, so the units that make up the map are the ones named.

The legend stands beside the map rather than on it: categorised it runs to some
thirty entries, and inside the frame those cover the corner you were most
likely looking at. The *Legend* box in the panel moves it at any time —
`beside`, `inside`, `hidden` — and `--legend` picks where it starts.
Wherever it goes it belongs to the figure and not to a widget alongside, so
both the saved and the copied screenshot carry it.

`--settings file.json` reopens a saved orientation. Explicit arguments win over
the file, so `--settings x.json --z 900` is the saved plane at a new elevation.
Files written before the interface changed language still load: the Italian
keys are read as a fallback.

### Things worth knowing

**Dip direction is true azimuth**, as a compass reads it once declination is
corrected. The DEM is on the projection's grid, and the two norths do not
agree: meridian convergence is subtracted before the kernel is called, and
shown under the attitude. In the southern Apennines in EPSG:25833 it runs
between +0.41° and +1.04°, which over five kilometres of trace is up to 91 m —
twenty times the DEM cell. It is measured rather than taken from a formula: a
100 m step along true north, then read what azimuth that step has on the grid.
Eight microseconds, and it holds for any projection.

**The DEM is never loaded.** The background is a decimated overview; the kernel
reads one full-resolution window at a time, and `--window` sets its side in
cells. A 234 Mpx mosaic therefore costs the same per frame as a small crop,
where loading it would have been 2.5 GB in float64.

**Give your DEM overviews.** Without them the initial decimation has to read
the whole raster. On a 234 Mpx mosaic that is the difference between 3.5 s and
0.35 s at startup:

```bash
gdaladdo -r average dem.tif 2 4 8 16
```

**The source point can leave the ground.** Its three components are
independent: what you omit the DEM decides — no `--x`/`--y` puts it at the
centre, no `--z` takes the ground elevation. But a `z` you give stays given,
and survives dragging the point, which is how a plane is laid on a horizon
passing above or below today's topography. The *elevation from DEM* checkbox makes
and breaks the tie at any time, and both exports record which way it was along
with the ground elevation underneath.

**Exports carry their own frame.** Both azimuths (true and grid) and both
coordinate pairs (projected and geographic) go into the JSON and the shapefile.
Coordinates in a single EPSG are unusable outside it, and the `.prj` is the
file that goes missing first.

### Performance

Measured on the 234 Mpx 5 m mosaic, dragging the dial, PyQt6 with matplotlib
blitting:

| window | ms/frame | fps |
|---|---|---|
| 500² (2.5 km) | 10.9 | 92 |
| 1000² (5 km) | 26.6 | 38 |
| 2000² (10 km) | 95.7 | 11 |

Real time holds to 1000². At 2000² the kernel alone takes most of the frame,
and decimation or tiling would be needed. On the same plane and grid the misah
kernel runs about 77× the pure-Python equivalent in geogst, which is what makes
dragging possible at all.

### Related

- [misah](https://gitlab.com/mauroalberti/misah) — the Rust kernels
- [geogst](https://gitlab.com/mauroalberti/geogst) — types, CRS and I/O
- [qgSurf](https://gitlab.com/mauroalberti/qgSurf) — the QGIS plug-in, whose
  plane/DEM intersection this replaces with an interactive one
