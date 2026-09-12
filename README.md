# gSurf

Structural geology you steer by hand: the answer is recomputed on every frame,
not behind a "Calculate" button, so a parameter is something you sweep through
rather than something you guess and check.

```bash
python -m gsurf
```

Two tools so far. **Plane on a DEM** lays an unbounded geological plane on the
topography and shows where it crops out while you turn the dial. **Fold axes**
drags a circular window across a map of bedding attitudes and shows, on a
stereonet that follows it, the girdle the poles spread on and the axis they
turn about.

They are picked from one launcher, and the tool comes before the question: pick
one and it asks for the sources *it* takes, with what it cannot run without
marked as required — the plane needs a DEM, the fold axes need attitudes, and
neither is asked for the other's. What you answer is put back the next time,
and the session behind it is reopened only when the answer has changed.

![gSurf, real-time plane/DEM intersection](ims/realtime_intersection.png)

The heavy lifting is done by [misah](https://gitlab.com/mauroalberti/misah), a
Rust crate with Python bindings: its marching-squares kernel returns the chords
where the plane cuts the grid, and this application is the minimum needed to
steer that kernel by hand and see the answer move. Kernel, drawing and frame
rate are reported in the status bar on every frame, so the cost stays visible
while you work.

### Status

`gsurf/tools/` holds the tools, one module each, and the package around them
holds what is not about any one calculation, so that the next tool inherits it
rather than copying it:

- `gsurf/__main__.py`, `gsurf/launcher.py` — the front door: the tools, what
  each one is opened on, and the session kept between them.
- `gsurf/session.py` — the projection, the area, and what has been opened in
  them. The DEM is one of the things in a session and not the frame itself: a
  session can be opened on vector layers alone, taking its CRS and extent from
  their metadata.
- `gsurf/sources.py` — what to open, asked once a tool has been picked. A tool
  declares its slots and which are required; layers are listed and fields read
  off the metadata, so the dialog filters without loading.
- `gsurf/mapview.py` — the map: hillshade if there is a DEM, vector backdrop,
  navigation, the blitting surface a tool draws its own artists on, and a
  legend whose entries are switches rather than captions — click one and what
  it names comes off the map.
- `gsurf/dem.py`, `gsurf/vectors.py`, `gsurf/convergence.py` — the DEM read by
  windows, the backdrop layers, and grid north against true north.
- `gsurf/attitudes.py`, `gsurf/folds.py`, `gsurf/stereonet.py` — located
  attitudes read strictly, the orientation tensor read as a fold, and an
  equal-area net that redraws while you move. The net is a plain widget and
  owns no window, which is what lets the fold-axis tool float it over the map.

The pre-2026 application lived in a `gSurf/` package beside this one and was
removed in 2026-09: it had not run since the rebuild — `pygsf`, `gst` and
`pygmt` for the main window, PyQt5 and Python-2 implicit relative imports for
its intersection GUI, a vendored `apsg` that was never in the tree for its
stereoplot — and nothing outside it imported it. Two things in it are worth
porting and are a `git show a972c58:gSurf/...` away: the topographic profiles
of `gSurf.py`, with attitudes projected onto them, and the fault-and-slickenline
stereoplot of `stereoplot/`, which reads a rake and a movement sense that
`gsurf/stereonet.py` does not.

Alpha stage. The repository dates from 2012-04-08, was worked on through 2019,
lay dormant for three years, was restarted 2022-11-28, and was rebuilt around
the misah kernel in 2026. Development is on `master` here on GitLab; the
GitHub repository is archived, and the old `profiles` branch survives only
there.

### Checks

```bash
python checks/run.py            # off-screen, about ten seconds
python checks/run.py --show     # let the windows appear
```

A hundred and fifty-one assertions over four scripts, each also runnable on its
own. They drive real windows through synthesized mouse events, so Qt is put in
its off-screen mode unless you ask otherwise.

They are checks and not unit tests, in that most of them assert against
something known from outside the code: a fold axis recovered from a synthetic
fold that has one by construction, a right-hand-rule strike agreeing with a dip
direction, a regated field equalling a recomputed one. `check_interaction.py`
in particular was written to run against either side of a refactor, which is
how the map was lifted out of the intersection tool without changing it — the
same clicks, the same window offsets, the same point counts, digit for digit.

`checks/synthetic.py` is not a check but the bench the frame costs quoted below
were measured on, and builds the synthetic DEM the others use.

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

The fold axes need those two as well, and additionally:

```bash
pip install geogst mplstereonet
```

geogst is where the orientation tensor and Woodcock's parameters come from, and
mplstereonet draws the net — `import mplstereonet` is also what registers the
equal-area projection with matplotlib, so it is not an optional extra there.

### Usage — the launcher

```bash
python -m gsurf
```

It opens on the tools and asks nothing yet. Pick one and it asks for what that
tool takes — the plane for a DEM and three optional backdrop slots, the fold
axes for an attitude layer and the two columns its angles are in. The slot it
cannot run without is named `(required)` and coloured, and goes green once it
holds something; `Open` stays refused until it does, and until at least one
source has been named at all, since a session takes its projection and its
extent from what was opened.

What you answer is kept, slot by slot. Going to the other tool re-proposes the
same files, so the second question is usually one keystroke — and if the answer
comes back unchanged, the session itself is handed on as it stands rather than
reopened, which on a large DEM is the whole cost of starting a tool. Closing a
tool brings the launcher back with it still open.

Asking per tool is also what keeps the dialog short. Asked before the tool is
known, it has to cover every source any tool might want: a longer dialog that
says less about the one you picked, and tall enough to push its own buttons off
a short screen. It scrolls now, and can be dragged to any size.

A tool can still be started on its own, which is what a repeated run wants:

```bash
python -m gsurf.tools.intersection
python -m gsurf.tools.fold_axes
```

Bare, each asks for what it needs in the same dialog. With arguments, each
takes them as below and skips the asking.

### Usage — plane on a DEM

```bash
python -m gsurf.tools.intersection dem.tif \
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

**And every entry in it is a switch.** Click one and what it names comes off
the map, click it again and it is back, the entry greyed while it is off. Over
its categories each layer has an entry of its own, in bold, and that one click
takes the whole layer: without it a backdrop of twenty units costs twenty-one
clicks to clear, and clearing it is the thing one actually does. Bold reads as
"a layer" down the whole legend and plain as "a category inside the one
above", which is otherwise not said at all — a run of unit codes does not
announce where one layer's entries end. The categories past the twelfth are
listed as `+N more` and switch together, or they would be unreachable.

`Show all categories`, in the same box, is the way back, and is enabled only
when there is something to bring back. It is not a convenience: hide the
legend with a category switched off and there is nothing left to click, which
without that button is a dead end.

`--settings file.json` reopens a saved orientation. Explicit arguments win over
the file, so `--settings x.json --z 900` is the saved plane at a new elevation.
Files written before the interface changed language still load: the Italian
keys are read as a fallback.

### Usage — fold axes

```bash
python -m gsurf.tools.fold_axes attitudes.gpkg:giaciture \
    --dip-dir Immersione --dip Inclinazione \
    --radius 2000 --dem dem.tif
```

The attitude layer is the only thing required, and it is what the session is
built on: with no DEM the projection and the extent come from the layer itself.
A DEM, if given, is backdrop and nothing else — this calculation never reads an
elevation. `--strike-rhr` reads the azimuth field as a right-hand-rule strike
instead of a dip direction.

Naming the layer and saying what its columns mean are two things, and either
can be left out — whatever is missing is asked for, with the rest filled in
from what was given. Asked, the two angle fields are offered as the layer's
numeric columns and guessed from their names first: on a CARG sheet
`immersione` and `inclinazione` are already selected when the dialog opens. A
field called `strike` also switches the convention, since that is better
evidence of what it holds than the default is. Nothing is guessed silently —
the guess is a selection you can see and change.

Drag the circle across the map. The stereonet follows it, showing the poles of
the bedding inside, the best-fit girdle and the axis they turn about; the panel
gives that axis, Woodcock's K and C, and how many attitudes it came from.

**The net has a window of its own**, floating over the map rather than wedged
into the top of the panel at whatever width the panel allows. It closes by its
own X and reopens from the button in the panel, which is bound to that same
action so the two cannot fall out of step; drag it against the panel and it
docks there instead, if that is where you want it. Closed, it is not redrawn
at all — the frame cost in the status bar goes on measuring what is really on
screen — and on reopening it catches up to the window you are in, not the one
you closed it on.

**The axis is only an axis if the poles form a girdle.** Above K = 1 they
cluster instead, which is a homocline, and the minimum eigenvector of a cluster
is the least determined direction in the data rather than a fold axis. On the
1757 CARG attitudes of the Potenza-Irsina sheet, three windows in four fail
that test — which is the reason the gate exists rather than an argument against
it. `--min-points`, `--max-k` set the thresholds, the panel changes them while
you work, and both they and the verdict go into the saved JSON, because an
answer recorded without the choice that produced it cannot be checked.

A refused axis is drawn in grey rather than hidden. Hiding it would answer "is
this a fold?" by showing nothing, which reads the same as an empty window;
greyed, you watch it turn colour as the window crosses a hinge.

Read the same place at three radii and you see what the window is for. At
590000/4500000 on that sheet: at 1 km ten attitudes and a cluster, at 2 km
312/13 from 38 attitudes with K = 0.59, at 5 km 312/19 from 162 — an axis that
survives a change of scale is a structure, one that does not is a coincidence.

**The grid** asks the same question everywhere at once: a window at every node
of a square grid over the attitudes, `--step` apart. The ticks are drawn at the
cells that pass, coloured by plunge — the plunge is half of what an axis is,
and the length of a bar cannot carry it. `Export grid` writes the cells that
hold data, refused ones included, with their trend in both norths, K, C, n, the
verdict and the thresholds that produced it.

**A field of axes is what made the backdrop worth switching off.** The ticks
are on top by zorder, and twenty tints of geology under them win anyway. The
same *Legend* box is in this panel now — `beside`, `inside`, `hidden`, with
`--legend` for where it starts; the three placements had existed here all
along, but only as that flag, which cannot be changed once the map is open.
Its entries switch what they name, layer by layer or category by category, as
above. The attitudes come off from their own entry too, and so do the ones
inside the window: on a dense survey those dots are what the field has to be
read through. The axis and the window circle do not, being what the hand is
steering — one dragged invisible is worse than one in the way.

**A grid overlaps itself, and the panel says by how much.** A circular window
of radius R laid down every S metres covers πR²/S² cells, so at r = 2000 with a
500 m step every attitude falls into fifty of them and neighbouring cells share
nine tenths of their data. The two controls sit in different groups and their
ratio is never a thing you set, so a field of a thousand axes reads as a
thousand observations when it carries perhaps fifty windows' worth. The line
under the step box names both numbers — `each attitude in ~50 cells; ~50
windows would tile the area` — and they multiply back to the cell count, which
is the point: a denser step buys resolution in the picture and no further
information under it. The same line follows the field into the status bar,
because the count of axes is what gets quoted.

The tiling number is geometry, not statistics. Windows that fail to overlap
are still not independent observations: a fold is continuous, and two windows
that do not touch can be looking at the same one.

The step also decides the cost, so the cost of the step currently typed is
shown beside it, before the button rather than after. On the 1757 CARG
attitudes:

| step | cells | with data | fold axes | time |
|---|---|---|---|---|
| 2000 m | 312 | 246 | 61 | 0.2 s |
| 1000 m | 1104 | 932 | 241 | 0.7 s |
| 500 m | 4140 | 3556 | 948 | 2.6 s |
| 250 m | 16289 | 14025 | 3762 | 9.6 s |

**Moving a threshold re-decides an existing grid without recomputing a single
tensor** — 3 to 7 ms for the whole field, against seconds to build it. K, C and
the count are what the gate reads and they are already there, which is the
payoff of keeping a field as arrays rather than as a picture. On that sheet,
sweeping K from 0.5 to 2.5 takes the field from 369 axes to 2049 out of 3556
occupied cells; how much of a map depends on where a threshold was put is not a
question you can answer by recomputing it four times, and it is not one to
leave unasked.

For the same reason the tensor is computed wherever it is defined — from two
poles up — and not from `--min-points`. Stopping at today's minimum would make
lowering it later impossible without recomputing, silently.

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
file that goes missing first. A fold axis is written the same way: true
azimuth, grid azimuth and the convergence between them.

**A horizontal bed has no dip direction.** The fold-axis tool reads the dip first
and only then the azimuth, because the dip is what decides whether the azimuth
means anything: at zero dip the field is not read at all. That is not
pedantry — the CARG sheets write 999 there, and a reader that took it at face
value would drop every horizontal bed on the map, or worse, keep it as a
bearing. Whatever is dropped and why is printed on the way in, never silently.

**And 360 is north.** The admissible range is closed at both ends, not the
half-open one a normalised azimuth lives in: on the Marsico Nuovo sheet ten
attitudes are written 360 and not one is written 0, so a half-open rule threw
away every north-dipping bed there and called them errors. It cost nothing on
the Potenza-Irsina sheet, which happens to contain no 360 at all — which is the
argument for trying a reader on a second survey before believing it.

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

Fold axes are not a kernel problem. Dragging the window across the 1757 CARG
attitudes costs 4.5 ms a frame, of which the search is 0.1 ms and the
orientation tensor 0.8; the rest is drawing two canvases. The tensor is
geogst's, called once per frame — 0.39 ms for a window of 20 poles, 0.99 for
68, 4.2 for 313 — which is affordable at a frame and is why there is no second
copy of that mathematics here.

The grid was expected to be where that stopped being true, and it is not. A
16289-cell field takes 9.6 s, of which geogst is 6.0 and the search 0.9: slow
enough to want a progress bar and a Stop button, nowhere near slow enough to
justify a second implementation of the orientation tensor. A vectorised one
still belongs upstream in geogst if it is ever written, but nothing here is
waiting for it.

The search stayed a plain scan for the same reason. A KD-tree does the 16289
windows in 0.046 s against 0.89 — nineteen times faster, and 13% of a field —
which does not buy a dependency. It would if the tensor stopped dominating,
which is the opposite of what happened.

### Related

- [misah](https://gitlab.com/mauroalberti/misah) — the Rust kernels
- [geogst](https://gitlab.com/mauroalberti/geogst) — types, CRS, orientation
  statistics and plots
- [qgSurf](https://gitlab.com/mauroalberti/qgSurf) — the QGIS plug-in, whose
  plane/DEM intersection this replaces with an interactive one
