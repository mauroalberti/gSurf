"""
The trace editor: a file in this format, read against the map and written in
its own words.

The other two ways into gstruct are readers. A `.gstruct` opens in the traces
slot like a layer, and the section panel writes a curation *over* one -- a file
of differences, laid on a source it never touches. This is the third thing, and
the one the format was for: the file itself, open, with what it says drawn along
the ground it says it about.

**What a text editor cannot show you.** Precedence in this format is a
computation over several lines at once -- a refusal beats a measurement, a
measurement within reach beats a fit, a fit beats a measurement further off --
so which line is winning at a given metre, and where along the fault the winner
changes, is not readable by looking at them. `attitude_at` answers that at one
place; the band at the top of the panel asks it everywhere. On F0055 of
`merid_faults` it shows a fit holding for 3177 m and a compass reading taking
over for the last 354, with the dip stepping from 31 to 35 where they meet. That
is one line of the file beating another, and neither line says so.

**And what two anchors enclose.** A picked anchor is a coordinate; what a `span`
or a `fit` decides is the ground between two of them. Two shift-clicks put two
coordinates on a line and nothing showed the stretch they made, so the thing
`Apply` was being asked about was on screen only as a pair of numbers. It is
drawn now, as a pale band under the trace in the one colour this map had left,
and it follows the caret rather than the selection -- click into a line already
in the file and it lights what that line covers. Read off the text and not off
the model, because the model does not have the half-written line in it and the
half-written line is what `Apply` will be handed.

**A band under, and not a line over**, which took two tries. Dashed and on top
it was drawn across the orange highlight, and what showed through the gaps was
the selection: the claim read as a purple-and-orange stripe along the trace,
which is a texture. A texture is a thing a line can be; an extent is what this
is. And **the words go on the gesture, never on the keystroke** -- `+ fit` and a
shift-click each report what they claimed, and the caret moving does not, because
the status bar is one line and a sentence written on every change overwrites
whichever sentence was answering the last thing somebody pressed.

The case that argues for it is the pair written the wrong way round. `covers` is
`s0 <= s <= s1`, so such a line parses, applies, and sits in the file looking
like a decision while holding over no part of the trace at all. There is nothing
to draw for it -- an empty highlight is what no ground looks like -- so that one
is said in words.

**And the plane that would make that stretch true.** `fit off the DEM` reads the
trace and answers, and on most traces its answer is nothing: 27 of 185 attempts
pass the gate on the AOI, and three of the four synthetic traces in the check
carry a fit only because they were built to. The gate is right to be that quiet
-- a plane through a straight trace is arbitrary and not merely imprecise -- but
a curator looking at a bend that carries nothing still has the topography in
front of them, and the other tool in this program has been steering a plane
across it by hand since before this one existed. What that tool has never had is
anything to aim at: the point goes wherever you click, and the answer is read out
loud into a notebook.

Here the fault is on screen, the stretch being decided is drawn under it, and the
file that wants the number is open beside it. So the plane hangs at **the middle
of the claim** -- not an end, where it would be exactly right and free to swing
away over the rest, which is the error this exists to show -- at the elevation
the DEM has there, because the trace is a contact somebody walked. Turn the dial
until the cut runs along the trace, and the button writes that attitude into the
line. On an `attitude` the pin is the anchor instead, which is the same rule read
at a place, and the loop also runs backwards: click into a `fit` somebody
computed and the dial shows it cutting the ground it was computed over.
`conflicts.py` asks whether a plane agrees with the ground arithmetically. This
asks it by looking.

**Where it goes wrong is where the geology says it should.** The cut passes
through the pin within half a cell -- 1.5 m over four attitudes on the check's
5 m DEM -- until the plane lies down on the slope, and then it wanders: 6 m at
ten degrees from the ground's own attitude, 23 m at under three. That is not
slack to be tightened. Two nearly parallel planes barely determine the line they
share, and a plane nearly parallel to the hillside is exactly the case FORMAT.md
writes `drape` for -- a fit reproducing the topography is not evidence about the
fault. The number says so in an attribute; this says so by making the curves
unsteerable.

**Three windows.** The map is the tool; the panel and the net are windows beside
it. The panel used to be one frame split down the middle with the map, and a
splitter cannot be dragged across a screen boundary, so the map could not be
made bigger without making the table smaller and neither could be moved -- while
the arrangement this tool is for is a monitor of map with the file open next to
it. The net was a dock and is a window for the same reason read off a different
measurement: a dock is as wide as the map can spare, which was 276 px, and 276
px of widget is a stereonet 268 px across. All three are parented to the map and
the group is `gsurf.windows`, the section tool's; what that costs here is two
shortcuts and a status bar, and `_build_shortcuts` and `say` are those.

**Finding the one to open.** 45 of the 393 faults of `merid_faults` carry a
plane; the other 348 are mapped contacts nobody has read one off yet. So the
first question this tool has to answer is which forty-five, and it answers it
three times over: the table sorts on what is written, the map draws a trace
carrying something firmly and one carrying nothing faintly, and every station in
the file has its dot whether or not its fault is the one open. The first two go
through `carries`, so they can never disagree; the third is finer than either,
because a fault carrying a plane is a line and a station is a place on it. A row
picked in the table then brings its trace into view -- the one thing a highlight
cannot do is say where to look, and on an AOI-wide framing a 1 km fault is three
pixels of orange somewhere.

**What a dot cannot say about itself.** A station dot is drawn on the trace, at
the progressive the anchor gives; the reading was taken wherever somebody stood,
and `off` is how far apart those two are. Over the 23 measurements of
`merid_faults` it runs from 0.0 to 69.7 m -- inside the width of the line at
1:25000 and a visible lie at 1:5000, with nothing to tell the two apart by
looking. Nor does the dot show that the curation has *refused* it: a rejected
measurement stays drawn, because somebody did stand there, and until now the band
in the panel was the only thing that said it does not hold. Resting on the dot
says both, which is why the hover exists and why what it says is text: `off` is a
number and a refusal is a word, and neither is a picture.

**All of them, and the selected fault's larger.** The dots were the selection's
alone at first, which made the map answer "where has anything been read" one
fault at a time -- 393 selections to see 23 dots. They are in the background now,
with the traces and for the traces' reason: they change when a block is applied
and at no other time. What separates the fault under the hand is size and not
hue, because every hue on this map already means something and the orange
highlight is saying which fault it is anyway. The cost of the change is that a
dot is now something to aim at that is not on the trace being edited, which
`pick` answers for.

**And the one thing text cannot say.** This argument was first made as a reason
not to draw a net at all -- one measurement per station, no population, and a
single pole says less than `145/35` written out. The first half is right and the
conclusion did not follow. A pole is a population's way of drawing a plane; a
*great circle* is a plane, and a striation drawn on it sits somewhere along it,
and where along it is the difference between a fault that moved down its dip and
one that moved along its strike. No pair of numbers shows that, and the net shows
it without arithmetic. The pole was drawn there at first and is not, because the
markers inside the circle are the striae and a pole is a mark inside it that is
not one.

**Whose planes the net is showing.** The selected fault's, all of them, and it
was one -- whichever the cursor was resting on. One is what the data looked like
from the side of the attitudes: 23 readings over 393 faults, no station repeated,
so a net per fault and a net per station were the same picture 17 times out of
20. They are not the same picture once the fits are on it. Counting both, seven
faults carry two planes or more and six carry a reading *and* a fit, and the
question those pose is agreement -- whether two planes are the same surface --
which is a question about a pair and cannot be asked one circle at a time.
F0074 is the case that settles it: 135/30 with a compass, an `exposed-facet` fit
at 141/29, and a `trace-dem` fit at 221/10 carrying
`caveat=immersione non vincolata dalla traccia`. Two circles nearly coincident
and a third across the net is that caveat, drawn.

So the net follows the selection and the cursor points within it: resting on a
station dot draws that station's circle again, thicker, with the rest dimmed.
Which also ends an asymmetry that needed explaining -- the net used to be filled
by the cursor and emptied by a change of trace, and the title had to carry whose
plane it was because nothing else did. The title still names the fault. It no
longer changes while the hand moves.

There are none. `merid_faults` has zero `lineation` records in it, and three of
its 23 attitudes mention striae in an Italian note -- `lineazione N080°` at S20,
`lineazione N075°` at S19, and at S4 two pitches whose values are on a paper
sheet. A trend alone would be enough, a striation lying in the plane it was read
on, so the plunge follows: 18.3 deg and 30.7 deg for those two. Turning prose
into a record is curation, and it is not something to do from inside a drawing
routine -- so the net asks `Structure.lineations` and currently draws nothing,
and the day somebody writes one it appears.

**What it does not rewrite.** Saving replaces the lines of the structure that
was edited and leaves every other byte alone. Not fastidiousness -- measured:
`curation.gstruct` through a load and a dump comes back without the ten lines of
comment in it, because comments are not in the model and a writer can only write
what it has, and those ten lines are the argument for why five thrusts are
`exposed`. A Save that deletes the geologist's reasoning is not a Save. See
`curation.Document`, which is where the splice lives and why.

**Two coordinate systems, and one place each crosses.** The model stays in the
file's own projection, because the model is the file: the progressives
`attitude_at` reasons over are metres on the ruler the file was written with.
The map is in the session's. So the paths are transformed on the way to being
drawn, and a picked anchor is transformed back on the way to being written, and
those are the only two crossings there are.

**Editing is textual, which is the format's own shape.** A correction here is a
line added, never a line changed: the last span covering a progressive wins, so
the way to say "not on this stretch after all" is to say it, under what was said
before. A panel of widgets would have had to invent an order of operations the
format already has. What the box is given is the block exactly as it stands in
the file, path and all -- median six vertices over the 393 faults, longest 35,
so there is nothing worth hiding and nothing hidden.
"""

from __future__ import annotations

import textwrap
from collections import Counter

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PyQt6 import QtCore, QtGui, QtWidgets

from gsurf.curation import (
    DEFAULT_MAX_GAP,
    PROVENANCE,
    SUFFIX,
    UNCONSTRAINED,
    Document,
    anchor_of,
    degrees_not_metres,
    interval_of,
    is_gstruct,
    nearest_structure,
    place_on,
    plane_of,
    point_on,
    provenance_of,
    stretch,
    with_attrs,
    with_plane,
)
from gsurf.convergence import MeridianConvergence
from gsurf.fits import AT_THE_END, as_line, dem_refusal, fits_along, gate_for
from gsurf.planes import (
    FROM_STEERED,
    broken_path,
    gaps_on,
    ground_on,
    laid_on,
    side_for,
    walked,
)
from gsurf.mapview import LegendControls, MapView, fit_to_screen
from gsurf.stereonet import StereonetView
from gsurf.traces import draped_length
from gsurf.windows import SatelliteWindow, WindowGroup

# The file is the subject, so it is the one thing this cannot run without. The
# DEM is optional and the tool is usable without it -- the traces are geometry
# and draw at any scale -- but a fault with no topography under it is a line on
# a white field, and where it runs is half of why it was drawn there.
WANTS = dict(
    traces="required",
    dem="optional",
    polygons="optional",
    lines="optional",
)

# And the slot takes one kind of file, which no other tool's does. A mapped
# layer in `traces` is a perfectly good answer to the sections and no answer at
# all here: there is nothing in a layer to edit, and every part of this window
# reads text. Said to the dialog rather than only at the door, so the choice is
# never offered -- `build` still refuses, for the ways in that do not pass
# through a dialog.
ONLY = dict(traces=SUFFIX)

# The five answers `attitude_at` can give, as colours. Measured and computed are
# different hues rather than different shades, because the distinction between
# them is the one the format exists to keep; far-off measurement is the pale
# version of measurement, because that is what it is. Refused is the only red on
# the panel, and absent is the colour of nothing.
PROVENANCE_TINT = {
    "misurata": "#1b7837",
    "misurata-lontana": "#a6dba0",
    "fit": "#2166ac",
    "rifiutata": "#b2182b",
    "assente": "#e4e4e4",
}

# How the `use` axis paints, it being the one axis that decides rather than
# describes. The others are grey: they are notes about the contact, and colouring
# them would put them in competition with the band above for the eye.
USE_TINT = {"rejected": "#b2182b", "accepted": "#1b7837", "unknown": "#9a9a9a"}
SPAN_TINT = "#6a7f95"

# The stretch the line under the caret claims, which is the one thing drawn on
# this map that is not in the file yet. Purple because every other hue here is
# already spoken for and the nearest free one would lie: green is a measurement,
# red a refusal, blue a fit, orange the selection. Kept beside them rather than
# inlined twice, since the artist and the legend entry have to be the same
# colour or the legend is a caption for something else.
CLAIMED_TINT = "#762a83"

# And wide and pale, because it is a band and not a line. Wider than the
# selection's 2.6 and the refusal's 3.4 so that it shows on both sides of them
# rather than competing with either, and `butt` caps so the ends fall where the
# anchors are: a round cap puts half a linewidth of claim past the coordinate
# somebody picked, which at this width is metres of trace nobody asked for.
CLAIMED_WIDTH = 8.0
CLAIMED_ALPHA = 0.45

# And where the plane being steered cuts the topography, which is the second
# thing on this map that is not in the file. The same purple, and that is now
# the rule here rather than a coincidence: everything drawn that nobody has
# applied yet is purple, and the two are told apart by weight, as the two
# weights of trace are. The band is the ground being claimed; the line is what
# is being claimed about it.
#
# Thin, and over everything rather than under. A match is the intersection
# running along the trace, so at the moment the answer is right the two are on
# top of each other -- under the 2.6 of the highlight it would vanish exactly
# then, and "hidden because it agrees" is not distinguishable by eye from "not
# computed". At 1.2 over 2.6 a match reads as a purple core down the orange.
CUTTING_WIDTH = 1.2

# Where the plane is pinned, drawn because the picture is meaningless without
# it: an intersection is a plane *through a point*, and every curve on screen
# turns about that one. Hollow, so the trace under it stays readable.
PIN_SIZE = 9

# And how near the cut runs to the trace, metre by metre along it. The widest
# thing on the map and the lowest, with the claimed band's 8 sitting inside it:
# the two are meant to be compared -- the gesture is to look at where the cut
# agrees and pull the claim onto it -- and nesting them makes "the claim is
# where the agreement is" a shape rather than two colours to hold in the head.
#
# Round caps here where the claimed band has butt ones, and the difference is
# not inconsistency. Butt caps are there because that band's ends are
# coordinates somebody picked, and half a linewidth past one is metres of trace
# nobody asked for. This band's ends are wherever the DEM window stops, which is
# nobody's decision, while its *interior* is hundreds of segments a pixel and a
# half long that have to read as one continuous strip.
AGREEING_WIDTH = 14.0
AGREEING_ALPHA = 0.55

# And drawn in steps rather than as a continuous ramp, which is two decisions
# that happen to be the same one.
#
# It is what the eye wants. The band is read for where the agreement *stops*,
# and an edge between two tints is a place, where a gradient is a feeling.
# Because `Gaps.close` is linear in log10 over the decade above a cell, four
# equal steps of it are four equal factors of distance -- 1, 1.8, 3.2, 5.6 and
# 10 cells -- so the scale can be said in words.
#
# And it is what the frame wants, which is the same measurement `broken_path`
# was written for. One path per sample costs about 7 microseconds whatever is
# in it: 536 of them across this window was 3.9 ms a frame, three times the
# whole blit without them. Quantising lets consecutive samples of the same step
# be drawn as one polyline, and on real ground the answer changes far less
# often than every five metres.
AGREEING_LEVELS = 4


# What the provenance band is sampled at. The band's edges are therefore good to
# one four-hundredth of the trace -- nine metres on a 3.5 km fault -- which is a
# picture of where the answer changes and not a measurement of it. The number
# where it changes exactly is in the file, as the anchor somebody wrote.
SAMPLES = 400

# And what the table's one-word summary of the same computation is sampled at.
# Far coarser, because it is 393 sweeps and not one: at this rate the whole
# column is 39 ms, against 250 at the band's own rate. What it buys is a
# fraction good to a percent and a half, which is the precision of a word.
HOLDS_SAMPLES = 64

# How much of a trace has to be on the DEM before its `length_3d` is quoted as
# the trace's own. Not 100%: the draping walks a part in a whole number of steps
# and the last one lands short of the end, so an exact demand would mark every
# trace in the file. At the 5 m cell that slack is metres on a kilometre, and
# what is left over the line is the two traces of `merid_faults` that genuinely
# run off the edge of the DTM.
WHOLE_TRACE = 0.97

# And where the column sorts the traces that have no DEM under them. Below every
# real length rather than at zero, which is a length a trace could have.
UNMEASURED = -1.0

# How a trace is drawn, by whether anything has been read off it. The difference
# is weight and darkness rather than hue, because every hue on this map already
# means something -- orange is the selection, red a refusal, green a measurement
# -- and a trace carrying only a compass reading is not a fit and must not be
# painted as one. It also puts the 348 that carry nothing where they belong:
# still on the map, no longer competing for the eye.
CARRYING_STYLE = dict(colors="#1a1a1a", linewidths=1.5)
BARE_STYLE = dict(colors="#8a8a8a", linewidths=0.7)

# And how big a station dot is, on the selected fault and anywhere else. Size and
# not hue, for the reason the trace weights are weight: a dot on a fault nobody is
# working on is still a place somebody stood and read a plane, which is what the
# green says, and the only thing that separates it from the others is that it is
# not the one under the hand. Which the orange highlight is already saying.
STATION_SIZE = 5
OTHER_STATION_SIZE = 3.5

# Room around a trace a made framing leaves, and the smallest window it will
# make. Both are in the docstring of `framing_for`, which is where the numbers
# they are measured against are.
FRAME_MARGIN = 0.25
FRAME_MIN_SPAN_M = 1200.0

# How long the framing waits before it moves. Arrow-keying down the table is one
# selection per keystroke, and each framing is a full redraw plus a hillshade
# reread -- tens to hundreds of milliseconds, paid for a view nobody looked at.
# Deferred like this a run of them costs one move, which is the same bargain
# `MapView.schedule_shade_refresh` strikes and for the same reason.
FRAME_DELAY_MS = 140

# How near the cursor has to come, in screen pixels, for a click to have been
# aimed at a trace. In pixels and not in metres so that it means the same thing
# at every zoom.
PICK_RADIUS_PX = 14

# And for the cursor to be resting on a station dot rather than near one. Tied to
# the marker, which is 5 points across: a threshold smaller than what is drawn
# would ask for an aim the dot does not reward, and a much larger one would claim
# ground where there is visibly nothing. Separate from `PICK_RADIUS_PX` because
# the two aim at different shapes -- a click is aimed at a line, which is long,
# and this at a dot, which is neither long nor wide.
HOVER_RADIUS_PX = 10

# Where a note in a tooltip is folded. Notes here run to 65 characters and a
# tooltip does not wrap plain text by itself, so a file with a paragraph in one
# would open a tooltip wider than the map it is covering.
TIP_WRAP = 64

# How near, along the trace, an unlabelled lineation has to be anchored to count
# as read at the same place as a plane. Anchors in these files are written to a
# hundredth of a metre, so this is not a tolerance for rounding: it is the width
# of one outcrop as somebody standing on it would place two readings. A lineation
# that carries `station=` is matched by that instead, and is not subject to this.
SAME_OUTCROP_M = 2.0

# What the net's window is called, and what it says with nothing on it. A title
# and not a label inside the figure, because an empty equal-area net is a circle
# with a grid in it and reads as a widget that has not loaded rather than as one
# with nothing to show.
#
# And what it says is which structure it is about, which is the whole of how the
# net says whose planes those are. It read `rest on a green dot` while the net was
# filled by the cursor; it is filled by the selection now, and 348 of the 393
# selections fill it with nothing, so an empty net has to name the fault that is
# empty or it cannot be told from a net that has not been pointed at yet.
NET_TITLE = "gSurf - stereonet"
NET_EMPTY_TITLE = f"{NET_TITLE} - nothing selected"

# How big the net opens. Its own window, so the map is no longer paying for it,
# and that is the whole of the difference: as a dock it had to stop at 276 px,
# because past that it came off the map's axes, and 276 px of widget is a
# primitive circle 268 px across. 420 px is 412. The circle follows the widget
# from there -- the figure size says where it starts, the window says where it
# stays, and unlike a floating dock's, that is remembered.
NET_FIGSIZE_IN = 3.6
NET_WINDOW_PX = (420, 440)

MAX_GAP_RANGE = (0.0, 20000.0)

# The three lines a curation is made of, with `*` where an anchor goes. They are
# not a form and do not save any typing worth counting: what they do is put the
# cursor somewhere known. An anchor is picked on the map and written at the
# cursor, so without a place to put it the first shift-click of a session lands
# wherever the box was last left -- which is the top of the block, where it
# reads as a word glued to `structure`.
#
# `use` and not one of the other two axes, because it is the one that decides:
# `certainty` and `exposure` describe the contact, and this is the tool for
# saying what holds.
TEMPLATES = (
    ("+ span", '  span use * * rejected reason=""'),
    ("+ attitude", "  attitude * plane 000/00 station= src=field"),
    ("+ fit", "  fit plane * * 000/00 from="),
)

PANEL_WIDTH_PX = 520

# What the panel comes up as, the first time and nothing being remembered. Taller
# than the 880 the whole tool used to ask for, because it is no longer sharing a
# frame with the map: the table wants rows and the band and the box want height,
# and both of those used to be cut to whatever was left beside a canvas.
PANEL_WINDOW_PX = (PANEL_WIDTH_PX, 940)

# What the map comes up as where the two do not fit side by side, and the least
# map worth putting a panel next to. Below that floor the map would be narrower
# than the panel, which is the wrong way round for a tool whose subject is where
# a fault runs, so the map takes the screen and the panel comes up over it.
MAP_WINDOW_PX = (1080, 880)
MAP_FLOOR_PX = 700

# The column on the map's own frame, which used to hold the legend's two
# controls and now holds the steering above them. Wide enough for a dial worth
# turning -- 132 px of dial is a degree every 1.2 px at the rim -- and no wider,
# because every pixel here comes off the map.
MAP_COLUMN_PX = 190

# What is left between them, and what is left for the map window's own frame.
# The gap is a gap; the allowance is a guess and has to be -- right after
# `setGeometry` the title bar does not exist yet, the window manager not having
# reparented the window, so its thickness cannot be measured before the window
# has to be placed. 40 px is the same allowance `WindowGroup.place_unremembered`
# already makes, and erring high costs a strip of desktop rather than the bottom
# of a window.
TILE_GAP_PX = 6
FRAME_ALLOWANCE_PX = 40

# What this tool's windows are kept under. `sections` is the other one.
SETTINGS_NAME = "editor"

# How much of the panel the table opens with, and the least it can be dragged
# to. The opening size is about eight rows, which is enough of a list to sort and
# read; the floor is four, which is enough to see that sorting did something.
TABLE_OPENING_PX = 240
TABLE_FLOOR_PX = 140

# Above this fraction of the trace, a bar has room for its own word in it.
LABEL_FRACTION = 0.14


def carries(structure):
    """
    Whether anything has been read off this trace: a measurement, or a fit.

    The one predicate the table's filter, the table's sorting and the map's two
    weights all go through, so that what the list calls carrying and what the map
    draws firmly are the same forty-five faults. A span is not counted: every one
    of the 393 has one, most of them saying `certainty` or `exposure`, and none
    of that is a plane.
    """

    return bool(structure.attitudes or structure.fits)


def _as_number(written):
    """
    A length an attribute claims to be, or nothing.

    Attributes are text and the format does not type them, so a file is free to
    put a word, an empty string or a range where a distance goes -- and a tooltip
    is the last place that should be the thing which raises.
    """

    try:
        return float(written)
    except (TypeError, ValueError):
        return None


def holds_along(structure, max_gap=DEFAULT_MAX_GAP, samples=HOLDS_SAMPLES):
    """
    The provenance covering most of the trace, as `(kind, fraction of it)`.

    `provenance_of` reduced to one word, which is what fits in a cell. It is a
    summary and reads as one: a fault measured over its first three hundred
    metres and fitted over the remaining three thousand says `fit`, because that
    is what most of it is. Where the answer changes is the band, and this column
    is how you find the trace worth opening the band on.

    `(None, 0.0)` where there is no ruler to sample along -- a structure with one
    vertex, or none.
    """

    sampled = provenance_of(structure, samples=samples, max_gap=max_gap)

    if not sampled:
        return None, 0.0

    kind, count = Counter(kind for _, _, _, kind in sampled).most_common(1)[0]

    return kind, count / len(sampled)


def framing_for(path, margin=FRAME_MARGIN, floor=FRAME_MIN_SPAN_M):
    """
    A view with one trace across the middle of it, as `extent` is ordered.

    The margin comes off the longer side and not off each one: a fault running
    due north has no width to take a fraction of, and a fraction of nothing is
    nothing. That is the reasoning `sections.framing_for` is built on; this is
    the same thing for a polyline rather than a two-point trace, and the floor is
    what the polyline adds to it.

    The floor is there because these lengths run over three orders of magnitude.
    The shortest fault in `merid_faults` is 12 m against a median of 1048 and a
    longest of 29641, and a window fitted to that short one is a screenful of
    hillshade with nothing in it to say where you are standing.
    """

    if not path:
        return None

    xs = [x for x, _ in path]
    ys = [y for _, y in path]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)

    span = max(x1 - x0, y1 - y0)
    room = max(margin * span, (floor - span) / 2.0, 0.0)

    return [x0 - room, x1 + room, y0 - room, y1 + room]


def runs_of(sampled):
    """
    The sampled provenance as contiguous `(s0, s1, kind, said)` stretches.

    One rectangle per stretch rather than one per sample: four hundred patches
    would draw the same picture and take four hundred times as long to, and the
    run is also the honest unit -- what is being shown is where the answer
    changes, and between two changes there is one answer.
    """

    runs = []

    for s, _, said, kind in sampled:
        if runs and runs[-1][2] == kind:
            runs[-1][1] = s
            continue

        if runs:
            runs[-1][1] = s

        runs.append([s, s, kind, said])

    return [tuple(run) for run in runs]


class _Ranked(QtWidgets.QTableWidgetItem):
    """
    A cell sorted on a number it does not show.

    `_number` can set the integer as the display role and let Qt compare it,
    because there the text *is* the number. Here it is not: `~1784` is a length
    with a caveat in front of it, and an empty cell is no length at all, and
    neither of those compares as the metres it stands for.
    """

    def __init__(self, text, key):
        super().__init__(text)
        self._key = float(key)

    def __lt__(self, other):
        return self._key < getattr(other, "_key", self._key)


class StructureTable(QtWidgets.QTableWidget):
    """
    Every structure in the file as a row: what is written on it, and what holds.

    This replaces a combo box that could be typed into, and that was all it could
    do. What was missing is the two things a list is for. Sorting: the faults
    carrying a plane are 45 of 393, and the way to them is a click on the `att`
    header, not a scroll. And a selection the map shares -- picking a row here and
    clicking a trace there are now one gesture, reported twice.

    Rows are never removed, only hidden, which is why the filter can no longer
    take away the structure being worked on. The combo was rebuilt on every
    filter change and had to notice when what was open had dropped out of it; a
    hidden row is still the open row, and there is nothing to notice.

    The sorting is what makes a row's position worth nothing, so the structure's
    index in the document travels in the ident cell rather than being the row
    number. `index_of` and `_row_of` are the two directions of that, and every
    write goes through `_hold_open` because a sort moves the current row out from
    under whatever the panel has on screen.
    """

    COLUMNS = ("ident", "length_2d", "length_3d", "att", "fit", "span", "holds")

    # Derived and not typed, because it is the one column written from two
    # places -- the fill and the reach dial -- and a number that had to be kept
    # in step with the tuple by hand is a number that would be tinting `span` the
    # first time a column was inserted.
    HOLDS_COLUMN = COLUMNS.index("holds")

    chosen = QtCore.pyqtSignal(int)
    framing_asked = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(0, len(self.COLUMNS), parent)

        self._filling = False
        self._open = None

        # What the topography came to, per structure index: `(metres, the metres
        # they cover)`, or absent where there was no DEM under the trace. Held
        # here rather than recomputed per row because it costs a raster read per
        # trace -- 1.1 s over the 393 of `merid_faults` -- and because nothing in
        # this window edits a path: the spans, fits and attitudes move, the line
        # they are written against does not.
        self._draped = {}

        self.setHorizontalHeaderLabels(self.COLUMNS)
        self.verticalHeader().setVisible(False)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.setSortingEnabled(True)
        self.setToolTip(
            "Click a header to sort: `length_2d` is the trace in plan and "
            "`length_3d` the same trace over the topography, `att` and `fit` "
            "are what has been read off it, `span` what has been said about it, "
            "`holds` the class covering most of it. Double-click a row to bring "
            "its trace into view."
        )

        # `Interactive` and widths fitted once per fill, rather than
        # `ResizeToContents` -- which is what the section panel's table uses and
        # what this one used first. Measured on the 393x6 cells of `merid_faults`,
        # inside this window rather than on a table by itself: rewriting them
        # costs 7.9 ms with this mode and 25.7 *seconds* with the header fitting
        # to contents, because that mode reflows every column on every cell
        # written. A table on its own does not show it -- with no laid-out
        # viewport the reflow never runs -- which is why the number above is from
        # the real window.
        header = self.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)

        # Said rather than left to Qt, which starts a header's indicator on
        # section 0 *descending* -- measured, not assumed. Enabling the sorting
        # then sorts by that, and a table filled in file order came out reversed:
        # F0395 on the first row, F0000 on the last, and nothing on screen saying
        # so. Ident ascending is the order the file is written in for anything the
        # import produced, which is what a reader opening it expects to see.
        header.setSortIndicator(0, QtCore.Qt.SortOrder.AscendingOrder)

        self.currentCellChanged.connect(self._on_current)
        self.doubleClicked.connect(
            lambda index: self.framing_asked.emit(self.index_of(index.row()))
        )

    # -- rows and indices --------------------------------------------------

    def index_of(self, row):
        """Which structure the row stands for, or None off the end."""

        item = self.item(row, 0)

        return None if item is None else item.data(QtCore.Qt.ItemDataRole.UserRole)

    def _row_of(self, index):
        """
        Which row a structure is on now, or -1.

        A scan and not a dictionary: 393 rows is microseconds, and a dictionary
        would have to be rebuilt on every sort by somebody who remembered to.
        """

        for row in range(self.rowCount()):
            if self.index_of(row) == index:
                return row

        return -1

    def point_at(self, index):
        """
        Puts the current row on a structure without saying anybody asked.

        The counterpart of the map holding its highlight: when the panel refuses
        to leave a block that was typed and not applied, this is what puts the
        table back on the one still on screen.
        """

        self._open = index
        row = self._row_of(index)

        if row < 0 or row == self.currentRow():
            return

        self._filling = True
        self.setCurrentCell(row, 0)
        self._filling = False

        self.scrollToItem(
            self.item(row, 0), QtWidgets.QAbstractItemView.ScrollHint.EnsureVisible
        )

    def _on_current(self, row, _column, previous, _previous_column):
        if self._filling or row < 0 or row == previous:
            return

        index = self.index_of(row)

        if index is not None:
            self.chosen.emit(index)

    # -- filling -----------------------------------------------------------

    def say_step(self, said):
        """What the `length_3d` column was measured at, on its own header."""

        header = self.horizontalHeaderItem(self.COLUMNS.index("length_3d"))

        if header is not None:
            header.setToolTip(said)

    def fill(self, structures, max_gap=DEFAULT_MAX_GAP, draped=None):
        """Every structure as a row, in the order the file has them."""

        self._draped = draped or {}

        with self._hold_open():
            self.setRowCount(len(structures))

            for index, structure in enumerate(structures):
                self._write_row(index, index, structure, max_gap)

        # Once, here, and not on the incremental writers: a column that resized
        # every time a row was rewritten would shift under the hand, and the
        # widths a whole file fits in are the right ones for one row of it.
        self.resizeColumnsToContents()

    def update_row(self, index, structure, max_gap=DEFAULT_MAX_GAP):
        """One row again, for a block that has just been applied."""

        row = self._row_of(index)

        if row < 0:
            return

        with self._hold_open():
            self._write_row(row, index, structure, max_gap)

    def refresh_holds(self, structures, max_gap=DEFAULT_MAX_GAP):
        """
        The `holds` column again, for a reach that has just changed.

        Only that column: the reach decides what holds and says nothing about
        what is written, so the tallies beside it are not in question.
        """

        with self._hold_open():
            for row in range(self.rowCount()):
                index = self.index_of(row)

                if index is not None:
                    self._holds(row, structures[index], max_gap)

    def set_filter(self, wanted, carrying_only, structures):
        """Hides the rows that do not match, and says how many are left."""

        wanted = (wanted or "").strip().lower()
        shown = 0

        for row in range(self.rowCount()):
            index = self.index_of(row)
            structure = None if index is None else structures[index]

            hidden = structure is None or (
                (carrying_only and not carries(structure))
                or bool(wanted and wanted not in (structure.ident or "").lower())
            )

            self.setRowHidden(row, hidden)
            shown += not hidden

        return shown

    # -- one row -----------------------------------------------------------

    def _hold_open(self):
        """
        Writes cells with the sorting off, and puts the current row back after.

        Both halves are needed. Writing into a sorted table reorders it as you
        write, so a fill would be laying rows down on ground that moves; and
        turning the sorting back on sorts, which moves the current row out from
        under whatever the panel has on screen -- the table would end up pointing
        at a different structure from the one whose text is in the box.
        """

        table = self

        class Held:
            def __enter__(self):
                table._filling = True
                table.setSortingEnabled(False)

            def __exit__(self, *_):
                table.setSortingEnabled(True)
                table._filling = False

                if table._open is not None:
                    table.point_at(table._open)

        return Held()

    def _write_row(self, row, index, structure, max_gap):
        ident = QtWidgets.QTableWidgetItem(structure.ident or "")
        ident.setData(QtCore.Qt.ItemDataRole.UserRole, index)
        self.setItem(row, 0, ident)

        self._number(row, 1, round(structure.length))
        self._over_topography(row, 2, structure, self._draped.get(index))
        self._number(row, 3, len(structure.attitudes))
        self._number(row, 4, len(structure.fits))
        self._number(row, 5, len(structure.spans))
        self._holds(row, structure, max_gap)

    def _number(self, row, column, value):
        """
        A count or a length, sorted as the number it is.

        Set as text it would sort as text, and a column reading 9, 84, 1048 would
        come out 1048, 84, 9 -- which is the order that hides the long faults at
        the top of the very sort meant to find them.
        """

        item = QtWidgets.QTableWidgetItem()
        item.setData(QtCore.Qt.ItemDataRole.DisplayRole, int(value))
        item.setTextAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )

        self.setItem(row, column, item)

    def _over_topography(self, row, column, structure, draped):
        """
        The trace's length over the DEM, and how much of the trace that is.

        The `~` is the whole point of the cell. A trace that runs off the edge of
        the DEM is measured over the part that is on it, and that length printed
        beside a plan length measured over all of it comes out *shorter* --
        F0168 is 2547 m in plan and on the 5 m DTM for 70% of them, so its
        `length_3d` is the smaller number of the two. Without the mark that is a
        subtraction anybody would read as a bug in the draping; with it, it is
        the sentence "this is 70% of a trace", which is what it is.

        Empty and not zero where there is no DEM under the trace at all, because
        zero is a length and this is the absence of one. Empty sorts to the
        bottom either way round, by the key rather than by the text -- a column
        sorted on "" would put the unmeasured traces between 999 and 1000.
        """

        if draped is None:
            item = _Ranked("", UNMEASURED)
            item.setToolTip(
                "no DEM under this trace"
                if self._draped
                else "no DEM in this session, or not one these traces may be "
                     "sampled against"
            )
        else:
            metres, covered = draped
            fraction = covered / structure.length if structure.length else 0.0
            whole = fraction >= WHOLE_TRACE

            item = _Ranked(f"{round(metres)}" if whole else f"~{round(metres)}",
                           metres)
            item.setToolTip(
                f"{round(metres)} m over the topography"
                if whole
                else f"{round(metres)} m over the topography, but measured on "
                     f"{round(covered)} m of {round(structure.length)} -- "
                     f"{fraction:.0%} of the trace is on the DEM"
            )

        item.setTextAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )

        self.setItem(row, column, item)

    def _holds(self, row, structure, max_gap):
        """
        What holds over most of the trace, tinted as the band tints it.

        The same colours and the same white-on-dark rule as `ProvenanceView._band`,
        so that the column and the picture under it are read as the one thing they
        are, and a row of `assente` grey is a trace with nothing on it at a glance.
        """

        kind, fraction = holds_along(structure, max_gap)
        item = QtWidgets.QTableWidgetItem(
            "" if kind is None else f"{kind} {fraction:.0%}"
        )

        if kind is not None:
            item.setBackground(QtGui.QColor(PROVENANCE_TINT.get(kind, "#cccccc")))
            item.setForeground(
                QtGui.QColor("#6a6a6a" if kind == "assente" else "white")
            )

        self.setItem(row, self.HOLDS_COLUMN, item)


class ProvenanceView(QtWidgets.QWidget):
    """
    One structure along its own trace: what holds, what says so, and what it is.

    Three rows over a shared axis of metres. The band is `attitude_at` swept end
    to end. The lanes under it are the lines of the file that were competing to
    answer -- the measurements as ticks where they were taken, the fits as the
    intervals they were computed over, the axes as the stretches they cover. The
    bottom is the plane that won, which is what makes a precedence boundary
    visible as the thing it is: a step in the dip.

    The gap between the band and the lanes is where the reading is. A fit drawn
    in the lanes with `assente` over it in the band is a fit its own verdict
    threw out -- a straight trace does not constrain a dip -- and that is a
    common and confusing state to be in with nothing but the text in front of
    you.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.figure = Figure(figsize=(6, 3.8), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(230)

        # In a label and not in the figure: five entries would take a row of
        # the plot to explain what a row of coloured text explains beside it,
        # and the figure has three axes to fit already.
        key = QtWidgets.QLabel(
            " ".join(
                f'<span style="background:{PROVENANCE_TINT[kind]}; color:{PROVENANCE_TINT[kind]}">'
                f"&nbsp;&nbsp;&nbsp;</span>&nbsp;{kind}"
                for kind in PROVENANCE
            )
        )
        key.setTextFormat(QtCore.Qt.TextFormat.RichText)
        key.setStyleSheet("font-size: 10px;")
        key.setWordWrap(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(key)

    def show_structure(self, structure, max_gap=DEFAULT_MAX_GAP):
        """Redraws for one structure, or says why there is nothing to draw."""

        self.figure.clear()

        if structure is None or len(structure.path) < 2 or structure.length <= 0.0:
            self._nothing(
                "no path, and so no ruler to lay anything along"
                if structure is not None else "nothing selected"
            )
            return

        sampled = provenance_of(structure, samples=SAMPLES, max_gap=max_gap)

        if not sampled:
            self._nothing("nothing to sample along")
            return

        band, lanes, angles = self.figure.subplots(
            3, 1, sharex=True, height_ratios=[0.5, 2.0, 1.5]
        )

        length = structure.length

        self._band(band, runs_of(sampled), length)
        self._lanes(lanes, structure, length)
        self._angles(angles, sampled)

        angles.set_xlim(0.0, length)
        angles.set_xlabel("m along the trace")

        self.canvas.draw_idle()

    def _nothing(self, why):
        axes = self.figure.subplots()
        axes.text(0.5, 0.5, why, ha="center", va="center", color="#8a8a8a", fontsize=9)
        axes.set_axis_off()

        self.canvas.draw_idle()

    def _band(self, axes, runs, length):
        for s0, s1, kind, said in runs:
            axes.axvspan(s0, s1, color=PROVENANCE_TINT.get(kind, "#cccccc"), lw=0)

            if s1 - s0 > LABEL_FRACTION * length:
                # The detail and not the class: the class is the colour, and
                # `misurata:S26@245m` is the sentence -- which station, and how
                # far away it was when it won.
                axes.text(
                    (s0 + s1) / 2.0,
                    0.5,
                    said,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if kind != "assente" else "#6a6a6a",
                )

        axes.set_yticks([])
        axes.set_ylabel("holds", fontsize=8)

    def _lanes(self, axes, structure, length):
        axes.set_ylabel("written", fontsize=8)

        # The axes the file actually uses on this structure, in the order they
        # were written. Showing the three the vocabulary defines whether or not
        # they are there would be three empty rows saying nothing.
        axis_rows = []

        for span in structure.spans:
            if span.axis not in axis_rows:
                axis_rows.append(span.axis)

        rows = ["attitudes", "fits"] + axis_rows
        at = {name: -n for n, name in enumerate(rows)}

        for attitude in structure.attitudes:
            if attitude.s is None:
                continue

            axes.vlines(
                attitude.s, at["attitudes"] - 0.3, at["attitudes"] + 0.3,
                color="#1b7837", lw=1.8,
            )
            axes.text(
                attitude.s, at["attitudes"] + 0.36,
                " ".join(
                    part for part in (
                        attitude.attrs.get("station", ""),
                        "" if attitude.plane is None else str(attitude.plane),
                    ) if part
                ),
                ha="center", va="bottom", fontsize=7, color="#1b7837",
            )

        for fit in structure.fits:
            s0 = 0.0 if fit.s0 is None else fit.s0
            s1 = length if fit.s1 is None else fit.s1

            # The same test `attitude_at` makes, and the reason a fit can be
            # drawn here under an `assente` band: a plane read off a straight
            # trace is a number the arithmetic produced, not a measurement.
            counted = (
                fit.plane is not None
                and fit.attrs.get("verdict") != UNCONSTRAINED
            )

            axes.barh(
                at["fits"], max(s1 - s0, length * 0.002), left=s0, height=0.6,
                color=PROVENANCE_TINT["fit"] if counted else "#d6d6d6",
                edgecolor="#8a8a8a" if not counted else "none", lw=0.6,
            )

            said = fit.attrs.get("verdict") or fit.attrs.get("from") or "fit"

            if s1 - s0 > LABEL_FRACTION * length:
                axes.text(
                    (s0 + s1) / 2.0, at["fits"],
                    f"{fit.plane} {said}" if fit.plane is not None else said,
                    ha="center", va="center", fontsize=7,
                    color="white" if counted else "#4a4a4a",
                )

        stacked = {}

        for span in structure.spans:
            if span.s0 is None or span.axis not in at:
                continue

            # Later spans are drawn narrower and in front of earlier ones,
            # because that is what the rule does: the last span covering a
            # progressive is the one in force, and a correction in this format
            # is written by adding a line over the one being corrected. Drawn
            # flat they would be one bar with two words printed on top of each
            # other -- which is how this read before, on the two `exposure`
            # lines of F0055.
            layer = stacked.get(span.axis, 0)
            stacked[span.axis] = layer + 1

            middle = (span.s0 + span.s1) / 2.0
            tint = USE_TINT.get(span.value, SPAN_TINT) if span.axis == "use" else SPAN_TINT

            # Whether this is the line in force where it is widest. A proxy, and
            # said as one: a span can hold over part of itself and be shadowed
            # over the rest, and the picture of that is the stacking rather than
            # the tint. What the tint is for is the common case -- two lines
            # over one stretch, one of them answering.
            answering = structure.span_at(span.axis, middle) is span

            axes.barh(
                at[span.axis], max(span.s1 - span.s0, length * 0.002),
                left=span.s0, height=max(0.6 - 0.17 * layer, 0.16),
                color=tint, alpha=1.0 if answering else 0.35, zorder=3 + layer,
            )

            # Only the one that is winning. The shadowed line stays drawn,
            # because it is still in the file and taking it off would hide why
            # the winner is the winner -- but it is not what holds, and putting
            # its word on the chart would say that it was.
            if answering and span.s1 - span.s0 > LABEL_FRACTION * length:
                axes.text(
                    middle, at[span.axis], span.value, ha="center", va="center",
                    fontsize=7, color="white", zorder=12,
                )

        axes.set_ylim(-len(rows) + 0.4, 0.75)
        axes.set_yticks([at[name] for name in rows])
        axes.set_yticklabels(rows, fontsize=8)
        axes.tick_params(axis="y", length=0)

        for side in ("top", "right", "left"):
            axes.spines[side].set_visible(False)

    def _angles(self, axes, sampled):
        progressives = [s for s, _, _, _ in sampled]
        dips = [np.nan if plane is None else plane.dip for _, plane, _, _ in sampled]
        azimuths = [
            np.nan if plane is None else plane.dip_dir for _, plane, _, _ in sampled
        ]

        axes.step(progressives, dips, where="post", color="#333333", lw=1.4)
        axes.set_ylim(0, 90)
        axes.set_yticks([0, 30, 60, 90])
        axes.set_ylabel("dip", fontsize=8, color="#333333")

        # Two scales because they are two different measurements of one plane,
        # and drawing a dip direction on an axis that stops at 90 would fold
        # three quarters of the compass onto the top of the frame.
        twin = axes.twinx()
        twin.step(progressives, azimuths, where="post", color="#d95f02", lw=1.1, ls="--")
        twin.set_ylim(0, 360)
        twin.set_yticks([0, 90, 180, 270, 360])
        twin.set_ylabel("dip dir", fontsize=8, color="#d95f02")

        for one in (axes, twin):
            one.tick_params(labelsize=7)


class EditorPanel(QtWidgets.QWidget):
    """
    The structure being worked on: which one, what holds along it, and its lines.

    The buttons divide the way the file does. `Apply` settles a block -- it goes
    through the parser, so a block that would not read is refused with the
    parser's own words and the text stays on screen to be fixed. `Save` puts the
    document on disk. Nothing is written to disk until it is asked for, and
    nothing is applied to the model until it parses.
    """

    selected = QtCore.pyqtSignal(int)
    applied = QtCore.pyqtSignal(int)

    # Asked of the window, which is the only thing here that knows what a view
    # is. Separate from `selected` because the two are not the same question: a
    # click on the map selects without moving the view, and it would be a poor
    # map that jumped to the thing you had just pointed at.
    framing_asked = QtCore.pyqtSignal(int)

    # News from in here, for the bar the window owns. The window's own `say`
    # writes the status bar and the echo under the panel together, so this is
    # connected to that rather than to a label of its own -- what a fit came to
    # is read from the map as often as from the box.
    said = QtCore.pyqtSignal(str)

    # The stretch the line under the cursor claims, as `(s0, s1)` or None. Asked
    # of the window for `framing_asked`'s reason turned the other way round: the
    # panel is the only thing that knows which line is being worked on, and the
    # window is the only thing that knows what the ground looks like.
    covering = QtCore.pyqtSignal(object)

    # And the plane that line carries, as `(dip dir, dip)` or None. A separate
    # signal and a separate guard, because the two change apart: typing digits
    # into the plane moves this and not the stretch, and picking an anchor moves
    # the stretch and not this.
    holding = QtCore.pyqtSignal(object)

    def __init__(self, document, dem=None, crs=None, parent=None):
        super().__init__(parent)

        self.document = document
        self.index = None

        # The last stretch reported, as `(which structure, the interval)`, so
        # that moving the caret along one line does not re-emit it: every
        # emission costs the window a blit, and a caret crossing a
        # forty-character line would spend forty of them redrawing a picture
        # that did not change.
        #
        # The index is in the key and not only the interval, because two traces
        # can be claimed over the same pair of progressives -- `* *` on any two
        # of them starts at zero -- and then moving between them would report no
        # change and leave the stretch drawn on the one being left.
        self._covering = (None, None)

        # And the same for the plane, keyed the same way and for the same
        # reason: two traces can carry the same attitude, and moving between
        # them has to re-aim what the steering is hanging on.
        self._holding = (None, None)

        # The topography, and whether it may be sampled for these traces at all.
        # The refusal is a fact about the pair and not about the click, so it is
        # settled once here and shown as the disabled button's reason -- a button
        # that looks available and answers with a message box every time would be
        # offering something this session cannot do.
        self.dem = dem
        self.dem_said = None if dem is None else dem_refusal(dem, crs)

        # Built from the file's own CRS rather than taken off the session, which
        # the panel has never had: the convergence a fit is corrected by has to
        # be the one for the projection the coordinates on the line are written
        # in, and that is the `crs` this panel was opened against. See
        # `fits.fits_along`.
        self.convergence = MeridianConvergence(crs)

        # The gate, measured off every path in the file the first time a fit is
        # asked for, and the sentence about it said once. See `_gate`.
        self._gate_measured = None
        self._gate_said = False

        self.table = StructureTable()
        self.table.setMinimumHeight(TABLE_FLOOR_PX)
        self.table.chosen.connect(self._chosen)
        self.table.framing_asked.connect(self._asked_framing)

        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("find an ident")
        self.filter.setClearButtonEnabled(True)
        self.filter.setToolTip(
            "Narrows the table to the idents containing what is typed. It hides "
            "rows rather than removing them, so the structure being worked on "
            "stays the one being worked on."
        )
        self.filter.textChanged.connect(lambda _: self._apply_filter())

        self.carrying = QtWidgets.QCheckBox("only the ones carrying a plane")
        self.carrying.setToolTip(
            "45 of the 393 faults of merid_faults carry an attitude or a fit. "
            "The rest are mapped contacts nobody has read a plane off yet, and "
            "they are still editable -- this only shortens the list."
        )
        self.carrying.toggled.connect(lambda _: self._apply_filter())

        self.shown = QtWidgets.QLabel()
        self.shown.setStyleSheet("color: #6a6a6a; font-size: 11px;")

        self.frame_wanted = QtWidgets.QCheckBox("frame it")
        self.frame_wanted.setChecked(True)
        self.frame_wanted.setToolTip(
            "Bring the trace into view when it is picked from the table. A click "
            "on the map never moves the view -- you are already looking at what "
            "you clicked. A double-click on a row frames it whatever this says."
        )

        self.gap_spin = QtWidgets.QDoubleSpinBox()
        self.gap_spin.setRange(*MAX_GAP_RANGE)
        self.gap_spin.setSingleStep(50.0)
        self.gap_spin.setValue(DEFAULT_MAX_GAP)
        self.gap_spin.setSuffix(" m")
        self.gap_spin.setToolTip(
            "How far from where it was taken a measurement still answers. The "
            "same judgement the section panel's reach makes, from the other "
            "end: past this the fit under it takes over."
        )
        self.gap_spin.valueChanged.connect(lambda _: self._reach_changed())

        self.view = ProvenanceView()

        self.text = QtWidgets.QPlainTextEdit()
        self.text.setFont(QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.FixedFont
        ))
        self.text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.text.setTabStopDistance(28)
        self.text.setMinimumHeight(120)

        # Both, because a stretch changes under either gesture and neither
        # implies the other: typing an anchor moves the text without moving the
        # caret off the line, and clicking from one line to the next moves the
        # caret without touching a character.
        for moved in (self.text.cursorPositionChanged, self.text.textChanged):
            moved.connect(self._covering_changed)
            moved.connect(self._holding_changed)

        self.problem = QtWidgets.QLabel()
        self.problem.setWordWrap(True)
        self.problem.setStyleSheet("color: #b2182b; font-size: 11px;")
        self.problem.setVisible(False)

        self.apply_button = QtWidgets.QPushButton("Apply")

        # No shortcut of its own: the window owns both of this tool's shortcuts
        # and hands them to every window of the group, which is what makes them
        # work from the map as well. A second Ctrl+Return here would be an
        # ambiguous one -- see `EditorWindow._build_shortcuts`.
        self.apply_button.setToolTip(
            "Read the block back through the parser and put it in the document "
            "(Ctrl+Return). Nothing reaches the file until Save."
        )
        self.apply_button.clicked.connect(self.apply_block)

        self.revert_button = QtWidgets.QPushButton("Revert")
        self.revert_button.setToolTip("Put back the block as the document has it.")
        self.revert_button.clicked.connect(self._redraw)

        buttons = QtWidgets.QHBoxLayout()

        for label, template in TEMPLATES:
            adder = QtWidgets.QPushButton(label)
            adder.setToolTip(
                f"Write `{template.strip()}` above the path, with the first "
                f"anchor selected: shift-click the map to fill it in, and the "
                f"next one is selected in turn."
            )
            adder.clicked.connect(lambda _, line=template: self.add_line(line))
            buttons.addWidget(adder)

        # The fourth button, which is not a fourth template: the three above write
        # a line for somebody to finish and this one writes lines that are already
        # finished. In the same row because what comes out of it is the same kind
        # of thing -- lines in the box, kept by Apply and by nothing else.
        self.fit_button = QtWidgets.QPushButton("fit off the DEM")
        self.fit_button.clicked.connect(lambda: self.fit_off_dem())

        if self.dem is None:
            self.fit_button.setEnabled(False)
            self.fit_button.setToolTip(
                "No DEM in this session, and a plane read off the topography "
                "needs one. The slot is optional because the traces draw without "
                "it; this is the one thing here that does not."
            )
        elif self.dem_said:
            self.fit_button.setEnabled(False)
            self.fit_button.setToolTip(self.dem_said)
        else:
            self.fit_button.setToolTip(
                "Sweep a window along this trace and write a `fit` for every "
                "stretch whose plane the topography determines -- the import's "
                "own producer, on one trace at a time. A stretch too straight to "
                "carry a plane gets nothing, which is an answer. The lines go in "
                "the box: nothing is kept until Apply."
            )

        buttons.addWidget(self.fit_button)

        buttons.addStretch(1)
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.revert_button)

        heading = QtWidgets.QHBoxLayout()
        heading.addWidget(self.filter, stretch=1)
        heading.addWidget(self.carrying)
        heading.addWidget(self.frame_wanted)

        gap = QtWidgets.QHBoxLayout()
        gap.addWidget(QtWidgets.QLabel("a measurement answers for"))
        gap.addWidget(self.gap_spin)
        gap.addStretch(1)
        gap.addWidget(self.shown)

        finding = QtWidgets.QWidget()
        finding_layout = QtWidgets.QVBoxLayout(finding)
        finding_layout.setContentsMargins(0, 0, 0, 0)
        finding_layout.setSpacing(4)
        finding_layout.addLayout(heading)
        finding_layout.addWidget(self.table, stretch=1)

        working = QtWidgets.QWidget()
        working_layout = QtWidgets.QVBoxLayout(working)
        working_layout.setContentsMargins(0, 0, 0, 0)
        working_layout.addLayout(gap)
        working_layout.addWidget(self.view, stretch=3)
        working_layout.addWidget(self.text, stretch=2)
        working_layout.addWidget(self.problem)
        working_layout.addLayout(buttons)

        # In a splitter because the two halves are wanted at different times and
        # the panel is not tall enough for both at their best: hunting for the
        # trace worth opening wants rows, and working on the one that is open
        # wants the band and the box. Neither is allowed to collapse to nothing --
        # a table dragged shut has no way back that looks like one.
        #
        # Opened at sizes rather than at stretch factors, which is what this had
        # first: a stretch factor divides what is left over after the size hints
        # are met, and a table's hint asks for almost nothing -- the panel came up
        # with 70 pixels of table in it, which is one row, which is a list you
        # cannot read to find anything.
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        split.addWidget(finding)
        split.addWidget(working)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        split.setChildrenCollapsible(False)
        split.setSizes([TABLE_OPENING_PX, 3 * TABLE_OPENING_PX])

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(split)

        self.table.say_step(self._step_said())
        self.table.fill(
            self.document.dataset.structures, self.max_gap, self._drape_all()
        )
        self._apply_filter()

        if self.document.dataset.structures:
            # Pointed at rather than chosen: the window emits nothing yet -- it
            # has not connected to this panel -- and it opens the first structure
            # itself once it has a map to draw it on.
            self.show_index(0)

    # -- the list ----------------------------------------------------------

    def _step(self):
        """What the traces are walked at over the DEM, or None if they cannot be."""

        if self.dem is None or self.dem_said:
            return None

        return max(self.dem.res_x, self.dem.res_y)

    def _step_said(self):
        """
        The `length_3d` header's tooltip: the step, or why there is no column.

        The step is said and not assumed, because the number depends on it and
        does not converge -- see `traces.draped_length`. A reader comparing a
        3D length here against one computed elsewhere is comparing two different
        measurements unless both say what they were walked at.
        """

        step = self._step()

        if step is None:
            return (
                self.dem_said
                or "No DEM in this session, so there is nothing to hang the "
                   "traces on and the column is empty."
            )

        return (
            f"The trace over the topography, walked at {step:g} m -- the DEM's "
            f"own cell. The number depends on that step and gets no truer below "
            f"the cell: sampling finer only counts the same cells' edges as "
            f"relief. A `~` means the trace runs off the DEM and the length is "
            f"over the part that is on it."
        )

    def _drape_all(self):
        """
        Every path over the DEM, once, as `{index: (metres, metres covered)}`.

        Once and up front rather than per row, because it is a raster read per
        trace: 1.13 s over the 393 of `merid_faults` against the 5 m DTM, which
        is a wait at the door and would be a stutter on every sort, filter and
        Apply if it were left to `_write_row`. Nothing in this window edits a
        path, so the answer cannot go stale while the window is open.

        The traces with no DEM under them are absent rather than present as
        `None`: 13 of those 393 are, and the column reads the difference between
        a missing key and an empty dictionary -- the first is this trace, the
        second is this session.
        """

        step = self._step()

        if step is None:
            return {}

        drapes = {}

        QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor)
        )
        try:
            for index, structure in enumerate(self.document.dataset.structures):
                metres, covered = draped_length(structure.path, self.dem, step=step)

                if metres is not None:
                    drapes[index] = (metres, covered)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        return drapes

    def _apply_filter(self):
        """
        Narrows the table, and says how much of the file is on show.

        The count is the point of the label: `45 of 393` is the sentence the
        `carrying` box exists to produce, and without it the box looks like it
        has thrown most of the file away.
        """

        structures = self.document.dataset.structures
        shown = self.table.set_filter(
            self.filter.text(), self.carrying.isChecked(), structures
        )

        self.shown.setText(f"{shown} of {len(structures)}")

    def _chosen(self, index):
        """A row picked in the table, which the panel may refuse to leave."""

        if not self.show_index(index):
            self.table.point_at(self.index)
            return

        self.selected.emit(index)

        if self.frame_wanted.isChecked():
            self.framing_asked.emit(index)

    def _asked_framing(self, index):
        """
        A double-click, which frames whatever the box says.

        Only on the row that is already open, which after a double-click is
        always this one: the first of the two clicks has been through `_chosen`
        already. A refusal there leaves `index` where it was, and this then does
        nothing -- which is right, because the trace worth looking at is still
        the one whose text is in the box.
        """

        if index == self.index:
            self.framing_asked.emit(index)

    def _reach_changed(self):
        """
        The dial: the picture, and the one column that reads from the same number.

        Not the tallies beside it. How far a measurement answers for decides what
        holds and says nothing about what is written, and a dial that rewrote the
        counts would be claiming otherwise.
        """

        self._redraw_view()
        self.table.refresh_holds(self.document.dataset.structures, self.max_gap)

    # -- one structure -----------------------------------------------------

    def show_index(self, index):
        """
        Puts a structure on screen, whoever asked. False if the old one was kept.

        Selecting is the one gesture that would throw away work without being
        asked to: a block typed and not applied lives only in the box, and a
        click on another trace would have redrawn over it. So leaving is a
        question, and the caller has to be able to hear no -- the map holds its
        highlight where it was, and the table goes back to the row it was on.
        """

        if index == self.index:
            return True

        if not self._may_leave():
            return False

        self.index = index

        self.table.point_at(index)
        self._redraw()

        return True

    def _may_leave(self):
        """Asks before a block that was typed and never applied is redrawn over."""

        if self.index is None or self.text.toPlainText() == self.document.text_of(
            self.index
        ):
            return True

        answer = QtWidgets.QMessageBox.question(
            self,
            "Not applied",
            f"{self.document.dataset.structures[self.index].ident} has been "
            f"edited in the box and not applied.\n\nApply reads it into the "
            f"document; moving on without it leaves it behind.",
            QtWidgets.QMessageBox.StandardButton.Apply
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel,
        )

        if answer == QtWidgets.QMessageBox.StandardButton.Cancel:
            return False

        if answer == QtWidgets.QMessageBox.StandardButton.Apply:
            # A block that will not parse is a block that cannot be left on
            # these terms either: the error is now on screen under the text it
            # is about, which is where it can be fixed.
            return self.apply_block()

        return True

    def _redraw(self):
        """The block put back as the document has it, and the picture with it."""

        if self.index is None:
            return

        self.problem.setVisible(False)
        self.text.setPlainText(self.document.text_of(self.index))
        self._park_cursor()
        self._redraw_view()

    def _redraw_view(self):
        """
        The picture alone, for the things that change it and not the file.

        The reach is one of those: how far a measurement answers for decides
        what holds along the trace and says nothing about what is written on it,
        so turning that dial must not put the block back and take a half-written
        line away with it.
        """

        if self.index is None:
            return

        self.view.show_structure(
            self.document.dataset.structures[self.index], self.max_gap
        )

    @property
    def max_gap(self):
        return float(self.gap_spin.value())

    def apply_block(self):
        """Parses the box into the document, or shows why it will not go."""

        if self.index is None:
            return False

        try:
            self.document.replace(self.index, self.text.toPlainText())
        except ValueError as err:
            self.problem.setText(str(err))
            self.problem.setVisible(True)
            return False

        self.problem.setVisible(False)

        # Re-read from the document rather than left as typed: what is on screen
        # is now what the file holds, and the two being the same object is what
        # keeps `Revert` meaning something.
        self.text.setPlainText(self.document.text_of(self.index))
        self._park_cursor()
        self._redraw_view()

        # The row as well: an applied block can have added the first attitude to
        # a trace, which changes its tally, what holds along it, and whether the
        # `carrying` filter keeps it -- and the filter is re-run for that last
        # one, or a trace would stay hidden from a list it now belongs in.
        self.table.update_row(
            self.index, self.document.dataset.structures[self.index], self.max_gap
        )
        self._apply_filter()

        self.applied.emit(self.index)

        return True

    # -- writing a line ----------------------------------------------------

    def _path_line(self, lines):
        """Which line the geometry starts on, or the end of the block."""

        return next(
            (n for n, line in enumerate(lines) if line.strip().partition(" ")[0] == "path"),
            len(lines),
        )

    def add_line(self, template):
        """
        Writes a template line above the path, and selects its first anchor.

        Above the path because that is where an assertion goes: `dumps` puts the
        geometry last, so a line written after it would sit among the vertices
        and read as one of them.

        And it says what it just claimed, which is the other half of the band
        turning on. A template arrives with both ends `*`, so pressing this
        lights the whole trace -- correctly, since applied unchanged that is
        exactly what the line would claim, and unexplained it reads as the map
        having been coloured in. The words go here and not in
        `_covering_changed`, for the reason written there: a button press is a
        gesture asking for an answer, and a keystroke is not.
        """

        if self.index is None:
            return

        lines = self.text.toPlainText().splitlines()
        at = self._path_line(lines)

        lines.insert(at, template)

        self.text.setPlainText("\n".join(lines))
        self._aim_at_anchor(sum(len(line) + 1 for line in lines[:at]))

        claim = self.claim_said()

        if claim is not None:
            self.said.emit(f"{claim} -- shift-click the map to pick an end")

    def add_written(self, written):
        """
        Lines that are already finished, above the path, with nothing aimed at.

        `add_line`'s sibling rather than `add_line` itself, and the difference is
        the `*`. That one selects the next one so a shift-click on the map fills
        it in; a computed fit can carry a `*` of its own, which is the format's
        word for an end of the path and the one token on the line that is not a
        coordinate. Aiming at it would offer to overwrite it, and the next pick
        anywhere on the map would land there.
        """

        if self.index is None or not written:
            return

        lines = self.text.toPlainText().splitlines()
        at = self._path_line(lines)

        self.text.setPlainText(
            "\n".join(lines[:at] + list(written) + lines[at:])
        )
        self._park_cursor()

    # -- the topography, read along this trace ------------------------------

    def _gate(self):
        """
        The gate these traces are read through, as `(gate, the sigma it measured)`.

        Off every path in the file and never the one being fitted, which is
        `fits.gate_for`'s own rule: a floor measured on one trace would move from
        trace to trace, and then two verdicts in this file would not be answers to
        the same question.

        Measured when the first fit is asked for rather than at opening, because a
        session that never fits should not pay for it -- and kept, because it does
        not depend on which trace is selected.
        """

        if self._gate_measured is None:
            self._gate_measured = gate_for(
                [structure.path for structure in self.document.dataset.structures]
            )

        return self._gate_measured

    def fit_off_dem(self):
        """
        The topography read along the selected trace, as `fit` lines in the box.

        Returns the `fits.Reading`, so that what happened can be asked about
        rather than read out of a label.

        **Into the box and not into the document**, which is the rule the three
        template buttons already follow: nothing reaches the model until it has
        been through the parser, and Apply is where that happens. Here it buys
        something more than consistency -- a fit is an assertion about a plane,
        these arrived without anybody looking at them, and the block can be read
        and reverted before it is kept.

        **And no dialog**, which is the other half of the same choice. One trace
        is about ten milliseconds, so there is nothing to put a progress bar in
        front of; and a modal box is a thing the headless checks would sit down
        in front of forever, which is why the section panel keeps its own fitting
        and its own message box in two methods.
        """

        if self.index is None or self.dem is None or self.dem_said:
            return None

        structure = self.document.dataset.structures[self.index]

        gate, sigma = self._gate()
        reading = fits_along(
            structure, self.dem, gate, convergence=self.convergence
        )

        self.add_written([as_line(fit) for fit in reading.fits])

        told = [f"{structure.ident}: {reading.describe()}"]

        if reading.fits:
            told.append("Apply to keep")

        # The precedence, said rather than settled. `attitude_at` takes the
        # *first* fit that covers a progressive, and these lines go where
        # `add_written` puts them, which is after the fits the block already had.
        # So on a file that came out of the import -- where every trace that could
        # be fitted carries one already -- a fit computed here answers nowhere
        # until somebody moves it up. Which line is winning at which metre is
        # exactly what the band above draws, so this is visible; leaving it
        # unsaid as well is what would make it a trap.
        carried = len(structure.fits)

        if reading.fits and carried:
            told.append(
                f"it already carries {carried} fit(s), and the first one "
                f"covering a metre is what answers there"
            )

        if not self._gate_said:
            self._gate_said = True
            told.append(
                f"gate: lever {gate.min_lever:.1f} m "
                + (
                    f"(3x the {sigma:.1f} m measured off these traces)"
                    if sigma is not None
                    else "(the default: these traces do not measure a pen)"
                )
            )

        self.said.emit("; ".join(told))

        return reading

    def _park_cursor(self):
        """Puts the cursor where a new line goes, rather than at the top."""

        lines = self.text.toPlainText().splitlines()
        at = self._path_line(lines)

        cursor = self.text.textCursor()
        cursor.setPosition(max(sum(len(line) + 1 for line in lines[:at]) - 1, 0))

        self.text.setTextCursor(cursor)

    # -- the stretch the caret's line claims ---------------------------------

    def _covering_now(self):
        """The interval the line under the caret says it covers, or None."""

        if self.index is None:
            return None

        return interval_of(
            self.text.textCursor().block().text(),
            self.document.dataset.structures[self.index].path,
        )

    def claim_said(self):
        """
        What the caret's line claims, in words, or None if it claims nothing.

        The band shows where and the sentence shows how much, and the second is
        not a caption for the first: on an AOI-wide framing the whole of a
        kilometre of fault is a few pixels, so *how much of it* is a question
        the picture cannot answer at the scale the work is done at.

        Read off `self._covering` and not off the box, so that the sentence is
        always about the stretch that was last reported -- `pick` asks for this
        after `insert_anchor` has run, and the two have to be the same claim.
        """

        index, interval = self._covering

        if interval is None or index is None:
            return None

        s0, s1 = interval

        # There is no drawing for this one, and that is why it is here rather
        # than left to the picture: reversed, `Span.covers` holds the line over
        # no part of the trace, so it parses, applies, and sits in the file
        # looking like a decision while doing nothing.
        if s0 > s1:
            return (
                f"this line runs from {s0:.0f} m back to {s1:.0f} m: written "
                f"this way round it covers no part of the trace, because a "
                f"stretch is `s0 <= s <= s1`"
            )

        whole = self.document.dataset.structures[index].length

        if s1 - s0 >= whole - AT_THE_END:
            return f"this line claims the whole trace, {whole:.0f} m"

        return (
            f"this line claims {s0:.0f} to {s1:.0f} m -- "
            f"{s1 - s0:.0f} m of {whole:.0f}"
        )

    def _covering_changed(self):
        """
        Reports the caret's stretch when it becomes a different one.

        **On change and not on every gesture**, which is what `self._covering`
        is for: the caret moves character by character and the stretch does not
        move with it.

        **The band always, the words almost never.** Saying the claim on every
        change was tried and is wrong: the status bar is one line, this fires on
        every keystroke, and `fit off the DEM` writes its report into that same
        bar by typing into this same box -- so the sentence that says a trace
        already carries a fit and which of the two will answer was overwritten
        by a generic extent, immediately and every time. A picture can sit beside
        other things on the map; a sentence cannot sit beside another sentence.

        So the words are left to the gesture that asks for them -- `pick` puts
        the claim in its own report, where it is about the click somebody just
        made -- and the one case kept here is the pair written the wrong way
        round, which has no band, because `covers` holds it over no part of the
        trace. That one has to speak: there is nothing to look at.
        """

        reported = (self.index, self._covering_now())

        if reported == self._covering:
            return

        self._covering = reported
        interval = reported[1]

        self.covering.emit(interval)

        if interval is not None and interval[0] > interval[1]:
            self.said.emit(self.claim_said())

    # -- and the plane that line carries ------------------------------------

    def _line_now(self):
        """The text of the line the caret is on."""

        return self.text.textCursor().block().text()

    def _holding_changed(self):
        """Reports the caret's plane when it becomes a different one."""

        reported = (self.index, plane_of(self._line_now()))

        if reported == self._holding:
            return

        self._holding = reported
        self.holding.emit(reported[1])

    def has_plane_slot(self):
        """Whether the caret's line has somewhere to write a plane."""

        return with_plane(self._line_now(), 0.0, 0.0) is not None

    def pinned_at(self):
        """
        Where a plane steered against the caret's line hangs, and over how much
        ground -- `((x, y), metres)` in the file's own projection, or None.

        **The middle of the stretch, for a line that claims one.** Not an end:
        the judgement is whether the cut runs *along* the claim, and a plane
        pinned at one end of it is exactly right there and free to swing away
        over the rest, which is the error this is meant to show rather than
        hide.

        For an `attitude` it is the anchor, which is the same rule read at a
        place instead of over a stretch, and it is the more useful half on a
        file that already has readings in it: it puts a compass measurement on
        the topography and asks whether the ground agrees. `conflicts.py` asks
        that arithmetically. This asks it by looking.

        A pair written the wrong way round pins nothing, for the reason it draws
        nothing: `covers` holds it over no ground, so there is no middle of it.
        """

        if self.index is None:
            return None

        line = self._line_now()
        path = self.document.dataset.structures[self.index].path
        interval = interval_of(line, path)

        if interval is None:
            anchor = anchor_of(line)

            return None if anchor is None else (anchor, 0.0)

        s0, s1 = interval

        if s0 > s1:
            return None

        return point_on(path, (s0 + s1) / 2.0), s1 - s0

    def take_plane(self, dip_dir, dip, attrs=None):
        """
        Writes a steered attitude into the caret's line. Returns it, or None.

        One replacement of one line, so it is one step of the undo stack: a
        dial that wrote as it turned would have put ninety of them there, which
        is the reason nothing here is written until this is pressed.

        The attributes go on with it and are not decoration. FORMAT.md's rule
        for a derived plane is that it says which producer made it and does not
        borrow another's diagnostics -- so this writes `from=` and what the
        number is referenced to, and writes none of `snr`, `flat` or `jack`,
        having no gate and no residual to put in them. What it has instead is a
        person who looked, and `from=plane-dem` is the honest name for that.
        """

        written = with_plane(self._line_now(), dip_dir, dip)

        if written is None:
            return None

        if attrs:
            written = with_attrs(written, attrs)

        edit = self.text.textCursor()
        edit.movePosition(QtGui.QTextCursor.MoveOperation.StartOfBlock)
        edit.movePosition(
            QtGui.QTextCursor.MoveOperation.EndOfBlock,
            QtGui.QTextCursor.MoveMode.KeepAnchor,
        )
        edit.insertText(written)

        return written

    def _aim_at_anchor(self, start, same_line=False):
        """Selects the next `*`, so that a picked anchor replaces it."""

        text = self.text.toPlainText()
        stop = len(text)

        if same_line:
            ends = text.find("\n", start)
            stop = len(text) if ends < 0 else ends

        at = text.find("*", start)

        if at < 0 or at >= stop:
            return False

        cursor = self.text.textCursor()
        cursor.setPosition(at)
        cursor.setPosition(at + 1, QtGui.QTextCursor.MoveMode.KeepAnchor)

        self.text.setTextCursor(cursor)

        return True

    def insert_anchor(self, x, y):
        """
        Writes a picked anchor over the selected `*`, or at the cursor.

        A space goes in front of it where one is needed, an anchor being a token
        and not a suffix: without it a shift-click at the end of a line glues
        `@x,y` onto whatever that line ended with, and the parser reads the pair
        as one word it cannot make sense of.
        """

        cursor = self.text.textCursor()
        written = f"@{x:.2f},{y:.2f}"
        at = cursor.selectionStart() if cursor.hasSelection() else cursor.position()

        if not cursor.hasSelection():
            before = self.text.toPlainText()[:at]

            if before and not before[-1].isspace():
                written = " " + written

        cursor.insertText(written)

        # The next one on the same line, so that `span use * *` takes two clicks
        # and stops there rather than running on into the line below.
        self._aim_at_anchor(at + len(written), same_line=True)

        self.text.setFocus()


class PlaneSteering(QtWidgets.QWidget):
    """
    A dial, a slider, and the number they are both saying: the plane on the DEM.

    `tools/intersection.py`'s panel, cut down to what is left once the tool has
    something to aim at. Gone are the source point's three boxes and the
    compute-window spinner, and neither is a simplification: there the point is
    typed in or clicked anywhere, because the tool has no idea what you are
    looking at, and the window is a cost dial because the plane is unbounded.
    Here the point is on a trace -- the middle of the stretch the caret's line
    claims, or wherever ctrl-click puts it -- and the window is sized from the
    stretch or from what is on screen, so both were answers to questions this
    tool can work out for itself.

    What is left is the pair of controls that *are* the tool -- turn it, watch
    the curves move, stop when they run along the fault -- plus the one button
    that was missing from the other tool entirely: the number going into the
    file.

    **It reports into its own label and not into the status bar.** The bar is
    one line and shared with everything the panel says, and this fires on every
    step of a dial: a per-frame report there would be the mistake the claimed
    stretch already made once, scaled up by a factor of ninety.
    """

    steered = QtCore.pyqtSignal(float, float)
    take_asked = QtCore.pyqtSignal()
    armed_changed = QtCore.pyqtSignal(bool)
    unpin_asked = QtCore.pyqtSignal()

    # QDial puts its minimum at six o'clock, not at twelve, and it runs
    # clockwise like an azimuth -- so between its scale and dip direction there
    # is exactly half a turn. Measured in the other tool by grabbing the widget
    # and hunting for the needle, and repeated here rather than imported,
    # because importing it would make this window depend on that one.
    DIAL_NORTH_OFFSET = 180

    def __init__(self, refusal=None, parent=None):
        super().__init__(parent)

        self.on = QtWidgets.QCheckBox("plane on the DEM")
        self.on.setToolTip(
            "Lay a plane on the trace and draw where it cuts the topography, "
            "with a band along the trace saying how near the cut runs at each "
            "metre. The plane hangs at the middle of the stretch the line in "
            "the box claims, or wherever you ctrl-click the trace -- which is "
            "how to turn the dial first and decide which stretch afterwards."
        )

        self.dial = QtWidgets.QDial()
        self.dial.setRange(0, 359)
        self.dial.setWrapping(True)
        self.dial.setNotchesVisible(True)
        self.dial.setMinimumSize(132, 132)

        # The dial alone steps by a whole degree and the convergence here is
        # 0.8, so without the tenth the correction would be finer than the
        # control meant to apply it. The box is what commands; the dial follows.
        self.dip_dir = QtWidgets.QDoubleSpinBox()
        self.dip_dir.setRange(0.0, 359.9)
        self.dip_dir.setDecimals(1)
        self.dip_dir.setSingleStep(0.1)
        self.dip_dir.setWrapping(True)
        self.dip_dir.setSuffix("°  dip dir")

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 90)
        self.slider.setTickInterval(15)
        self.slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)

        self.dip = QtWidgets.QDoubleSpinBox()
        self.dip.setRange(0.0, 90.0)
        self.dip.setDecimals(1)
        self.dip.setSingleStep(0.1)
        self.dip.setSuffix("°  dip")

        self.dip_dir.setValue(90.0)
        self.dip.setValue(30.0)
        self._sync_dial()
        self._sync_slider()

        self.label = QtWidgets.QLabel()
        self.label.setWordWrap(True)
        self.label.setStyleSheet("color: #6a6a6a; font-size: 10px;")
        self.label.setMinimumHeight(48)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        self.take = QtWidgets.QPushButton("write it in the line")
        self.take.clicked.connect(lambda: self.take_asked.emit())

        # Disabled rather than hidden, and that is the whole design of it. A pin
        # put by hand is a mode -- the plane stops following the caret until it
        # is given back -- and a mode with no visible way out is a trap. Shown
        # greyed when there is no pin to release, so the way out is legible
        # before it is needed and the panel does not change height when one is
        # put down.
        self.release = QtWidgets.QPushButton("release the pin")
        self.release.setEnabled(False)
        self.release.setToolTip(
            "Give the plane back to the line the caret is on. Ctrl-click the "
            "trace to pin it anywhere instead, which is how to steer a plane "
            "before deciding which stretch it is about."
        )
        self.release.clicked.connect(lambda: self.unpin_asked.emit())

        self.dial.valueChanged.connect(self._dial_moved)
        self.dip_dir.valueChanged.connect(self._dip_dir_typed)
        self.slider.valueChanged.connect(self._slider_moved)
        self.dip.valueChanged.connect(self._dip_typed)
        self.on.toggled.connect(self._toggled)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.on)
        layout.addWidget(self.dial)
        layout.addWidget(self.dip_dir)
        layout.addWidget(self.slider)
        layout.addWidget(self.dip)
        layout.addWidget(self.take)
        layout.addWidget(self.release)
        layout.addWidget(self.label)

        # The refusal is a fact about the session and not about the gesture --
        # the same one the `fit off the DEM` button is disabled by -- so it is
        # settled once, at the door, and shown as the reason the controls are
        # grey. A control that looks available and answers with a message box
        # would be offering something this session cannot do.
        self.refusal = refusal

        if refusal:
            self.setEnabled(False)
            self.setToolTip(refusal)
        else:
            self._steering(False)

    # -- what it is saying -------------------------------------------------

    def armed(self):
        return self.isEnabled() and self.on.isChecked()

    def plane(self):
        return float(self.dip_dir.value()), float(self.dip.value())

    def show_plane(self, dip_dir, dip):
        """
        Puts the controls on a plane without answering.

        Silent because this is the caret's doing and not the hand's: clicking
        into a line that already carries `140.5/31` should show that plane, and
        a control that emitted on being set would then redraw once per keystroke
        typed into the box. The caller recomputes if it wants to.
        """

        for box, value in ((self.dip_dir, float(dip_dir) % 360.0), (self.dip, float(dip))):
            with QtCore.QSignalBlocker(box):
                box.setValue(value)

        self._sync_dial()
        self._sync_slider()

    def note(self, said):
        self.label.setText(said or "")

    def set_pinned(self, pinned):
        """
        Whether a pin has been put by hand, which is the only way back from one.

        Not routed through `_steering`, which follows the checkbox: a pin can be
        put down with the steering off -- ctrl-click says so and offers to
        switch it on -- and a release button greyed out at exactly that moment
        would be the one state in which the mode is real and looks like it is
        not.
        """

        self.release.setEnabled(bool(pinned))

    def set_writable(self, may, why=None):
        """Whether there is a plane slot under the caret for the button to fill."""

        self.take.setEnabled(bool(may) and self.armed())
        self.take.setToolTip(
            why or "Write this attitude into the plane slot of the line the "
                   "caret is on, and say where it came from. Nothing reaches "
                   "the file until Apply, and then Save."
        )

    # -- the two controls, each following the other ------------------------

    def _sync_dial(self):
        with QtCore.QSignalBlocker(self.dial):
            self.dial.setValue(
                int(round(self.dip_dir.value() - self.DIAL_NORTH_OFFSET)) % 360
            )

    def _sync_slider(self):
        with QtCore.QSignalBlocker(self.slider):
            self.slider.setValue(int(round(self.dip.value())))

    def _dial_moved(self, value):
        with QtCore.QSignalBlocker(self.dip_dir):
            self.dip_dir.setValue(float((value + self.DIAL_NORTH_OFFSET) % 360))

        self._answer()

    def _dip_dir_typed(self, value):
        self._sync_dial()
        self._answer()

    def _slider_moved(self, value):
        with QtCore.QSignalBlocker(self.dip):
            self.dip.setValue(float(value))

        self._answer()

    def _dip_typed(self, value):
        self._sync_slider()
        self._answer()

    def _steering(self, on):
        for widget in (self.dial, self.dip_dir, self.slider, self.dip, self.take):
            widget.setEnabled(on)

    def _toggled(self, on):
        self._steering(on)

        if not on:
            self.note("")

        self.armed_changed.emit(bool(on))
        self._answer()

    def _answer(self):
        if self.armed():
            self.steered.emit(*self.plane())


class EditorWindow(QtWidgets.QMainWindow):
    """The map with the file's traces on it, and the file beside them."""

    def __init__(self, session, document, legend="beside"):
        super().__init__()

        self.session = session
        self.document = document
        self.index = None

        self._forward, self._back = self._transformers()

        # The paths in map coordinates, worked out once. The model is in the
        # file's projection and stays there; this is the other side of that.
        self._drawn = [self.on_map(st.path) for st in document.dataset.structures]

        # Whether map and panel were laid side by side for want of anything
        # remembered about them, which is what `build` reads to know that the
        # map has already been given a size.
        self.tiled = False

        # Where the steered plane is laid, as `(x, y, z)` in the file's own
        # projection, and the crop of DEM it is being cut against. The window is
        # kept rather than re-read per frame: reading one costs 4.9 ms on
        # 1000x1000 and the pin only moves when the claim does, which is a
        # gesture and not a dial step. `_cut_for` is what it was read for, so
        # that "has the pin moved" is a comparison and not a raster read.
        self._pin = None
        self._cut_window = None
        self._cut_for = None

        # A pin put by hand, as `(x, y, the ground it was put while looking at)`,
        # which overrides the one the caret's line implies for as long as it is
        # set. The point of it is the order of work: the line's pin is the middle
        # of a stretch, so it can only exist once somebody has declared which
        # stretch -- and the stretch is the conclusion, not the premise. This one
        # lets the plane be steered first and attributed afterwards.
        #
        # The extent is stored with it rather than read per frame so the window
        # does not resize under a zoom: the ground being cut against is the
        # ground that was on screen when the pin went down, and moving it is how
        # you ask for different ground.
        self._free_pin = None

        # The topography read along the selected trace, for the band. Kept
        # against what it was read for, which is the trace and the window: the
        # DEM half of the band costs a gather and the plane half costs six
        # multiplications, so a dial turned ninety times does the second ninety
        # times and the first not at all.
        self._ground = None
        self._ground_for = None

        # And what the band's segments were last built for, kept apart from
        # `_ground_for` because they are two different costs: the ground is a
        # raster gather and the segments are a reprojection and a path per
        # segment. They happen to move together today; a band drawn over a
        # reframed map would move the second without the first.
        self._agreeing_for = None
        self._joined = np.zeros(0, dtype=bool)
        self._band_xy = np.zeros((0, 2))

        # Which trace the framing is on its way to, and the timer it waits on.
        self._framing = None
        self._frame_timer = QtCore.QTimer(self)
        self._frame_timer.setSingleShot(True)
        self._frame_timer.timeout.connect(lambda: self.frame_now())

        # The station dots that are drawn, as `(where on the map, the record, which
        # circle on the net)`, and which of them the tooltip is currently about.
        # The artist holds coordinates and nothing else, so the records they were
        # made from have to be kept alongside or there is no way back from a dot to
        # a station.
        #
        # The third element is what ties the two windows together, and it is stored
        # rather than worked out because the two lists are filtered differently: a
        # dot is drawn for a station that has an anchor and a circle for a station
        # that has a plane, and in `merid_faults` all 23 have both. All 23 today --
        # an attitude carrying `at=` and no plane is a legal record, it would be a
        # dot with nothing to point at, and a net indexed by counting dots would
        # then answer with its neighbour's plane. `None` there is that station.
        self._marked = []
        self._tipped = None

        # And which circle the cursor is pointing at, which is not the same
        # question: the dot is on the map and the circle is on the net, and between
        # the two sits a station that may have no plane.
        self._netted = None

        # Before the map, which draws them, and before the first selection, which
        # picks its own out of them rather than working them out again.
        self._marked = self._all_stations()

        self._build_ui(legend)
        self._draw_base_map()

        self.select(0)

        self._retitle()
        self.say(self._opening())

    # -- the two projections ----------------------------------------------

    def _transformers(self):
        """
        File to map and back, or `(None, None)` where they are the same.

        A file with no CRS line is read as the session's rather than refused:
        the coordinates are then taken at face value, which is the only thing
        left to do with them, and it is a thing worth being able to fix from
        inside the editor.
        """

        from pyproj import CRS, Transformer

        declared = self.document.dataset.crs
        theirs = CRS.from_user_input(declared) if declared else None
        ours = self.session.crs

        if theirs is None or ours is None or theirs.equals(ours):
            return None, None

        return (
            Transformer.from_crs(theirs, ours, always_xy=True),
            Transformer.from_crs(ours, theirs, always_xy=True),
        )

    def _path_crs(self):
        """
        What the paths are actually in: the file's projection, or the session's.

        The same fallback `_transformers` makes and `build` checked for degrees
        under -- a file with no `crs` line is read as the session's, that being
        the only thing left to do with the coordinates. Named here because the
        DEM is refused against this and not against the declared line: a file
        that declares nothing would otherwise refuse every DEM there is.
        """

        from pyproj import CRS

        declared = self.document.dataset.crs

        return CRS.from_user_input(declared) if declared else self.session.crs

    def on_map(self, path):
        """A path in the file's projection, as the map's coordinates."""

        # `len(...) == 0` and not `not path`, because one of the callers now
        # hands over an `(n, 2)` array: numpy raises on the truth value of one
        # with more than one element, so the emptiness test refused every path
        # that had anything in it.
        if len(path) == 0:
            return []

        if self._forward is None:
            return [(float(x), float(y)) for x, y in path]

        xs, ys = self._forward.transform(*zip(*path))

        return [(float(x), float(y)) for x, y in zip(xs, ys)]

    def in_file(self, x, y):
        """A point off the map, in the projection the file is written in."""

        if self._back is None:
            return float(x), float(y)

        moved = self._back.transform(float(x), float(y))

        return float(moved[0]), float(moved[1])

    # -- construction ------------------------------------------------------

    def _build_ui(self, legend):
        self.map_view = MapView(self.session, legend=legend)
        self.map_view.legend_handles_provider = self._legend_handles
        self.map_view.pressed.connect(self._on_map_pressed)
        self.map_view.hovered.connect(self._on_map_hover)
        self.map_view.hover_off.connect(lambda: self._on_map_hover(None, None))
        self.map_view.status.connect(self.say)

        # A third window, and not the dock it was first built as. The fold tool's
        # argument for a dock -- small, looked at beside the map, draggable back
        # into the side -- holds there and came apart here on two counts, and
        # both of them are about how big the picture is allowed to be. A dock is
        # as wide as the map can spare, which was 276 px and a circle 268 px
        # across; and the one way out of that, floating it, is a Qt::Tool window
        # whose geometry this tool does not save, so a net dragged out and made
        # readable would have to be made readable again every run. In the group
        # it is saved with the map and the panel, it goes on the second monitor
        # the rest of this tool is built around, and the circle is as big as the
        # window -- 412 px at the size it opens.
        self.net = StereonetView(figsize=NET_FIGSIZE_IN)
        self.net_window = SatelliteWindow(
            NET_EMPTY_TITLE, self.net, NET_WINDOW_PX, parent=self
        )

        self.panel = EditorPanel(
            self.document, dem=self.session.dem, crs=self._path_crs()
        )
        self.panel.selected.connect(self.select)
        self.panel.applied.connect(self._on_applied)
        self.panel.framing_asked.connect(self.frame_on)
        self.panel.said.connect(self.say)
        self.panel.covering.connect(self._show_claimed)
        self.panel.holding.connect(self._show_held)

        self.save_button = QtWidgets.QPushButton("Save")
        self.save_button.setToolTip(
            "Write the file, replacing the lines of the structures that were "
            "edited and leaving every other byte of it alone (Ctrl+S)."
        )
        self.save_button.clicked.connect(self.save)

        self.save_as_button = QtWidgets.QPushButton("Save as...")
        self.save_as_button.clicked.connect(self.save_as)

        writing = QtWidgets.QHBoxLayout()
        writing.addWidget(self.save_button)
        writing.addWidget(self.save_as_button)
        writing.addStretch(1)

        # The status bar's words again, beside the panel. The windows can be on
        # two screens, and which of them the news belongs on depends on the
        # news: the map reports what a click found, the panel what Apply and Save
        # did, and each is read from the other window often enough to matter.
        # `say` writes both -- one call, no state between them, so there is
        # nothing in the echo that can drift out of step with the bar.
        self.echo = QtWidgets.QLabel()
        self.echo.setStyleSheet("color: #555555; font-size: 11px;")
        self.echo.setMinimumHeight(16)

        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(self.panel, stretch=1)
        side_layout.addLayout(writing)
        side_layout.addWidget(self.echo)
        side.setMinimumWidth(PANEL_WIDTH_PX)

        # The map is the window and the panel is beside it, which is not the same
        # arrangement as before: the two were one frame split down the middle, so
        # the map could not be made bigger without making the table smaller, and
        # neither could go on the other screen. What the split cost is not room
        # but reach -- a second monitor is the map at the size the DEM deserves
        # and the file open next to it, and a splitter cannot be dragged across a
        # screen boundary.
        self.panel_window = SatelliteWindow(
            self._panel_title(), side, PANEL_WINDOW_PX, parent=self
        )

        self.group = WindowGroup(
            self,
            SETTINGS_NAME,
            {"panel": self.panel_window, "net": self.net_window},
            placer=self._place_unremembered,
        )

        central = QtWidgets.QWidget()
        map_layout = QtWidgets.QHBoxLayout(central)
        map_layout.setContentsMargins(0, 0, 0, 0)
        map_layout.addWidget(self.map_view, stretch=1)
        map_layout.addWidget(self._beside_the_map(legend))

        self.setCentralWidget(central)

        self._build_menu()
        self._build_shortcuts()

    def _beside_the_map(self, legend):
        """
        The narrow column on the map's own frame: the steering, and the legend.

        The steering goes here and not in the panel, and the two windows can be
        on different screens, so this is a choice about where the hand has to
        be. What it steers is a picture -- curves swinging about a pin -- and
        the judgement it serves is made by looking at them against the trace.
        The number it produces lands in the box, which is a thing read
        afterwards; the curves are the thing watched while the hand moves. A
        dial on the other monitor would be steering by feel.
        """

        self.steering = PlaneSteering(refusal=self._steering_refusal())
        self.steering.steered.connect(self._steer)
        self.steering.take_asked.connect(self._take_plane)
        self.steering.armed_changed.connect(self._steering_armed)
        self.steering.unpin_asked.connect(self.unpin)

        beside = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(beside)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.steering)
        layout.addStretch(1)
        layout.addWidget(LegendControls(self.map_view, placement=legend))

        beside.setMaximumWidth(MAP_COLUMN_PX)

        return beside

    def _steering_refusal(self):
        """Why no plane can be laid on this session's topography, or None.

        The same refusal the `fit off the DEM` button is disabled by, and it is
        settled once at the door rather than per gesture, because it is a fact
        about the pair -- this DEM, these traces -- and not about the click.
        """

        if self.session.dem is None:
            return (
                "No DEM in this session, and a plane laid on the topography "
                "needs one. The traces draw without it; this is the one thing "
                "here that does not."
            )

        return self.panel.dem_said

    def _build_menu(self):
        """The way back to the panel and the net, once they have been closed."""

        menu = self.menuBar().addMenu("&Windows")

        self.window_actions = self.group.actions_into(
            menu, {"panel": "&Structures", "net": "Stereo&net"}
        )

        menu.addSeparator()

        front = QtGui.QAction("Bring all to &front", self)
        front.triggered.connect(self.group.raise_all)
        menu.addAction(front)

    def _build_shortcuts(self):
        """
        Save and Apply from either window, which takes saying so.

        Both were shortcuts on their buttons, and a button's shortcut reaches
        only the window the button is in -- which used to be the only window
        there was. Left alone, Ctrl+S would have worked over the panel and done
        nothing over the map, where half the work is: clicking anchors along a
        trace and then writing the file is one motion, and it goes through the
        map.

        Given to every window of the group, rather than made
        `ApplicationShortcut` -- which reaches every window of the application,
        including another tool's. The section tool binds Ctrl+S to saving a
        section, and two tools open at once would have had one ambiguous
        shortcut between them and Qt firing neither.

        Every window, including the one nothing is typed into: the net is a
        canvas and clicking it gives it the keyboard, so leaving it out would
        make Ctrl+S depend on which window was last clicked -- which is worse
        than no shortcut, because it fails silently and only sometimes.
        """

        for label, shortcut, slot in (
            ("Save", "Ctrl+S", self.save),
            ("Apply", "Ctrl+Return", self.panel.apply_block),
        ):
            action = QtGui.QAction(label, self)
            action.setShortcut(shortcut)
            action.triggered.connect(slot)

            self.addAction(action)

            for satellite in self.group.satellites.values():
                satellite.addAction(action)

    def _panel_title(self):
        mark = "*" if self.document.dirty else ""

        return f"gSurf - structures - {self.document.path.name}{mark}"

    def say(self, message):
        """One line of news, on the map's bar and in the panel's echo."""

        self.statusBar().showMessage(message)

        # Set as the tooltip too, because the label is one line in a 520-pixel
        # panel and the longer messages here run past that. Clipped text with
        # nothing behind it would be a sentence the panel silently shortens.
        self.echo.setText(message)
        self.echo.setToolTip(message)

    def showEvent(self, event):
        """The panel comes up with the map, the first time and only then."""

        super().showEvent(event)

        self.group.show_satellites()

    def _place_unremembered(self, skip=()):
        """
        Map and panel side by side, filling the screen, the first time ever.

        The group's own answer -- satellites down the right edge, and only on a
        desktop 1600 wide -- is for a tool whose windows do not fit, where the
        window manager's guess is as good as any. Two of these three do fit
        where that rule says they do not: on the 1366x741 of usable area this is
        written on, 840 of map beside 520 of panel is the splitter's own
        arrangement made out of two windows. Which is the one thing this change
        must not be worse than -- taking a pane out of a frame should buy a
        second monitor, not cost a first one.

        It places the map as well, which is why `build` asks whether this ran
        before falling back to `fit_to_screen`: a map maximised over a panel
        placed beside it would be the arrangement this exists to avoid.

        Once. The first drag onto another screen is remembered, and after that
        there is something remembered and this never runs again.
        """

        available = self.main_screen_area()

        # The net goes first and goes over the map, there being no third
        # rectangle once the other two have filled the screen. The corner is a
        # choice about what it covers: the map is re-framed on the trace being
        # edited every time a row is picked, so what the cursor is hunting for
        # is in the middle of it. Lifted clear of the status bar, which runs the
        # width of the map window and is where `say` writes -- a net resting on
        # the bottom edge would cover the left end of every message, which is
        # where they start.
        #
        # Placed on any screen, narrow or not, because it is small enough to
        # land somewhere on all of them, and because a window covering a corner
        # of the map is undone by one drag -- which is then remembered.
        if "net" not in skip:
            net_width, net_height = NET_WINDOW_PX

            self.net_window.setGeometry(
                available.left() + TILE_GAP_PX,
                available.bottom()
                - net_height
                - FRAME_ALLOWANCE_PX
                - self.statusBar().sizeHint().height(),
                net_width,
                net_height,
            )

        width = available.width() - PANEL_WIDTH_PX - TILE_GAP_PX

        if width < MAP_FLOOR_PX:
            self.tiled = False
            return

        height = available.height() - FRAME_ALLOWANCE_PX

        if "panel" not in skip:
            self.panel_window.setGeometry(
                available.right() - PANEL_WIDTH_PX + 1,
                available.top(),
                PANEL_WIDTH_PX,
                height,
            )

        if "map" not in skip:
            self.setGeometry(available.left(), available.top(), width, height)

        self.tiled = True

    def main_screen_area(self):
        """
        The desktop the windows are placed against.

        A method rather than the one call it is, so that a check can stand a
        narrow screen in front of it: the placement has two branches and a run
        can only ever be on one screen, which off-screen is 800x800 -- narrow
        enough that the branch a real laptop takes would never be exercised.
        """

        return self.screen().availableGeometry()

    def _draw_base_map(self):
        self.map_view.draw_base_map()

        axes = self.map_view.axes

        # Two collections and not 393 lines: they change only when a block is
        # applied, so they belong in the background the blitting is cut from, and
        # one artist is one draw.
        #
        # Two rather than one because which forty-five carry a plane is the
        # question the window opens on, and a map that cannot answer it leaves the
        # table to answer it alone -- you would be able to list them and not to
        # aim at one. The carrying ones go on top, so that where the two cross it
        # is the one with something written on it that stays whole.
        self.traces = {
            carrying: axes.add_collection(
                LineCollection(
                    self._segments(carrying),
                    zorder=3.5 if carrying else 3.0,
                    **(CARRYING_STYLE if carrying else BARE_STYLE),
                )
            )
            for carrying in (False, True)
        }

        # Every station in the file, and in the background with the traces for the
        # same reason they are: they change when a block is applied and at no other
        # time, so redrawing them on a selection would be paying a full draw for a
        # picture that did not move.
        #
        # Under the animated ones and not over: where a dot is both -- the selected
        # fault's own -- what stays whole is the larger one.
        (self.stations,) = axes.plot(
            [], [], color=PROVENANCE_TINT["misurata"], marker="o",
            markersize=OTHER_STATION_SIZE, linestyle="none", zorder=4,
        )
        self._draw_stations()

        self.highlight = self.map_view.add_animated(
            axes.add_line(Line2D([], [], color="#ff7f0e", lw=2.6, zorder=6))
        )
        self.refused = self.map_view.add_animated(
            axes.add_line(
                Line2D([], [], color=PROVENANCE_TINT["rifiutata"], lw=3.4, zorder=7)
            )
        )
        self.marks = self.map_view.add_animated(
            axes.add_line(
                Line2D(
                    [], [], color=PROVENANCE_TINT["misurata"], marker="o",
                    markersize=STATION_SIZE, linestyle="none", zorder=8,
                )
            )
        )
        self.picked = self.map_view.add_animated(
            axes.add_line(
                Line2D(
                    [], [], color="#000000", marker="+", markersize=11,
                    markeredgewidth=1.6, linestyle="none", zorder=9,
                )
            )
        )

        # A band under everything else rather than a line over it, and that is
        # the second try. Dashed and on top it was drawn over the orange
        # highlight, so what showed through the gaps was the selection: the
        # claimed stretch read as a purple-and-orange stripe along the trace --
        # a texture, which is a thing a line can be, and not an extent, which is
        # what this is. Thick, solid, pale and underneath, it is a highlighter
        # stroke and the trace still runs over it in its own colour.
        #
        # Lowest of the four steering artists for the same reason read the other
        # way: a refusal and a selection are about the trace, and this is about a
        # piece of ground the trace happens to cross.
        self.claimed = self.map_view.add_animated(
            axes.add_line(
                Line2D(
                    [], [], color=CLAIMED_TINT, lw=CLAIMED_WIDTH,
                    alpha=CLAIMED_ALPHA, solid_capstyle="butt", zorder=5.5,
                )
            )
        )

        # And under that, how near the cut runs to the trace at each metre of
        # it. A collection and not a line because the quantity varies along the
        # trace and a `Line2D` has one colour: what has to be visible is *where*
        # the agreement stops, which is the number about to be written as the
        # end of a `fit`.
        #
        # Lowest of everything, so that the claim, the trace, the refusals and
        # the cut all lie on top of it in their own colours. It is the only
        # artist here that is about neither the file nor the plane but about how
        # the two are getting on.
        self.agreeing = self.map_view.add_animated(
            axes.add_collection(
                LineCollection(
                    [], linewidths=AGREEING_WIDTH, capstyle="round",
                    joinstyle="round", zorder=5.2,
                )
            )
        )

        # And the plane the hand is steering, where it cuts the ground. Over
        # everything rather than under, which is the opposite of the band above
        # and for a reason the band does not have: a match *is* the cut running
        # along the trace, so at the moment the answer comes right the two lie
        # on top of each other. Underneath, it would disappear exactly then --
        # and "hidden because it agrees" is not something the eye can tell from
        # "not computed". Thin over thick, so a match reads as a purple core
        # down the middle of the orange.
        self.cutting = self.map_view.add_animated(
            axes.add_line(
                Line2D([], [], color=CLAIMED_TINT, lw=CUTTING_WIDTH, zorder=9.5)
            )
        )

        # And where it is pinned, without which the picture means nothing: an
        # intersection is a plane *through a point*, every curve on screen turns
        # about that one, and it is the one thing here nobody chose directly --
        # it is the middle of the claim, which is a consequence and reads as an
        # arbitrary spot until it is drawn.
        self.pin = self.map_view.add_animated(
            axes.add_line(
                Line2D(
                    [], [], color=CLAIMED_TINT, marker="o", markersize=PIN_SIZE,
                    markerfacecolor="none", markeredgewidth=1.6,
                    linestyle="none", zorder=9.6,
                )
            )
        )

        # Built here and not left to the placement combo, which is what used to
        # happen: the other three tools ask for the legend once the map is drawn
        # and this one never did, so its four entries were made on every rebuild
        # and there was no rebuild -- `legend_handles` answered and nothing
        # called it. The window opened with no legend at all until somebody moved
        # the combo, and a colour nobody can look up is a colour that says nothing.
        self.map_view.refresh_legend()
        self.map_view.anchor_home()

    def _all_stations(self):
        """
        Every station dot in the file, as `(on the map, the record, whose trace,
        which of that trace's planes)`.

        Worked out once and kept, rather than per selection, and that is the whole
        of what makes the dots global: 348 of the 393 faults contribute nothing to
        this list and the 45 that do contribute 23 dots between them, so the cost
        of knowing them all is a list of 23 tuples.

        The fourth element counts planes within one structure, so it survives the
        selection moving: a dot points at a circle on the net only while its own
        fault is the one on the net, and that is a comparison made at hover time
        rather than a number that has to be rebuilt.

        A structure with fewer than two points in its path is skipped whatever it
        carries. `point_on` has no line to place a progressive on there, and until
        now such a record could only break the window by being selected -- built
        for the whole file, it would break the window by being in it.
        """

        found = []

        for index, structure in enumerate(self.document.dataset.structures):
            if len(structure.path) < 2:
                continue

            anchored = [a for a in structure.attitudes if a.s is not None]

            if not anchored:
                continue

            drawn = self.on_map([point_on(structure.path, a.s) for a in anchored])
            circles = 0

            for point, attitude in zip(drawn, anchored):
                on_net = None

                if attitude.plane is not None:
                    on_net = circles
                    circles += 1

                found.append((point, attitude, index, on_net))

        return found

    def _draw_stations(self):
        """Hands the background artist every dot there is."""

        self.stations.set_data(
            [point[0] for point, _, _, _ in self._marked],
            [point[1] for point, _, _, _ in self._marked],
        )

    def _segments(self, carrying):
        """The drawn paths of the traces on one side of `carries`."""

        return [
            path
            for path, structure in zip(self._drawn, self.document.dataset.structures)
            if len(path) > 1 and carries(structure) is carrying
        ]

    def _reset_traces(self):
        """
        Hands both collections their geometry again.

        Both, and not the one that was edited: applying a block can add the first
        attitude to a trace, and the trace then belongs to the other collection.
        A pale line that has just been given a plane has to stop being pale, or
        the map is answering last minute's question.
        """

        for carrying, collection in self.traces.items():
            collection.set_segments(self._segments(carrying))

    def _legend_handles(self):
        # The two weights are switchable, and taking the pale one off is the
        # third way to the same place the table's filter and its sorting lead:
        # 348 contacts off the map, and what is left is the forty-five with
        # something written on them. The steering artists below are not -- a
        # highlight switched off would leave a selection with nothing to show it.
        return [
            self.map_view.switchable(
                Line2D(
                    [], [],
                    color=CARRYING_STYLE["colors"],
                    lw=CARRYING_STYLE["linewidths"],
                    label="carrying a plane",
                ),
                self.traces[True],
            ),
            self.map_view.switchable(
                Line2D(
                    [], [],
                    color=BARE_STYLE["colors"],
                    lw=BARE_STYLE["linewidths"],
                    label="mapped, nothing read",
                ),
                self.traces[False],
            ),
            Line2D([], [], color="#ff7f0e", lw=2.6, label="selected"),
            Line2D(
                [], [], color=PROVENANCE_TINT["rifiutata"], lw=3.4,
                label="rejected stretch",
            ),
            Line2D(
                [], [], color=PROVENANCE_TINT["misurata"], marker="o",
                markersize=5, linestyle="none", label="measured",
            ),
            # The only entry here for something that is not in the file, and the
            # one the legend has to carry hardest: every other colour on this map
            # means a decision somebody made, and this one means a line nobody
            # has applied yet.
            Line2D(
                [], [], color=CLAIMED_TINT, lw=CLAIMED_WIDTH,
                alpha=CLAIMED_ALPHA, label="claimed in the box",
            ),
            # The same purple as the entry above it, and that is the point:
            # everything on this map that is not in the file yet is this colour,
            # and the two are told apart by weight, as the two weights of trace
            # are. One is the ground being claimed, the other is what is being
            # claimed about it.
            Line2D(
                [], [], color=CLAIMED_TINT, lw=CUTTING_WIDTH,
                label="that plane, on the DEM",
            ),
        ]

    def _opening(self):
        dataset = self.document.dataset
        said = (
            f"{len(dataset.structures)} structure(s), "
            f"{sum(1 for s in dataset.structures if carries(s))} carrying a plane: "
            f"{sum(len(s.attitudes) for s in dataset.structures)} attitude(s), "
            f"{sum(len(s.fits) for s in dataset.structures)} fit(s)"
        )

        if dataset.observations:
            # Preserved and not editable: an observation attaches to no path, so
            # there is no trace to select it on and no ruler to draw it along.
            # It stays in the file untouched, and is counted here so that it is
            # a thing left alone rather than a thing gone missing.
            said += f", {len(dataset.observations)} loose observation(s), kept as they are"

        if not dataset.crs:
            said += "; no CRS declared, read as the session's"

        return (
            f"{said}. Click a trace to select it, shift-click to pick an anchor; "
            f"or pick a row in the table, which brings its trace into view."
        )

    # -- selection ---------------------------------------------------------

    def select(self, index):
        """
        Puts one structure under the hand, on the map and in the panel.

        The panel is asked first and can say no, which is what keeps a click on
        the map from redrawing over a block that was typed and not applied. When
        it does, nothing here moves: the highlight stays on the structure whose
        text is still in the box, so the two never disagree about which one is
        being worked on.
        """

        if not self.panel.show_index(index):
            return

        self.index = index

        structure = self.document.dataset.structures[index]
        drawn = self._drawn[index]

        # `[y for _, y in ...]`, and it read `[y for y, _ in ...]` from the day
        # this was written: the name bound is the first of the pair whatever it
        # is called, so the highlight was drawn at (easting, easting) -- off the
        # map, on a diagonal no extent here ever covers. Selecting a trace worked
        # and showed nothing, which is not a thing the eye reads as a bug in the
        # drawing. `check_editor` now asserts the highlight is on the trace.
        self.highlight.set_data([x for x, _ in drawn], [y for _, y in drawn])

        self._mark_refusals(structure)
        self._mark_attitudes(index)
        self.picked.set_data([], [])

        # A hand-placed pin belongs to the trace it was placed on, so it does
        # not survive moving to another one: kept, it would hang a plane over
        # one fault while the band measured it against a second, and both
        # pictures would look exactly as they do when they are right.
        self._free_pin = None
        self._ground = self._ground_for = None
        self.steering.set_pinned(False)

        # And laid again rather than blitted, because the panel has already
        # shown the new block and steered off the old pin on the way in: a plain
        # blit here would leave that frame on the map, drawn against a trace
        # that is no longer the selected one.
        self._resteer()

        self._fill_net(index)

    def _show_claimed(self, interval):
        """
        The stretch the line being written claims, drawn on the ground it claims.

        Two picked anchors put two coordinates on a line, and what they enclose
        is the thing being decided -- so it is the thing Apply should be asked
        about, and until now nothing showed it. A coordinate is not a stretch,
        and neither is a pair of them until somebody walks the path between.

        **The panel's index and not this window's**, which is not
        interchangeable here: `select` asks the panel first and sets its own
        index afterwards, so a block redrawn on the way in emits this while
        `self.index` still names the trace being left. The interval was measured
        against the panel's path, and it has to be drawn on the same one or it
        is drawn on a different fault -- which would look like an answer.

        Nothing is drawn for a reversed pair. `Span.covers` holds it over no
        ground at all, and an empty highlight is what no ground looks like; the
        panel says so in words, because that is the one thing a picture here
        cannot.
        """

        index = self.panel.index

        if interval is None or index is None or interval[0] > interval[1]:
            self.claimed.set_data([], [])
        else:
            path = self.document.dataset.structures[index].path
            drawn = self.on_map(stretch(path, *interval))

            self.claimed.set_data([x for x, _ in drawn], [y for _, y in drawn])

        # And the plane hangs off the middle of that stretch, so a stretch that
        # moved is a plane that has to be laid again. Through here rather than a
        # blit of its own, because a blit is the expensive half of a frame and
        # two of them for one change is one too many.
        self._resteer()

    def _resteer(self):
        """Lays the steered plane again if there is one, and puts the frame up."""

        if self.steering.armed():
            self._steer(*self.steering.plane())
        else:
            self.map_view.blit()

    def _show_held(self, plane):
        """
        Puts the caret's own plane on the dial, and says whether it can take one.

        This is the half of the loop that runs backwards, and it is the more
        useful half on a file that already has planes in it: clicking into a
        `fit` somebody computed shows that plane cutting the ground it was
        computed over. Whether it runs along the trace is the question the
        diagnostics answer with a number, asked by looking instead.

        Silent on the way in -- `show_plane` does not emit -- and then laid
        again here, deliberately: a control that answered on being set would
        redraw once per character typed into the plane, which is four redraws to
        write `140.5`, three of them of numbers nobody meant.
        """

        self.steering.set_writable(self.panel.has_plane_slot())

        if plane is not None:
            self.steering.show_plane(*plane)

        if self.steering.armed():
            self._steer(*self.steering.plane())

    def _steering_armed(self, on):
        """The box turning the steering on and off."""

        self.steering.set_writable(self.panel.has_plane_slot())

        if on:
            self._steer(*self.steering.plane())
            return

        self._clear_steering()
        self.map_view.blit()

    def _clear_steering(self):
        """Everything the steering draws, taken off the map. Does not blit."""

        self.cutting.set_data([], [])
        self.pin.set_data([], [])
        self.agreeing.set_segments([])
        self._agreeing_for = None
        self._joined = np.zeros(0, dtype=bool)
        self._band_xy = np.zeros((0, 2))

    def pin_freely(self, x, y):
        """
        Puts the plane's pin on the selected trace, wherever the hand says.

        The gesture the tool was missing, and the reason is an order of work.
        Everything else here pins the plane from the line under the caret --
        the middle of the stretch it claims, or an attitude's anchor -- so a
        plane could not be steered until somebody had already written down
        which stretch it was about. That is backwards: which stretch is the
        conclusion. This lets the plane be laid at a point, turned until the
        cut runs along something, and only then attributed.

        Snapped to the trace and not left where the mouse was, for the same
        reason `pick` snaps an anchor: the elevation under it is taken from the
        DEM because the trace is a contact somebody walked, and that sentence
        is only true of a point the trace passes through. A pin fifty metres
        off the line would be a plane through ground nobody stood on, drawn in
        the same purple as one that is.

        The window is sized from what is on screen now and kept that way, so
        that turning the dial afterwards does not resize the ground being cut.
        """

        if self.index is None:
            self.say("nothing selected to pin a plane on")

            return

        structure = self.document.dataset.structures[self.index]

        if len(structure.path) < 2:
            self.say(f"{structure.ident} has no path to pin on")

            return

        if self.steering.refusal:
            self.say(self.steering.refusal)

            return

        s, distance = place_on(structure.path, x, y)
        snapped = point_on(structure.path, s)

        left, right = self.map_view.axes.get_xlim()

        self._free_pin = (snapped[0], snapped[1], abs(right - left))
        self.steering.set_pinned(True)

        said = (
            f"plane pinned at {s:.0f} m of {structure.ident}, {distance:.0f} m "
            f"from where you clicked"
        )

        if not self.steering.armed():
            said += " -- switch the steering on to lay a plane there"

        self.say(said)
        self._resteer()

    def unpin(self):
        """Gives the pin back to the line under the caret."""

        if self._free_pin is None:
            return

        self._free_pin = None
        self.steering.set_pinned(False)
        self.say("the plane is back on the line the caret is on")
        self._resteer()

    def _repin(self):
        """
        Where the plane hangs and the crop it is cut against. True if there is one.

        The elevation comes off the DEM, which is the one choice this makes for
        the curator and the right one here: the trace is a contact somebody
        walked on the ground, so the plane through it passes through the ground.
        The other tool can lift a plane off the topography onto a projected
        horizon and needs a box to say so; here that would be a plane through a
        point the trace does not pass through, which is not what any of this is
        being asked.

        On nodata there is no pin. Refusing is right rather than falling back on
        a median elevation: the cut would be drawn, would look like an answer,
        and would be a plane through a point nobody chose.
        """

        dem = self.session.dem

        if dem is None:
            asked = None
        elif self._free_pin is not None:
            asked = (self._free_pin[:2], self._free_pin[2])
        else:
            asked = self.panel.pinned_at()

        if asked is None:
            self._pin = self._cut_window = self._cut_for = None

            return False

        (x, y), span = asked
        z = dem.elevation_at(x, y)

        if z is None:
            self._pin = self._cut_window = self._cut_for = None

            return False

        self._pin = (x, y, z)

        wanted = (x, y, side_for(span, max(dem.res_x, dem.res_y)))

        if wanted != self._cut_for:
            self._cut_window = dem.window_at(*wanted)
            self._cut_for = wanted

        return True

    def _steer(self, dip_dir, dip):
        """One frame of the plane on the topography: the kernel, and the blit."""

        if not self.steering.armed():
            return

        if not self._repin():
            self._clear_steering()
            self.map_view.blit()
            self.steering.note(
                "nothing to lay a plane on: ctrl-click the trace to pin one "
                "anywhere, or put the caret on a line that claims a stretch or "
                "carries an anchor"
            )

            return

        convergence = self.session.convergence.at(*self._pin[:2])

        laid = laid_on(
            self._cut_window,
            self._pin,
            dip_dir,
            dip,
            nodata=self.session.dem.nodata,
            convergence=convergence,
        )

        gaps = self._show_agreement(dip_dir, dip, convergence)

        if laid.chords:
            xs, ys = laid.points[:, 0], laid.points[:, 1]

            # Reprojected before the NaNs go in and not after: pyproj turns a
            # NaN into an infinity, so a path already broken into chords cannot
            # make this crossing. In this session it is almost always the
            # identity -- a DEM is refused unless it is in the traces' own
            # projection -- but "almost always" is not a thing to draw on.
            if self._forward is not None:
                xs, ys = self._forward.transform(xs, ys)

            self.cutting.set_data(*broken_path(xs, ys, laid.segments))
        else:
            self.cutting.set_data([], [])

        at = self.on_map([self._pin[:2]])[0]
        self.pin.set_data([at[0]], [at[1]])

        self.map_view.blit()

        # Into the steering's own label and never the status bar. This fires on
        # every step of a dial, the bar is one line, and the bar is where the
        # panel answers for what was last pressed -- a per-frame report there is
        # the mistake the claimed stretch already made once, ninety times a turn.
        said = laid.describe()

        if gaps is not None:
            said += "\n" + gaps.describe()

        self.steering.note(said)

    def _ground_now(self):
        """
        The topography along the selected trace, inside the window being cut.

        Cached against the pair it was read for, which is the whole reason the
        band can be drawn inside a frame: the DEM half of it is a gather over
        the window and the plane half is six multiplications, so a dial turned
        ninety times should do the second ninety times and the first not at all.

        Sampled at the DEM's own cell, because that is the finest the cut can be
        located anyway -- `walked` thins what comes out of the window if the
        trace wanders far enough through it to need it.
        """

        if self.index is None or self._cut_window is None:
            return None

        for_now = (self.index, self._cut_for)

        if for_now == self._ground_for:
            return self._ground

        dem = self.session.dem
        cell = max(dem.res_x, dem.res_y)
        structure = self.document.dataset.structures[self.index]

        s, xy = walked(structure.path, box=self._cut_window.bounds, step=cell)

        self._ground = ground_on(
            self._cut_window, s, xy, nodata=dem.nodata, cell=cell,
            whole=structure.length,
        )
        self._ground_for = for_now

        return self._ground

    def _show_agreement(self, dip_dir, dip, convergence):
        """
        How near the cut runs to the trace, drawn along the trace. The gaps, or None.

        The band is the answer to the question the picture alone cannot settle.
        A cut and a trace lying on top of each other is what agreement looks
        like, but at 1:100000 thirty metres of it is one pixel, a window holds a
        dozen curves of which only one is the one near the trace, and none of
        that says *where along the trace* the agreement stops -- which is the
        number about to be written as the ends of a `fit`.

        It draws and does not decide. The ends stay a gesture, and that is not
        timidity: the same measurement that finds a long stretch of good
        agreement finds the longest one of all where the plane is lying down on
        the hillside and the cut would follow whatever it was put on. A
        threshold reading this column would propose that stretch most
        confidently of all. `planes.FLAT` is why nothing is drawn there instead.
        """

        ground = self._ground_now()

        if ground is None or not len(ground):
            self.agreeing.set_segments([])

            return None

        gaps = gaps_on(ground, self._pin, dip_dir, dip, convergence=convergence)

        # The reprojection is what is cached, not the segments: the segments
        # depend on the plane now, because runs of one step get drawn as one
        # polyline, and the steps move as the dial turns. This does not.
        if self._agreeing_for != self._ground_for:
            self._band_xy = np.asarray(self.on_map(ground.xy), dtype=float)

            # Which samples are actually neighbours. `ground_on` drops what
            # falls on nodata or against the window's rim, so two consecutive
            # rows can be a kilometre apart -- and a band drawn across that
            # would be the one place it spoke confidently about ground it had
            # refused to read.
            self._joined = np.diff(ground.s) <= 1.5 * (ground.step or 1.0)
            self._agreeing_for = self._ground_for

        if len(self._band_xy) < 2 or not self._joined.any():
            self.agreeing.set_segments([])

            return gaps

        # The worse of a segment's two ends and not their average: the band is
        # read for where the agreement stops, and a mean would carry the last
        # good sample half a step into ground that has already lost it.
        close = np.minimum(gaps.close[:-1], gaps.close[1:])

        # `ceil`, so nought is the only thing that lands on nought: a sample
        # with any agreement at all gets the faintest step rather than being
        # rounded out of the picture, and only a sample the gap refused is not
        # drawn.
        level = np.ceil(np.clip(close, 0.0, 1.0) * AGREEING_LEVELS).astype(int)
        level[~self._joined] = 0

        # One polyline per run of equal step. A run of `k` samples covers
        # `k + 1` points, which is why the slice goes one past the end.
        change = np.flatnonzero(np.diff(level)) + 1
        segments, alphas = [], []

        for start, end in zip(
            np.concatenate(([0], change)),
            np.concatenate((change, [len(level)])),
        ):
            if level[start] <= 0:
                continue

            segments.append(self._band_xy[start:end + 1])
            alphas.append(level[start] / AGREEING_LEVELS * AGREEING_ALPHA)

        colours = np.empty((len(alphas), 4))
        colours[:, :3] = to_rgb(CLAIMED_TINT)
        colours[:, 3] = alphas

        self.agreeing.set_segments(segments)
        self.agreeing.set_color(colours)

        return gaps

    def _take_plane(self):
        """
        The steered attitude, into the line the caret is on.

        What goes with it is the provenance, and that is not decoration:
        FORMAT.md's rule for a derived plane is that it names the producer that
        made it and does not fill in another's diagnostics. This one has no gate,
        no residual and no window swept, so it writes none of `snr`, `flat` or
        `jack`; what it has is somebody who looked at two lines and judged them
        to run together, and `from=plane-dem` is the honest name for that.

        **And it writes which north the number is measured from**, which no line
        in these files currently does. The dial is a true azimuth, as a compass
        is and as the other tool's is; the DEM is on the grid; and around here
        the two are 0.4 to 1.0 degrees apart. That is small against everything
        else on these traces and it is not nothing, and a number whose reference
        is written down can be argued with later, where one without cannot.
        """

        if not self.steering.armed() or self._pin is None:
            return

        dip_dir, dip = self.steering.plane()
        convergence = self.session.convergence.at(*self._pin[:2])

        written = self.panel.take_plane(dip_dir, dip, {
            "from": FROM_STEERED,
            "src": "gsurf",
            "dem": self.session.dem.path.name,
            "north": "true",
            "converg": f"{convergence:+.2f}",
            "at": f"@{self._pin[0]:.2f},{self._pin[1]:.2f}",
        })

        if written is None:
            self.say("the line the caret is on has no plane to write into")

            return

        self.say(f"written: {written.strip()} -- Apply to keep it, then Save")

    def _mark_refusals(self, structure):
        """The stretches somebody has rejected, drawn where they are."""

        xs, ys = [], []

        for span in structure.spans:
            if span.axis != "use" or span.value != "rejected" or span.s0 is None:
                continue

            for x, y in self.on_map(stretch(structure.path, span.s0, span.s1)):
                xs.append(x)
                ys.append(y)

            # A break, so two rejected stretches do not join across the good
            # ground between them.
            xs.append(np.nan)
            ys.append(np.nan)

        self.refused.set_data(xs, ys)

    def _mark_attitudes(self, index):
        """The selected fault's own dots, drawn again at full size over the rest.

        Drawn twice, which is what the map already does with `traces` and
        `highlight`: the background holds every dot there is and cannot be redrawn
        without a full draw, so the handful belonging to the fault under the hand
        go in an animated artist above it. Above and larger, so that the pair lands
        as one dot of the bigger size rather than as two.
        """

        here = [point for point, _, whose, _ in self._marked if whose == index]

        # The same reversed pair as in `select`, and the same consequence: the
        # green dot marking where a plane was measured was never on the fault.
        self.marks.set_data([x for x, _ in here], [y for _, y in here])

        # Dropped because the cursor has not moved and the answer has: a tooltip
        # naming a station on the fault that was selected a moment ago is now
        # naming one on a fault the hand is no longer on, and the circle the net
        # was pointing at is not on the net any more.
        self._tip_on(None)
        self._netted = None

    def _on_applied(self, index):
        """A block that parsed: the map has to agree with it again."""

        self._drawn[index] = self.on_map(self.document.dataset.structures[index].path)

        # The static artists hold a copy of the geometry, so a path edited in the
        # box has to be handed over again -- and then a full draw, which is what
        # recaptures the background the rest is blitted over.
        #
        # The dots are rebuilt for the whole file and not for the block that was
        # applied. An edit that moves a path moves the dots on it, an edit that
        # adds an `attitude` line adds one, and the list is 23 tuples: working out
        # which of those happened would cost more to write than redoing it.
        self._reset_traces()
        self._marked = self._all_stations()
        self._draw_stations()
        self.map_view.canvas.draw()

        self.select(index)
        self._retitle()
        self.say(
            f"{self.document.dataset.structures[index].ident} applied; "
            f"the file is written by Save"
        )

    # -- where the view is -------------------------------------------------

    def frame_on(self, index):
        """
        Asks for the view to move onto one trace, in a moment.

        Deferred rather than done, because a run of selections is one gesture:
        holding the down arrow in the table is a row a keystroke, and each of
        these is a full redraw and a hillshade reread for a view nobody stopped
        to look at.
        """

        self._framing = index
        self._frame_timer.start(FRAME_DELAY_MS)

    def frame_now(self, index=None):
        """Moves the view onto a trace, and says where it went."""

        self._frame_timer.stop()

        if index is None:
            index, self._framing = self._framing, None

        if index is None:
            return False

        structure = self.document.dataset.structures[index]
        extent = framing_for(self._drawn[index])

        if extent is None:
            self.say(f"{structure.ident} has no path to frame on")
            return False

        self.map_view.restore_framing(extent)

        # The way back said out loud, because this is the one thing in the window
        # that moves the map without the hand having moved it.
        self.say(
            f"framed on {structure.ident}, {structure.length:.0f} m "
            f"-- the back arrow returns to where you were"
        )

        return True

    # -- what the cursor is resting on --------------------------------------

    def _on_map_hover(self, x, y):
        """
        A cursor resting on a station dot, or on nothing.

        Two answers about the same dot, and both die with the cursor: the tooltip
        says which station it is and the net says which of the circles already on
        it is that station's plane. Neither outlives the hover, which is why
        `hover_off` arrives here as a hover over nothing rather than at a second
        slot -- a highlight left behind by a cursor that has gone would be
        pointing at a circle for no reason anybody could see.

        What does not die with it is the net's contents. Those belong to the
        selected structure, and the cursor never chooses them: it chooses which of
        them to point at. That is the whole division, and it is what makes the
        answer to "whose planes are these" a thing the title can state once.
        """

        which = None if x is None else self._station_near(x, y)

        self._tip_on(which)
        self._mark_on_net(which)

    def _station_near(self, x, y):
        """
        Which drawn station dot the cursor is on, as an index, or nothing.

        In display pixels, through the axes' own transform, because the dots are
        five points across whatever the view is showing: a reach in metres would
        be a target a kilometre wide framed on the whole AOI and unhittable
        framed on one fault.

        The nearest of the ones in reach and not the first. The two anchors on
        F0273 are 26 m apart -- one dot at most scales, two that overlap just
        before they separate -- and which of them is meant is the nearer one.
        """

        if not self._marked:
            return None

        transform = self.map_view.axes.transData
        here = transform.transform((x, y))
        offsets = (
            transform.transform([point for point, _, _, _ in self._marked]) - here
        )
        squared = (offsets * offsets).sum(axis=1)
        nearest = int(np.argmin(squared))

        if squared[nearest] > HOVER_RADIUS_PX**2:
            return None

        return nearest

    def _tip_on(self, which):
        """
        Puts the tooltip on one station dot, or takes it off.

        Guarded on which dot and not on where the cursor is, because this runs on
        every motion event the canvas sees: crossing a dot is two changes and the
        hundreds of pixels either side of it are none, so the text is built when
        the answer changes rather than when the mouse moves. Which makes a motion
        event 0.032 ms when the answer stands and 0.063 when it does not -- the
        guard is not what makes this affordable, it is what keeps a tooltip from
        being rebuilt sixty times a second while the cursor sits still.

        Qt's own tooltip, so it appears after the delay everything else on this
        desktop appears after and goes away without being told to. What that
        costs is that a tooltip already on screen does not re-read its text, so
        moving between two dots close enough to be under one tooltip can show the
        first one's -- which is why they are 10 pixels apart at most and why this
        is the place it would be noticed.
        """

        if which == self._tipped:
            return

        self._tipped = which
        self.map_view.canvas.setToolTip(
            "" if which is None else self._station_tip(*self._marked[which][1:3])
        )

    def _fill_net(self, index):
        """
        Puts one fault's planes on the net: what was measured and what was fitted.

        All of them and not the one under the cursor, which is what this did
        first. A great circle is what the two numbers in the table already say and
        the net says differently -- `145/35` is a plane you have to picture -- but
        that is the smaller half of it, and it is the half a tooltip can do. The
        half it cannot is agreement: of the forty-five faults of `merid_faults`
        that carry anything, seven carry two planes or more and six carry both a
        reading and a fit, and whether two planes are the same surface is a
        question no pair of columns answers. F0074 is why -- a compass reading of
        135/30, an `exposed-facet` fit at 141/29, and a `trace-dem` fit at 221/10
        with `caveat=immersione non vincolata dalla traccia` written beside it.
        Two circles almost on top of each other and a third across the net says
        that at a glance; three rows of numbers make you do it in your head.

        Measured and fitted go on as two lists, drawn solid and dashed, because
        the difference between them is the difference between a plane somebody put
        a compass on and one least-squares fitted to a scatter. Counting the fits
        is what makes the net worth opening at all: measurements alone leave it
        empty on 373 of the 393 selections, and with the fits that falls to 348.

        Whose planes they are is in the title, once, and it does not change while
        the cursor moves. The alternative -- a title following the hover, which is
        what it did -- flickered the window caption on a gesture that happens sixty
        times a second, to say a thing the tooltip was saying anyway.
        """

        structure = self.document.dataset.structures[index]

        # Out of the global list and filtered here, so the circles come out in the
        # order the dots were numbered in: the fourth element of an entry *is* its
        # place among these, and reading the planes off `structure.attitudes`
        # instead would be a second filter free to disagree with the first.
        measured = [
            attitude
            for _, attitude, whose, on_net in self._marked
            if whose == index and on_net is not None
        ]
        fitted = [fit for fit in structure.fits if fit.plane is not None]

        # Pooled across the planes, not split by which one they were read on: a
        # stria drawn on the net lies in the plane it was read on, so the circle it
        # belongs to is the one it is sitting on. See `show_planes`.
        lineations = [
            found
            for attitude in measured
            for found in self._lineations_at(structure, attitude)
        ]

        self.net_window.setWindowTitle(
            self._net_title(structure, measured, fitted, lineations)
        )
        self.net.show_planes(
            measured=[(a.plane.dip_dir, a.plane.dip) for a in measured],
            fitted=[(f.plane.dip_dir, f.plane.dip) for f in fitted],
            lineations=lineations,
            marked=self._netted,
            measured_color=PROVENANCE_TINT["misurata"],
            fitted_color=PROVENANCE_TINT["fit"],
        )

    def _net_title(self, structure, measured, fitted, lineations):
        """What the net's window says it is showing."""

        if not measured and not fitted:
            return f"{NET_TITLE} - {structure.ident} - nothing read"

        counted = []

        if measured:
            counted.append(f"{len(measured)} measured")

        if fitted:
            counted.append(f"{len(fitted)} fitted")

        if lineations:
            counted.append(f"{len(lineations)} lineation(s)")

        return f"{NET_TITLE} - {structure.ident} -- {', '.join(counted)}"

    def _mark_on_net(self, which):
        """
        Points at the circle of the station the cursor is on, or at no circle.

        Guarded like the tooltip, and for a stronger reason: this is a `set_data`
        and a blit on another canvas, which is a thousand times a motion event's
        own cost, and a motion event is what the cursor sitting still on a dot
        produces sixty times a second.

        A dot with no plane behind it points at nothing, which is `None` and not a
        refusal: the tooltip still says what that station is, and the net says --
        by dimming nothing and highlighting nothing -- that there is no circle on
        it to show. That is also what a cursor over bare map says, and the two
        being the same answer is right: neither is pointing at a plane.

        Drawn whether the window is open or not, which is the opposite of what the
        fold tool does with the same widget. The difference is what makes it
        redraw: there it is every frame of a drag, and a hidden canvas costs frame
        budget and makes the frame cost that tool reports a measurement of
        something nobody can see. Here it is a cursor crossing onto a dot.
        """

        on_net = None

        if which is not None:
            _, _, whose, on_net = self._marked[which]

            # A dot on some other fault points at nothing, because the net is not
            # showing that fault. Not an error and not a refusal: the tooltip still
            # answers, which is the whole reason the dot is drawn at all, and the
            # net saying nothing is the true answer to "which circle is this" when
            # none of them is.
            if whose != self.index:
                on_net = None

        if on_net == self._netted:
            return

        self._netted = on_net
        self.net.mark(on_net)

    def _lineations_at(self, structure, attitude):
        """
        The lineations read at the same place as one plane, as `(trend, plunge)`.

        By station code where the lineation carries one, and by distance along the
        trace only where it does not. That order is not a preference: `off` runs to
        69.7 m in `merid_faults`, so two readings made standing in one spot can
        snap to progressives seventy metres apart, and a rule that went by
        distance first would fail on exactly the outcrops the tooltip exists to
        warn about. `SAME_OUTCROP_M` is what is left for a lineation that says
        nothing about where it was read.

        Plural on purpose. `merid_faults` has a station whose note reads "strie
        osservate: vedi foglio (pitch 1 30 deg, p. 2 80 deg)": two sets of striae
        on one surface, which is two movements and the reason this returns a list
        rather than the one lineation a station usually has. It currently returns
        an empty list for all 23 of them, there being no `lineation` record in the
        file at all -- the striae that were read are prose inside `note=`, and
        turning prose into a record is curation and not something to guess at
        while drawing.

        The structure is a parameter and not `self.index`, which is what it read
        while the net was filled by the cursor. The same value, since the dots have
        only ever been the selected fault's -- but it was the one line in the net's
        half of this tool that could not be called with a structure in hand, and
        the net is filled with one in hand now.
        """

        station = attitude.attrs.get("station")
        found = []

        for lineation in structure.lineations:
            if lineation.trend is None or lineation.plunge is None:
                continue

            labelled = lineation.attrs.get("station")

            if labelled is not None:
                near = station is not None and labelled == station
            else:
                near = (
                    lineation.s is not None
                    and attitude.s is not None
                    and abs(lineation.s - attitude.s) <= SAME_OUTCROP_M
                )

            if near:
                found.append((lineation.trend, lineation.plunge))

        return found

    def _station_tip(self, attitude, whose):
        """
        What a station dot is, in the order somebody pointing at it wants it.

        The answer first -- which station, what plane -- then where along the
        trace, and then the one thing the dot cannot say about itself. `off` is
        how far the reading was from the line it has been snapped onto, and over
        the 23 measurements of `merid_faults` it runs from 0.0 to 69.7 m. Seventy
        metres is inside the width of the line at 1:25000 and a visible lie at
        1:5000; the dot looks the same either way, and the number is not in the
        block either -- it is an attribute nobody reads unless it is put in front
        of them.

        Then a refusal, where the curation has one. A rejected measurement stays
        drawn, because somebody did stand there and that does not stop being
        true, but it is not what holds -- and until now the band in the panel was
        the only thing that said so, which is a different window.

        `whose` is the fault the dot is on, and it is a parameter because it is no
        longer the selected one. This read `self.index` while the only dots drawn
        were the selection's, which was right then and became a lie the moment
        every station in the file got a dot: the line naming the trace would have
        named whichever fault happened to be open. Saying so where it differs, and
        saying it here, because the line already existed -- what a dot on another
        fault needs is not a new sentence but the one sentence to be true.
        """

        attrs = attitude.attrs
        station = attrs.get("station") or "no station code"
        structure = self.document.dataset.structures[whose]

        lines = [
            station if attitude.plane is None else f"{station} -- {attitude.plane}",
            f"{attitude.s:.0f} m along {structure.ident}"
            + ("" if whose == self.index else ", which is not the selected trace"),
        ]

        off = _as_number(attrs.get("off"))

        if off is not None:
            lines.append(
                "read on the trace"
                if off == 0.0
                else f"read {off:.1f} m off the trace, and snapped onto it"
            )

        # `attitude_at` at the anchor's own progressive: no measurement can be
        # nearer to it than it is to itself, so the only answer that is not this
        # one is a refusal.
        _, said = structure.attitude_at(attitude.s, self.panel.max_gap)

        if said.startswith("rifiutata"):
            lines.append(f"overruled here -- {said}")

        stamp = " ".join(
            part for part in (attrs.get("src", ""), attrs.get("date", "")) if part
        )

        if stamp:
            lines.append(stamp)

        for key in ("note", "site_note"):
            if attrs.get(key):
                lines.append(textwrap.fill(attrs[key], TIP_WRAP))

        return "\n".join(lines)

    # -- the map -----------------------------------------------------------

    def _on_map_pressed(self, x, y):
        modifiers = QtWidgets.QApplication.keyboardModifiers()

        # Ctrl before shift, so a hand holding both gets the pin rather than an
        # anchor. Neither order is obviously right; what matters is that it is
        # decided here and not by which `if` happens to be first in `pick`.
        if modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
            self.pin_freely(*self.in_file(x, y))

            return

        self.pick(
            x, y,
            anchor=bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier),
        )

    def pick(self, x, y, anchor=False):
        """
        A click on the map: the trace it was aimed at, or an anchor on this one.

        Shift-click projects onto the structure that is *selected*, whichever
        trace is nearest, because the anchor is about to be written into that
        block and an anchor snapped onto a neighbour would read back as a
        progressive on a fault it was never measured on.

        That rule is older than the dots being global, and the dots are what put
        it at risk. A green dot on a neighbouring fault is now a visible thing to
        aim at, and aiming at one writes a progressive on the selected trace --
        correctly, silently, and not where the eye was. So when the click lands
        nearer some other trace than the one it is being written onto, the line
        this says names that other trace instead of confirming the anchor. Not a
        refusal: a click 300 m off a trace is how you anchor the end of a fault
        that runs past a closer one, and nothing here can tell that from a
        mis-aim. What it can do is stop reading like a confirmation.
        """

        here = self.in_file(x, y)
        reach = self._reach_in_metres()

        if anchor:
            if self.index is None:
                self.say("nothing selected to anchor on")
                return

            structure = self.document.dataset.structures[self.index]

            if len(structure.path) < 2:
                self.say(
                    f"{structure.ident} has no path to anchor on"
                )
                return

            s, distance = place_on(structure.path, *here)
            snapped = point_on(structure.path, s)

            self.panel.insert_anchor(*snapped)

            drawn = self.on_map([snapped])[0]
            self.picked.set_data([drawn[0]], [drawn[1]])
            self.map_view.blit()

            said = (
                f"@{snapped[0]:.2f},{snapped[1]:.2f} -- {structure.ident} at "
                f"{s:.0f} m, {distance:.0f} m from where you clicked"
            )

            # Measured against every trace and not against the ones in reach: the
            # reach dial is how far a *measurement* answers for, and borrowing it
            # here would make the warning appear and disappear as that is turned.
            nearer = nearest_structure(self.document.dataset, *here)

            if nearer is not None and nearer[0] != self.index and nearer[2] < distance:
                said += (
                    f" -- but {self.document.dataset.structures[nearer[0]].ident} "
                    f"is nearer, at {nearer[2]:.0f} m; anchors go on the selected "
                    f"trace"
                )

            # What the anchor did to the line it landed in, which is what the
            # click was for. `insert_anchor` has already run, so the panel's
            # claim is the one this click just made -- and without this the
            # report would end here: `say` is one bar, so a message written after
            # the panel's own takes the panel's away, and a shift-click would
            # answer with a coordinate and silently drop the extent.
            claim = self.panel.claim_said()

            if claim is not None:
                said += f"; {claim}"

            self.say(said)
            return

        found = nearest_structure(self.document.dataset, *here, within=reach)

        if found is None:
            self.say("no trace within reach of the click")
            return

        index, s, _ = found

        self.select(index)
        self._report_at(index, s)

    def _reach_in_metres(self):
        """`PICK_RADIUS_PX` as ground metres at the framing now on screen."""

        axes = self.map_view.axes
        left, right = axes.get_xlim()
        width = axes.bbox.width or 1.0

        return abs(right - left) / width * PICK_RADIUS_PX

    def _report_at(self, index, s):
        structure = self.document.dataset.structures[index]
        plane, said = structure.attitude_at(s, self.panel.max_gap)

        self.say(
            f"{structure.ident} at {s:.0f} m of {structure.length:.0f} -- {said}"
            + ("" if plane is None else f", {plane}")
        )

    # -- the file ----------------------------------------------------------

    def _retitle(self):
        mark = "*" if self.document.dirty else ""

        self.setWindowTitle(f"gSurf - trace editor - {self.document.path.name}{mark}")

        # The panel carries the file's name and the star as well, because Save is
        # in the panel and so is the typing: a window that holds unwritten work
        # should say so on its own frame, not only on the map's.
        self.panel_window.setWindowTitle(self._panel_title())

    def save(self):
        """Writes the document to the file it came from."""

        return self._write(None)

    def save_as(self):
        # Over the panel and not over the map: Qt puts a dialog on the screen its
        # parent is on, and both of these are about the file -- which is what the
        # panel window is. Asked for from the map by Ctrl+S, the answer still
        # belongs beside the buttons that otherwise ask it.
        name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.panel_window,
            "Save the file as",
            str(self.document.path),
            "gstruct (*.gstruct)",
        )

        return self._write(name) if name else False

    def _write(self, target):
        try:
            written = self.document.save(target)
        except OSError as err:
            QtWidgets.QMessageBox.critical(self.panel_window, "Not written", str(err))
            return False

        self._retitle()
        self.say(f"written to {written}")

        return True

    def closeEvent(self, event):
        """
        A document with unapplied or unwritten work is asked about, once.

        Two different losses, and the one that catches people is the first: a
        block typed into the box and never applied is not in the document, so
        Save would write the file without it and report success.
        """

        pending = (
            self.panel.index is not None
            and self.panel.text.toPlainText() != self.document.text_of(self.panel.index)
        )

        if self.document.dirty or pending:
            said = "This file has changes that are not written."

            if pending:
                said += (
                    "\n\nThe block on screen has also not been applied: Apply "
                    "reads it into the document, and only then does Save write it."
                )

            answer = QtWidgets.QMessageBox.question(
                self,
                "Not written",
                said + "\n\nWrite it now?",
                QtWidgets.QMessageBox.StandardButton.Save
                | QtWidgets.QMessageBox.StandardButton.Discard
                | QtWidgets.QMessageBox.StandardButton.Cancel,
            )

            if answer == QtWidgets.QMessageBox.StandardButton.Cancel:
                event.ignore()
                return

            if answer == QtWidgets.QMessageBox.StandardButton.Save and not self.save():
                event.ignore()
                return

        # After the question and not before it: a close that was cancelled is not
        # an arrangement anybody finished working in, and writing the geometry
        # there would save the windows as they stood in front of a dialog.
        self.group.save_geometry()

        super().closeEvent(event)


def build(session, chosen, legend="beside"):
    """The window, on a session somebody else has already opened."""

    spec = chosen.get("traces") or {}
    path = spec.get("path")

    if not is_gstruct(path):
        QtWidgets.QMessageBox.critical(
            None,
            "Nothing to edit",
            f"{path}\n\nThis edits a .gstruct, which is a text format with one "
            f"fact per line. A layer has no text to edit and no way to hold a "
            f"span or a fit.\n\nTo get one from a layer: Import - lines to "
            f".gstruct, on the launcher. It asks what the columns mean and "
            f"writes a file, which this then opens.",
        )
        return None

    try:
        document = Document(path)
    except Exception as err:
        QtWidgets.QMessageBox.critical(
            None, "Unreadable", f"{path}\n\n{type(err).__name__}: {err}"
        )
        return None

    if not document.dataset.structures:
        QtWidgets.QMessageBox.critical(
            None, "Nothing to edit", f"{path}\n\nno structure in the file"
        )
        return None

    # Refused at the door rather than handled inside, because there is nothing
    # here that would only half work: the band along the top is metres, the
    # reach is metres, and an anchor written to two decimals of a degree lands
    # somewhere else entirely. A tool that opened anyway would be a tool that
    # writes the file wrong.
    said = degrees_not_metres(document.dataset, getattr(session, "crs", None))

    if said:
        QtWidgets.QMessageBox.critical(
            None, "Degrees, not metres", f"{path}\n\n{said}"
        )
        return None

    window = EditorWindow(session, document, legend=legend)

    # The panel follows from the map's own `showEvent`, so whichever of these two
    # shows it brings the group up with it. `tiled` is the third case: nothing
    # remembered, but a screen the two fit across, so the map has a size already
    # and `fit_to_screen` would undo it -- see `_place_unremembered`.
    if window.group.restore_geometry() or window.tiled:
        window.show()
    else:
        fit_to_screen(window, *MAP_WINDOW_PX)

    return window
