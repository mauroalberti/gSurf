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

**And the axis no producer can fill.** `exposure` is `unknown` on all 393
structures of both files in this project, `reason=assente-in-sorgente`, because
`geology.gpkg` has no such column and nothing computed can invent one. It is
also the axis FORMAT.md hands the one judgement the diagnostics refuse: a fit
that reproduces the hillside is `concorde-col-versante`, which is 17 of the 27
fits in `merid_faults`, and whether that means *the surface is the fault* or
*the trace was drawn along a scarp* is decided by this axis and by nothing else.
So `ExposureHere` exists, and what it shows while the stretch is being picked is
the hillside itself -- `hillside.hillside_on`, which is `drape` moved off the fit
and onto a stretch somebody chose. It draws no conclusion from the angle, and
the measurement is why: over those 27 traces the whole-trace corridor misses its
own best plane by 7 to 210 times what nearest-cell sampling costs, so every
`drape=` written in that file is an angle to a plane the ground does not hold.
Shorten the stretch and the residual comes down. That is a thing to see while
choosing where the claim goes, and not a threshold to be checked against.

**And then the surface itself.** Where a contact is declared exposed the DTM is
not sampling a hill, it is sampling the fault, and `FacetHere` grows the region
whose own slope matches a measured plane and fits a plane to that -- hectares
instead of a line. Three things it does that `gstruct/facets.py` did not, each
because the measurement asked for it. The **radius is a control**: at the
reference's 500 m, five of the eight facets of this AOI are still growing where
the window stops, and S19's plane turns eleven degrees when it is opened to
1000, towards the 107/35 its own compass reads -- so `Grow it further` sweeps a
ladder and reports how far the plane *moved*, which is the only way to tell a
surface measured further out from a surface that was never one plane. The
**seed is marked as the seed** in the angles, because the region is the cells
within the tolerance of that plane and its agreement with the result is
arithmetic, not evidence: `vs_field` in the old script was that angle, read as
corroboration. And the **stretch comes from the licence**, not from the region:
these facets lie a median of 47 to 296 m off their own trace and out to 714 m,
an exposed dip slope running away down the dip, so no projection of one onto the
line is a claim anybody made. The `exposure=exposed` span is a person saying
where this contact crops out, and the fit is written over exactly that, with
`off=` beside it saying how far away the surface measured actually is.

**And a reading written over.** The three gestures above either add a line or
take one out; `AmendReading` changes what a line claims, which is the only act
here with nothing to fall back on -- unless something keeps the old claim, and
what that has to be is settled by the format rather than by preference. `raw=`
holds the source string beside the normalised value, so on all 44 readings in
the AOI the old *plane* is still on the line after the slot is rewritten, a few
characters further along, and the difference between the two is the curation.
The old *anchor* is nowhere, ever: the source geometry is the anchor. So a move
always leaves a comment and a replanning usually does not, the comment says
which rule put it there, and the two exceptions are measured -- a reading typed
in here, which has no `raw=`, and one whose `raw=` an earlier amendment has
already overtaken.

What it shows before the press is what the move does to the *answer*, which is
not the question of where the dot goes: the candidate line is parsed and the
trace sampled either side of it, because a reading outranks every fit for 250 m
and past that still answers wherever no fit does. Two numbers come out and they
are different -- 109 m along `F0055` changes who answers over 106 m and the
plane over none of it, where 281 m along `F0058` moves 839 m of both, S25 lying
16 m from S22 and reading 14 degrees away from it. Counting that took two
corrections from the same data: the tier alone misses S22 giving way to S25, and
the provenance string whole counts the distance it answered from, which changes
at every metre. `curation.answering` is the quantity in between.
"""

from __future__ import annotations

import textwrap
import time
from collections import Counter
from typing import NamedTuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PyQt6 import QtCore, QtGui, QtWidgets

from gsurf.curation import (
    DEFAULT_MAX_GAP,
    ENDS_AT,
    PLANE_DECIMALS,
    PROVENANCE,
    SUFFIX,
    UNCONSTRAINED,
    Document,
    amend_note,
    anchor_of,
    anchors_written,
    answering,
    continued_at,
    covered_metres,
    degrees_not_metres,
    detachment_note,
    fits_in,
    from_a_file,
    interval_of,
    is_gstruct,
    module,
    nearest_structure,
    owed_record,
    place_on,
    plane_of,
    point_on,
    provenance_of,
    reading_line,
    reading_said,
    readings_in,
    rows_of,
    span_line,
    stretch,
    with_anchor,
    with_attrs,
    with_ends_in_order,
    with_plane,
    with_values,
)
from gsurf.convergence import MeridianConvergence
from gsurf.fits import (
    AT_THE_END,
    Sweep,
    as_line,
    dem_refusal,
    fits_along,
    gate_for,
)
from gsurf import facets
from gsurf.hillside import between, hillside_on
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
#- A `fit` for somebody to finish, named because two windows start one: the box's
# own button and the steering's. The steering needs it because the step line kept
# saying `press + fit in the box` to a hand that was in another window, and
# because without a fresh line the caret stays wherever it was parked -- which is
# how a steered plane ended up written over a compass reading.
FIT_TEMPLATE = "  fit plane * * 000/00 from="

# A `span` for somebody to finish, and the last of the three templates the panel
# used to carry as buttons. The other two went where the work went: a plane off
# the topography is the fit window's, a compass reading is `ReadingsHere`'s, and
# both of those write a finished line instead of an abbreviation to fill in. This
# one has nowhere to go, `use` being the one claim in the format with no window of
# its own, so it stays a line somebody writes -- off the menu now rather than off
# a button, which is the only place left that is about the open trace.
SPAN_TEMPLATE = '  span use * * rejected reason=""'

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

# The column on the map's own frame: the legend's two controls, and for a while
# the steering above them. A cap and not a width -- the column asks for what is
# in it and the map takes the rest -- so with the dial moved into the fit window
# the map gets those pixels back without this number having to be guessed again.
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

# And the least box worth having: about six lines of the fixed font, which is a
# heading, a claim and its path's first vertices -- enough to see what kind of
# block is open. It is a floor and not a size; what the box opens at is the outer
# splitter's business.
TEXT_FLOOR_PX = 120

# Above this fraction of the trace, a bar has room for its own word in it.
LABEL_FRACTION = 0.14

# What the fit window shows about each stretch the topography answered for, and
# how big it opens. `keep` first because it is the column being operated: the
# rest is evidence, and the tick is the decision.
FIT_COLUMNS = ("keep", "from", "to", "plane", "read over")

# Taller than it was by about what the steering is, now that the steering is in
# here. `SatelliteWindow` caps this at the screen, and on the 741 px of usable
# height this is written on it is capped: the two tables carry the stretch, so
# what the cap costs is rows, which scroll, and not the buttons at the bottom.
FIT_WINDOW_PX = (560, 800)

# And what it shows about the fits the file already holds there. The same four
# quantities in the same order, so the two tables read as one thing seen twice,
# plus where the line came from -- which the reading's table does not need,
# every row in it having come from here.
#
# `how` and not `from`, that being taken by the near end of the stretch. It is
# the file's own `from=`: `trace-dem`, `plane-dem`, `table`, `reach`.
CARRIED_COLUMNS = ("from", "to", "plane", "read over", "how")

# What the readings window shows, and the difference from the two above is the
# second column. A fit answers over the stretch it was computed on, written into
# the line; a reading answers over the ground nothing nearer answers for, which
# is not in the line at all -- it is `DEFAULT_MAX_GAP` either side of a point,
# worked out here. Which is exactly the quantity nobody can see in a file, and
# the reason S26 governed 500 m at the far end of a trace it was snapped onto.
READING_COLUMNS = ("at", "reads", "answers over", "source", "the rest")

# Narrower than the fit window and shorter: one table, no sweep, no steering.
READING_WINDOW_PX = (560, 460)

# The axis the exposure window writes on, and the one value on it with a
# consequence past being read. FORMAT.md licenses a plane fitted to a facet on
# `exposure=exposed` and on nothing else, so this is the word that turns a
# calculation on -- which is why the table tints it and the step line names it.
EXPOSURE_AXIS = "exposure"
LICENSING = "exposed"
EXPOSED_TINT = "#d9ead3"

# What the exposure window shows about the spans already on that axis. `in force`
# is the column the others are read through: the axis allows any number of
# overlapping spans and the last one covering a metre wins, so a table without it
# would list four lines of which one is answering and not say which.
EXPOSURE_COLUMNS = ("from", "to", "says", "in force", "why")

# Taller than the readings window by the evidence box, which is four lines of
# text and the angles to whatever the stretch already carries.
EXPOSURE_WINDOW_PX = (620, 640)

# What the facet window shows about the measurements it could grow from. The
# licence is a column and not a filter: a reading on ground nobody has declared
# is the thing to go and declare, so hiding it would hide the next step.
SEED_COLUMNS = ("at", "reads", "licensed over", "grown")

# And what the radius sweep shows, which is the measurement this window is for.
# `rim` last because it is the one that invalidates the two before it: an area
# measured against the edge of the window is a lower bound wearing a number.
SWEEP_COLUMNS = ("radius", "area", "plane", "moved", "rim")

# How far from the site to look, offered rather than assumed. `facets.RADIUS` is
# 500 m, which is the reference implementation's number, and on the AOI it cuts
# 5 of the 8 facets off at the window: opened to 1000 m, S26 goes from 47 to 95
# hectares and S19's plane moves 11 degrees, from 121/30 to 110/30, towards the
# 107/35 its compass reads. So the radius is not a frame around the answer, it
# is part of it -- the same thing `FIT_LENGTHS` says about a window length, and
# handled the same way: a ladder to ask along, no default beyond the first.
FACET_RADII = (500.0, 1000.0, 2000.0, 3000.0)

# How far the plane may move across the ladder and still be one surface measured
# further out. Three degrees, which is not a gate on anything -- nothing is
# refused by it -- but the threshold of a sentence: below it the four radii are
# four measurements of one plane, and above it the radius is choosing the answer.
# Measured: S26, S22, S20 and S21 move 0 to 2 degrees, where S19 moves 5 and its
# wider answer is the one that agrees with its compass.
FACET_ONE_PLANE = 3.0

# Two tables, the diagnostics and the angles: taller than the fit window.
FACET_WINDOW_PX = (660, 820)

# The sweep is four rows and must not take height from the seeds above it, which
# is the table that is read first and scrolls.
SWEEP_TABLE_PX = 132

# How the grown region is drawn on the map: its own cells, thinned, as a stipple
# under everything. A stipple and not an outline, because the region is what it
# is -- a hull would draw the ground between two lobes as though it belonged --
# and because the cells go through `on_map` one by one, so the picture is right
# even where the map's projection is not the file's.
#
# Thinned to this many, which is about what a 95 ha facet has at a 20 m stride:
# past that the stipple is a solid wash and the trace under it stops reading.
FACET_CELLS_DRAWN = 2400
FACET_TINT = "#7b3294"
FACET_ALPHA = 0.33
FACET_CELL_PX = 2.6

# What the amend window shows about the attributes of the reading it is pointed
# at. Two columns, because an attribute is a key and a value and this is the one
# table in the tool whose cells are typed into rather than picked.
AMEND_COLUMNS = ("attribute", "value")

# The keys this window shows and refuses to let anybody type over, each for its
# own reason and none of them for tidiness:
#
# - `src` says who produced the line. A curator editing it forges a provenance,
#   which is the one thing in a file like this that cannot be checked against
#   anything -- and the whole precedence rests on being able to tell a compass
#   from a computation.
# - `off` is derived. It is the distance from the anchor to the trace, which
#   `Anchored.resolve` recomputes and nothing downstream reads, so a typed one is
#   a sentence that contradicts the coordinates beside it. This window writes it
#   from the anchor it is about to write, every time.
#
# `raw` and anything under `raw.` go with them, by prefix: FORMAT.md's first rule
# is that the source string is conserved, and a conserved string somebody has
# edited is not one. They are also what `owed_record` reads to decide whether the
# old plane has anywhere to be, so editing them would edit the test as well as
# the record.
AMEND_KEPT = ("src", "off")
AMEND_RAW = ("raw",)

# How many metres of a trace have to change hands before the window says so in
# metres rather than saying the move changes nothing. A metre, `FULLY_M`'s
# reason: the provenance is sampled, so a few tenths is the sampling and not a
# consequence.
AMEND_MOVED_M = 1.0

# Taller than the readings window by two groups and the consequence box, and
# narrower than the facet window: no second table of diagnostics.
AMEND_WINDOW_PX = (620, 860)

# The attribute table is short and must not take height from the readings above
# it, which is the table that is read first and scrolls. `SWEEP_TABLE_PX`'s rule.
AMEND_TABLE_PX = 150

# Below this, a stretch counts as claimed to the last metre rather than claimed
# in part. A metre: two orders of magnitude above the two decimals an anchor is
# written to, and below anything a fault is read at -- nobody keeps a fit for
# the metre an earlier line left over.
FULLY_M = 1.0

# What a window the map has to stay visible behind is offset by, where there is
# nothing remembered about it. Down and in from the map's own top-left, like a
# palette: a fit is read against the trace it was read off, so the one thing
# this must not do is come up centred over the fault being looked at -- which is
# where Qt puts a child window left to itself.
FIT_OFFSET_PX = (40, 40)

# The window lengths that can be asked for by hand, which is not the ladder the
# trace is swept along.
#
# **Longer than `traces.DEFAULT_SWEEP`, and deliberately not used as a default.**
# The sweep stops at 900 m because it was calibrated on `elementi_tettonici`,
# where the median trace is 172 m; the median trace of `montealpi_01.gstruct` is
# 1048 m, and read through that ladder 82 of its 393 traces come back with
# something to keep. What blocks the rest is not the gate -- 97% of the ground is
# `line` either way -- but `holding_length`, which refuses a peak sitting at an
# end of the ladder, rightly, since the turn is then outside what was asked. Ask
# about longer windows and 179 traces answer.
#
# And that is still not a reason to ask about them by default, which the checks
# caught: on a 2683 m trace with one bend in it, a long window covers the bend
# wherever it is put, so the held share peaks at the long end and the fit comes
# back claiming the whole trace. The fit that used to sit on the bend was the
# useful one. **A long window buys traces by giving up where along them the
# answer holds**, which is a trade a curator can make looking at a fault and a
# default cannot make for them.
#
# So: the sweep is left alone, and these are what `read over` offers. Which
# length a fit was read at is `window=` in the file either way.
FIT_LENGTHS = (150.0, 250.0, 400.0, 600.0, 900.0, 1500.0, 2500.0)


class Read(NamedTuple):
    """
    One trace read off the topography, before anybody has decided anything.

    Three lists in step -- what came out, the lines it would be written as, and
    the stretch each of those lines covers -- rather than a list of triples,
    because the caller that keeps them hands `curation` a list of lines and the
    caller that draws them walks the spans. `spans` holds `None` where a line's
    ends could not be read back, which is a bug and not a case: the lines are
    built by `as_line` out of the format's own spellings.
    """

    reading: object
    lines: list
    spans: list


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

        # Blocks as they were before a button wrote in them, newest last, as
        # `(which structure, the text)`.
        #
        # **The box's own Ctrl+Z cannot reach any of these**, and that is the
        # whole reason this exists rather than a preference for having an undo:
        # applying re-reads the block out of the document and puts it back with
        # `setPlainText`, which clears a QTextDocument's history -- so every
        # change made *for* somebody, by a Keep or a Delete, lands in a box whose
        # undo stack has just been emptied. What the box can undo is typing.
        # What this undoes is a press.
        #
        # Snapshots and not inverse operations: what goes back is the block that
        # was there, which cannot be wrong about anything, where an undo built
        # out of `put the line back at index 3` has to be right about a file that
        # has moved since. They are a block of text each -- twenty-six lines at
        # the most in these files -- so there is no reason to bound the list.
        self._before = []

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

        # Its floor back, the claims table having gone out of the column it was
        # dropped for: with nothing else in here to protect from being squeezed,
        # the only thing a zero floor can still do is let the outer splitter
        # shut the box to a line.
        self.text.setMinimumHeight(TEXT_FLOOR_PX)

        # Both, because a stretch changes under either gesture and neither
        # implies the other: typing an anchor moves the text without moving the
        # caret off the line, and clicking from one line to the next moves the
        # caret without touching a character.
        for moved in (self.text.cursorPositionChanged, self.text.textChanged):
            moved.connect(self._covering_changed)
            moved.connect(self._holding_changed)

        # A reading of the box as it stands, half-written lines included -- so it
        # is taken again on every change rather than on Apply. It costs a
        # `rows_of` over a block, which is the twenty-six lines of the largest
        # structure in the files here, and it is what the claims table cost too.
        self.text.textChanged.connect(self._show_backwards)

        self.problem = QtWidgets.QLabel()
        self.problem.setWordWrap(True)
        self.problem.setStyleSheet("color: #b2182b; font-size: 11px;")
        self.problem.setVisible(False)

        # The one thing the claims table carried that nothing else does, kept as
        # a sentence now that the table has gone. A pair of ends written the
        # wrong way round parses, applies, saves, and holds over no ground at
        # all -- `covers` is `from <= s <= to` -- and the box cannot show it:
        # seeing it there means holding two eastings in your head and knowing
        # which way the trace was digitised. That is the pair of wrong lines the
        # table was built for, out of one afternoon on one file.
        #
        # Both AOI curations hold none today, which is the state this is for
        # rather than an argument against it: they hold none because the table
        # found the ones they had. What is in neither file is a check that
        # anybody *would* notice the next one, which is why this is a sentence
        # the panel says and not a colour two cells carry.
        #
        # The fit window flags its own in red, and an `attitude` carries a point
        # rather than an interval and cannot be reversed at all. What is left is
        # `span` -- the claim with no window of its own, written by hand into
        # this box, and so the one this label is really for.
        #
        # Its own label and not `self.problem`, which belongs to the parser: a
        # refusal stops the block going in and this does not, so one line
        # between them would have each wiping the other's news at the moment
        # both are true.
        self.backwards = QtWidgets.QLabel()
        self.backwards.setWordWrap(True)
        self.backwards.setStyleSheet("color: #b2182b; font-size: 11px;")
        self.backwards.setVisible(False)

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

        # Two buttons left of the six that were here, and both of them are about
        # the box rather than about the file: Apply is how the box is committed
        # and Revert is how it is abandoned, so neither has anywhere else to go
        # while there is a box. The four that went were three templates and a
        # door, and what they had in common is that they were all about the open
        # trace -- which is now one menu, where the other five such doors already
        # were. A panel of buttons duplicating a menu is two places to keep a
        # refusal in step, and the fit button was one of them: it worked out
        # "there is no DEM" for itself, beside a menu entry working it out
        # again.
        buttons = QtWidgets.QHBoxLayout()
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

        # The box on its own, where it used to share a splitter with a table
        # reading it back. That table was the first step of taking the box away
        # and it stopped halfway: it could show that a line was wrong and never
        # let anybody write one, so what it ended up being was a second copy of
        # the block costing a `rows_of` per keystroke. What took the rest of the
        # step is the five windows -- they write the lines, so the box is the pen
        # for `span` and for repairs, and that wants height rather than a
        # neighbour. The one fact the table alone could see is now
        # `self.backwards`.
        working_layout.addLayout(gap)
        working_layout.addWidget(self.view, stretch=3)
        working_layout.addWidget(self.text, stretch=3)
        working_layout.addWidget(self.backwards)
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

        # Again here and not only on `textChanged`, because the structure can
        # change under text that does not: two traces with the same lines on
        # them read against different paths are different claims, and which end
        # is the far one is measured along the path.
        self._show_backwards()
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

    # -- taking one out, and putting a press back ---------------------------

    def remember(self):
        """The block as the document has it, kept for an undo. See `self._before`."""

        if self.index is None:
            return

        self._before.append((self.index, self.document.text_of(self.index)))

    def may_undo(self):
        """Whether there is a press to put back."""

        return bool(self._before)

    def undo_applied(self):
        """
        The block before the last press, put back. What changed, or None.

        **The whole block and not the gesture reversed**, which is the one
        decision in here. An undo built as `put that line back at index 3` has to
        be right about a file that has moved under it -- lines added above it,
        the same line written twice, a block re-serialised -- and when it is
        wrong it writes a claim nobody made. A snapshot cannot be wrong about
        anything: it is the text that was there, and it went through this
        parser once already.

        It goes back through `Document.replace` all the same, because the model
        beside the text is what the map and the tables read: restoring the lines
        and not the structure would leave a fit drawn on the map that the file no
        longer holds. And the structure it restores is selected, through
        `applied` -- an undo you cannot see is indistinguishable from one that
        did nothing.
        """

        if not self._before:
            return None

        index, text = self._before.pop()
        now = self.document.text_of(index)

        try:
            self.document.replace(index, text)
        except ValueError as err:
            # Not reachable through anything in this program: what is going back
            # came out of `text_of`, so the parser has already accepted it. Kept
            # on the stack rather than dropped, because a snapshot that cannot be
            # restored now is the one thing in here nobody can retype.
            self._before.append((index, text))
            self.problem.setText(f"the block will not go back: {err}")
            self.problem.setVisible(True)

            return None

        if index == self.index:
            self._redraw()

        self.table.update_row(
            index, self.document.dataset.structures[index], self.max_gap
        )
        self._apply_filter()
        self.applied.emit(index)

        return self._difference(now, text)

    @staticmethod
    def _difference(before, after):
        """What a block gained and lost, as a sentence a status bar can hold."""

        back = Counter(after.splitlines()) - Counter(before.splitlines())
        away = Counter(before.splitlines()) - Counter(after.splitlines())

        told = []

        for lines, how in ((back, "put back"), (away, "taken out")):
            if not lines:
                continue

            how_many = sum(lines.values())

            told.append(
                f"{how}: {next(iter(lines)).strip()}" if how_many == 1
                else f"{how_many} line(s) {how}"
            )

        return "; ".join(told) if told else "the block was already that"

    def _which_line(self, lines, at, line):
        """
        Which line of the box a row is pointing at, as `(index, why not)`.

        `at` is where the line sat when the row showing it was read, and the
        text is what decides: the box can have been typed in since, and an index
        into a block that has moved is an index at somebody else's line. So the
        index is a hint, checked against the text before it is used, with a
        search by content behind it -- and where the content is ambiguous,
        because this file has two fits written byte for byte alike, the index
        that no longer matches is not guessed at.

        Shared by the two gestures that rewrite one line of a block from a table,
        and shared deliberately: a removal and a detachment that disagreed about
        which line a row means would be two ways of hitting the wrong one.
        """

        wanted = line.strip()

        if 0 <= at < len(lines) and lines[at].strip() == wanted:
            return at, None

        alike = [n for n, one in enumerate(lines) if one.strip() == wanted]

        if len(alike) == 1:
            return alike[0], None

        if not alike:
            return None, (
                "that line is not in the box any more -- Apply what is "
                "there, or Revert, and the row will be read again"
            )

        return None, (
            f"{len(alike)} lines in the box read exactly alike and the "
            f"row no longer says which of them it is: Apply or Revert "
            f"first"
        )

    def _rewrite_line(self, at, line, into, refusal):
        """One line of the block replaced by `into`, or dropped where it is None."""

        if self.index is None:
            return refusal

        was = self.text.toPlainText()
        lines = was.splitlines()
        which, why = self._which_line(lines, at, line)

        if why is not None:
            return why

        self.remember()

        if into is None:
            del lines[which]
        else:
            lines[which : which + 1] = into.splitlines()

        self.text.setPlainText("\n".join(lines))

        if not self.apply_block():
            # The parser's own words are on `problem`; this puts the text and the
            # stack back, so a refused edit costs nothing at all.
            self._before.pop()
            self.text.setPlainText(was)

            return "the block will not parse like that: see the message"

        return None

    def drop_line(self, at, line):
        """
        One line out of the block, applied. Why it would not go, or None.

        **Removing a `fit` is allowed where emptying a block is not**, and
        `Document.replace` is where the other half of that is written. A
        structure that does not hold is *said* not to hold, so that tomorrow a
        rejected fault can be told from one nobody ever mapped; a `fit` carries
        no such distinction to lose. It is a plane some producer computed, with
        the producer and its window written on it, and a file with it removed
        says what the file said before it was computed: nothing is claimed here.
        What the line said is in the status bar and on the undo stack.

        A **reading** does carry that distinction, which is why it goes out
        through `comment_out` and not through here.
        """

        return self._rewrite_line(
            at, line, None, "nothing is open to take a line out of"
        )

    def comment_out(self, at, line, note):
        """
        One line turned into the comment that records it going. Why not, or None.

        `drop_line` for a reading, and the difference is the whole argument of
        `detachment_note`: a fit removed leaves a file that says what it said
        before the computation, and a reading removed leaves a file that cannot
        tell a measurement somebody decided against from ground nobody walked.
        So the line does not vanish, it becomes a `#` holding itself.

        Comments survive this and survive the Save behind it, both for the same
        reason -- the box holds the block as text, and `Document` replaces the
        lines of one structure rather than dumping the file. `dumps` would eat
        them, and nothing here calls it.
        """

        return self._rewrite_line(
            at, line, note, "nothing is open to take a reading out of"
        )

    def amend_claim(self, at, line, into):
        """
        One line replaced by what it now claims, applied. Why not, or None.

        The third door onto `_rewrite_line` and a third name for it, because
        the three are three different acts and a shared name would be the only
        thing saying they are one. `drop_line` leaves a file that says what it
        said before a computation; `comment_out` leaves a file that can still
        tell a measurement somebody decided against from ground nobody walked;
        this one leaves a file making a *different* claim in the same place, and
        whether anything records the claim it replaced is `owed_record`'s
        question, settled in the window before the press and arriving here
        already spliced into `into`.

        So `into` may be one line or three: a comment keeping the old line
        verbatim, the reason it is being kept, and the amended claim under them.
        `_rewrite_line` splits on newlines, which is what makes that one call
        rather than two -- and one call is what makes it one entry on the undo
        stack. Two presses to undo an amendment would be an amendment that can
        be half taken back, leaving a comment over a line it no longer describes.
        """

        return self._rewrite_line(
            at, line, into, "nothing is open to amend a reading in"
        )

    def insert_claim(self, line):
        """
        One finished line into the block, applied. Why it would not go, or None.

        `add_line`'s sibling and not a second use of it, the difference being
        whether anything is left to pick. A template goes in with `*` where its
        anchors will be, waits in the box, and arms the caret for the clicks that
        fill it; this goes in complete and applies in the press, which is the rule
        every write from a window follows -- Apply stands for having looked, and a
        window that showed the numbers before the press has been that.

        Above the path for `add_line`'s reason: `dumps` puts the geometry last, so
        a line written after it would sit among the vertices and read as one.
        """

        if self.index is None:
            return "nothing is open to write a line into"

        was = self.text.toPlainText()
        lines = was.splitlines()

        self.remember()
        lines.insert(self._path_line(lines), line)
        self.text.setPlainText("\n".join(lines))

        if not self.apply_block():
            self._before.pop()
            self.text.setPlainText(was)

            return "the block will not parse with that line: see the message"

        return None

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

    def gate_said(self):
        """
        The floor these traces are read through, in words, every time it is asked.

        Separate from the one-shot sentence `fit_off_dem` says, and the two are
        not the same kind of thing. A sentence in the status bar answers a press
        and is said once because saying it on every press is noise; a window that
        shows the verdicts has to be able to show what produced them for as long
        as they are on screen. `too straight` is not readable without it -- it is
        a comparison, and this is the number it is against.
        """

        gate, sigma = self._gate()

        return f"lever {gate.min_lever:.1f} m " + (
            f"(3x the {sigma:.1f} m measured off these traces)"
            if sigma is not None
            else "(the default: these traces do not measure a pen)"
        )

    def _sweep(self, length=None):
        """
        How to sweep, given a length somebody picked or none at all.

        One length means exactly that length: `holding_length` wants three to
        find a peak between, so a ladder of one resolves nothing and drops
        straight to the fallback, which is set to the same number. `chosen` is
        what makes the reading say so instead of calling it a fallback.

        None is `fits.Sweep`'s own defaults and not a ladder of this tool's:
        `FIT_LENGTHS` says why the longer windows are offered and not assumed.
        """

        if length is None:
            return Sweep()

        return Sweep(lengths=(float(length),), fallback=float(length), chosen=True)

    def read_off_dem(self, length=None):
        """
        The topography read along the selected trace, and nothing written anywhere.

        `fit_off_dem` without the writing, which is the half a window that shows
        before it keeps needs: the reading, the lines it would write, and the
        stretch each of those lines claims.

        **The stretches are measured by reading the lines back**, rather than
        carried out of the computation that made them. `fits_along` anchors a fit
        at two coordinates -- which is the format's rule, so that a stretch
        survives the trace being redigitised -- and the progressives it had on the
        way there are not what a reader will recover. Reading them back through
        `interval_of` makes the preview a reading of the file that is about to
        exist, so what is shown and what will be meant cannot come apart.

        `length` is metres of window, or None for the ladder. Passing one is not
        a way of getting more rows out of a trace that would not give any: it is
        the assertion that this fault holds one orientation over that distance,
        which is a geological judgement and one the file records as `window=`.
        """

        if self.index is None or self.dem is None or self.dem_said:
            return None

        structure = self.document.dataset.structures[self.index]

        reading = fits_along(
            structure,
            self.dem,
            self._gate()[0],
            sweep=self._sweep(length),
            convergence=self.convergence,
        )

        lines = [as_line(fit) for fit in reading.fits]

        return Read(
            reading=reading,
            lines=lines,
            spans=[interval_of(line, structure.path) for line in lines],
        )

    def lengths_that_hold(self):
        """
        How many stretches each window length would give on this trace: `[(m, n)]`.

        Only the lengths that give something, in order, and never a length that
        gives none -- a list of zeros is the same statement as an empty list and
        four times as long to read.

        **Asked only when a reading came back with nothing**, which is where the
        question arises: `nothing held` is a verdict about one window length, and
        a curator reading it has no way of telling whether the trace is straight
        or whether it turns over a distance nobody asked about. That is the
        whole of what this answers, and it is the answer the ladder already had
        and threw away -- `window_sweep` computes every length and `fits_along`
        returns the spans of one.

        Recomputed rather than plumbed out of the sweep because it is cheap and
        the plumbing is not: the worst trace in the AOI is 29 km with nine
        vertices and the whole ladder on it costs 60 ms, while carrying seven
        `TraceSpans` back through `Reading` would put six unused sweeps in the
        way of every reading that did work.
        """

        if self.index is None or self.dem is None or self.dem_said:
            return []

        structure = self.document.dataset.structures[self.index]
        gate = self._gate()[0]

        out = []

        for length in FIT_LENGTHS:
            reading = fits_along(
                structure, self.dem, gate, sweep=self._sweep(length)
            )

            if reading.fits:
                out.append((length, len(reading.fits)))

        return out

    def keep_fits(self, lines):
        """
        Lines somebody has already looked at, into the block and applied at once.

        **Applied, where `fit_off_dem` leaves them in the box**, and the
        difference is not impatience. Apply is the step where a thing that has
        been read gets kept, and what made it necessary there is that the lines
        arrive without anybody having seen them. A window that shows every fit
        with the stretch it covers and a tick beside it has already been that
        step, and a second one after it would be asking the same question twice
        -- which teaches the answer rather than the question.
        """

        if self.index is None or not lines:
            return False

        self.remember()
        self.add_written(lines)

        if self.apply_block():
            return True

        self._before.pop()

        return False

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

        Both halves of that are being reconsidered and neither was wrong. What
        changed is not the cost of the computation but who the lines are for:
        `FitFromDem` shows them with the ground each one covers and keeps the
        ticked ones, which is the looking that Apply was standing in for. This
        stays as the path with no window in it -- the checks read it, and it is
        the shortest way to ask the topography a question.
        """

        if self.index is None or self.dem is None or self.dem_said:
            return None

        structure = self.document.dataset.structures[self.index]

        reading = fits_along(
            structure,
            self.dem,
            self._gate()[0],
            sweep=self._sweep(),
            convergence=self.convergence,
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
            told.append(f"gate: {self.gate_said()}")

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

    def forget_covering(self):
        """
        Makes the next caret move report its stretch even where it did not change.

        The guard in `_covering_changed` remembers what was last *emitted*, so
        that a caret crossing a forty-character line does not cost forty blits of
        a picture that did not move. It is a sound guard exactly as long as this
        panel is the only thing drawing that band, and it is not any more: the
        fit window points at a row and the band becomes the row's stretch,
        without anything here hearing about it.

        Left alone, a caret then landing on a line that claims what this panel
        last emitted reports no change, and the map goes on showing the row --
        so the band would be saying `fit 1313..1363` while the caret sat in a
        line claiming the whole trace. Found by a check that was measuring
        something else and got 5 points where it expected 135.
        """

        self._covering = (None, None)

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

    def _show_backwards(self):
        """
        Names the block's lines that hold over no ground, if it has any.

        `claim_said` makes the same finding one line at a time, and the
        difference is the whole reason this exists: that one answers about the
        line the caret is on, so a reversed pair three lines further down is a
        sentence nobody is ever shown. This asks the question of the block.

        Line numbers and not the text, because what the reader does next is look:
        the box is right above this label and the numbers are how a line is found
        in it. `Row.at` counts blocks from zero and a reader counts from one.
        """

        if self.index is None:
            self.backwards.setVisible(False)

            return

        path = self.document.dataset.structures[self.index].path
        held = [
            row.at + 1
            for row in rows_of(self.text.toPlainText(), path)
            if row.ends is not None and row.ends[0] > row.ends[1]
        ]

        if not held:
            self.backwards.setVisible(False)

            return

        which = ", ".join(str(line) for line in held)

        self.backwards.setText(
            f"line {which} has its two ends the wrong way round, so it covers "
            f"no part of the trace: the format reads the pair as written, and "
            f"`covers` is `from <= s <= to`."
            if len(held) == 1 else
            f"lines {which} have their ends the wrong way round, so they cover "
            f"no part of the trace: the format reads each pair as written, and "
            f"`covers` is `from <= s <= to`."
        )
        self.backwards.setVisible(True)

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

    def line_now(self):
        """
        The text of the line the caret is on.

        Public because the window asks it: what to tell somebody to do next
        depends on how far the caret's line has got, and the caret is this
        panel's. Read and not interpreted here, the reading being `curation`'s
        job and the window importing the same functions.
        """

        return self.text.textCursor().block().text()

    def _holding_changed(self):
        """Reports the caret's plane when it becomes a different one."""

        reported = (self.index, plane_of(self.line_now()))

        if reported == self._holding:
            return

        self._holding = reported
        self.holding.emit(reported[1])

    def in_document(self, line):
        """
        Whether the open block holds this line as it stands.

        What `dirty` cannot answer: that is a fact about the file having unsaved
        changes, and this is a fact about one line having been through the
        parser. A line finished in the box and kept by nothing sits in a document
        that is perfectly clean.
        """

        if self.index is None:
            return False

        wanted = line.strip()

        return any(
            one.strip() == wanted
            for one in self.document.text_of(self.index).splitlines()
        )

    def has_plane_slot(self):
        """
        Whether the caret's line is a `fit` with somewhere to write a plane.

        **A `fit` and nothing else**, which `with_plane` does not say on its own:
        `PLANE_AT` carries `attitude` as well, and has to, the importers writing
        measurements through it. What the steering is, though, is a producer of
        fits -- it computes a plane from a hand, a dial and a DEM and stamps
        `from=plane-dem` on it -- and an `attitude` is somebody's compass reading.
        There is no state of that window in which replacing a reading with a
        computed number is the thing being asked.

        Not a hypothetical. The caret parks on the last line before the path, and
        on `Mt. Alpi faults.2` that is the reading at station S26: the step line
        said `turn the dial, then press Keep this plane`, one press put the
        steered plane into it -- *and applied in the same press*, the anchor being
        written -- and the file was left holding `plane 237.0/60.0 ... src=points
        raw="dip_dir=140 dip=35" from=plane-dem`. A measurement overwritten by a
        computation, carrying the provenance of both, with no `fit` created
        anywhere. It survived only because the import had kept `raw=`.
        """

        line = self.line_now()
        tokens = line.split()

        if not tokens or tokens[0] != "fit":
            return False

        return with_plane(line, 0.0, 0.0) is not None

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

        line = self.line_now()
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

        # Asked here and not only where the button is enabled: enablement is
        # recomputed when the caret's *plane* changes, and moving from a reading
        # to a `fit` that happens to carry the same numbers changes no plane. The
        # gate has to be on the write.
        if not self.has_plane_slot():
            return None

        written = with_plane(self.line_now(), dip_dir, dip)

        if written is None:
            return None

        if attrs:
            written = with_attrs(written, attrs)

        edit = self.text.textCursor()
        edit.movePosition(QtGui.QTextCursor.MoveOperation.StartOfBlock)
        at = edit.position()
        edit.movePosition(
            QtGui.QTextCursor.MoveOperation.EndOfBlock,
            QtGui.QTextCursor.MoveMode.KeepAnchor,
        )
        edit.insertText(written)

        # **And the ends are left pickable**, which takes saying so. Replacing
        # the whole line drops the selection that was on a `*`, and a shift-click
        # with nothing aimed writes its coordinate at the caret -- which here is
        # the end of the line, where `loads` reads a token it has no slot for and
        # drops it. Measured on L0071: a plane written first and two ends clicked
        # after gave `fit plane * * 220.0/35.0 from=plane-dem @...,... @...,...`,
        # which applies, claims all 7239 m of the trace, and keeps neither click.
        #
        # On this line only. A plane written into a line that is already finished
        # must not reach down into the next one looking for a `*` to aim at.
        self._aim_at_anchor(at, same_line=True)

        return written

    def keep_plane(self, dip_dir, dip, attrs=None):
        """
        A steered plane into the caret's line, applied if the line is finished.

        Returns `(the line, whether it is in the document)`.

        `keep_fits`' sibling, and the condition is the one difference between
        them. The swept fits arrive with their ends computed, so there is nothing
        left for anybody to decide and applying them is right. A steered plane
        lands on a line whose ends may still be `*`, which `interval_of` reads as
        the whole trace -- correctly -- and applying *that* would assert a claim
        over ground nobody picked, in one press, with the band on the map looking
        exactly as it did.

        So the apply waits for two anchors. Not a refusal and not a dialog: the
        plane is in the box either way, the ends can be clicked after it now,
        and pressing again keeps it. Which also makes the button idempotent,
        because what it writes is a function of the dial and not of what is
        already there.
        """

        written = self.take_plane(dip_dir, dip, attrs)

        if written is None:
            return None, False

        if not anchors_written(written):
            return written, False

        self.remember()

        if self.apply_block():
            return written, True

        self._before.pop()

        return written, False

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

        Returns whether closing the pair put it the wrong way round and it was
        turned -- see `with_ends_in_order`, and `pick`, which says so. Or None,
        where the click had nowhere to go and nothing was written.

        **One step of the undo stack for one click**, which the turn is the
        reason for: two edits to the document would take two Ctrl-Zs to undo,
        and the first of them would leave the anchor sitting there with the
        ends back to front, which is a state no gesture produced. It is
        `_order_ends` that joins them and not a `beginEditBlock` around both,
        and that is not a preference -- see the note there.
        """

        nowhere = self._nowhere_for_an_anchor()

        if nowhere is not None:
            self.said.emit(nowhere)

            return None

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

        turned = self._order_ends()

        self.text.setFocus()

        return turned

    def _nowhere_for_an_anchor(self):
        """
        Why a picked anchor would not land, or None if it will.

        One case, and it was silent until it was looked for: a line whose anchor
        slots are all written, with no `*` selected. The coordinate goes in at
        the caret, `loads` drops it, and the line goes on claiming exactly what it
        claimed -- so the click is gone with no error anywhere and the only
        evidence is a band that did not move.

        Refused rather than guessed at. Which of two written ends a third click
        meant to replace is not something this can know, and overwriting the
        nearer one would be a rule nobody asked for applied silently to somebody's
        own typing. Said with the way out in it, because selecting an end and
        shift-clicking *does* replace it -- the selection is what this reads.
        """

        cursor = self.text.textCursor()

        if cursor.hasSelection() and cursor.selectedText() == "*":
            return None

        if not anchors_written(self.line_now()):
            return None

        # Said with the new line in it and not only with the way to edit this one,
        # because of how it was read: *inizio/fine erano già definiti e quindi non
        # si procedeva*. The condition was true and the sentence was about the
        # wrong thing -- the ends it names belong to a line the caret happened to
        # be parked on, and the stretch being chosen had not been written down
        # anywhere yet. `+ fit` is the step that was missing.
        return (
            "this line's ends are already written: press `+ fit` for a new line "
            "to click them on, or select one of them here to replace it"
        )

    def _order_ends(self):
        """
        Turns the caret's line round if its two ends were picked back to front.

        Nothing to do until the pair is closed: the line is looked at after the
        anchor has gone in, and `with_ends_in_order` answers None for everything
        that is not two anchors the wrong way round -- a half-filled template
        among them, since `*` and an anchor cannot be in the wrong order.

        The whole block is replaced, which is `take_plane`'s dance and leaves
        the caret on the same line, so what `claim_said` and `pinned_at` read
        next is this line as it now stands.

        **`joinPreviousEditBlock` and not a `beginEditBlock` around the pair**,
        which is what one click, one Ctrl-Z was written as first and what
        crashed: the outer block was opened on `insert_anchor`'s own cursor,
        and then this replaced the whole line out from under it with a second.
        Qt segfaulted on the `endEditBlock` that followed -- reproducibly, at
        the same assertion, and not at teardown where this project's other
        exit-139 lives. Joining from here is also the smaller claim: the edit
        that has to be grouped is *this* one, onto whatever the click just did,
        and nothing has to be held open across a caret move and a signal.
        """

        if self.index is None:
            return False

        path = self.document.dataset.structures[self.index].path
        written = with_ends_in_order(self.line_now(), path)

        if written is None:
            return False

        edit = self.text.textCursor()
        edit.joinPreviousEditBlock()

        try:
            edit.movePosition(QtGui.QTextCursor.MoveOperation.StartOfBlock)
            edit.movePosition(
                QtGui.QTextCursor.MoveOperation.EndOfBlock,
                QtGui.QTextCursor.MoveMode.KeepAnchor,
            )
            edit.insertText(written)
        finally:
            edit.endEditBlock()

        return True


class PlaneSteering(QtWidgets.QWidget):
    """
    A dial, two bars, two numbers: the plane laid on the DEM, by hand.

    `tools/intersection.py`'s panel, cut down to what is left once the tool has
    something to aim at. Gone are the source point's three boxes and the
    compute-window spinner, and neither is a simplification: there the point is
    typed in or clicked anywhere, because the tool has no idea what you are
    looking at, and the window is a cost dial because the plane is unbounded.
    Here the point is on a trace -- the middle of the stretch the caret's line
    claims, or wherever ctrl-click puts it -- and the window is sized from the
    stretch or from what is on screen, so both were answers to questions this
    tool can work out for itself.

    What is left is the controls that *are* the tool -- turn it, watch the
    curves move, stop when they run along the fault -- plus the one button that
    was missing from the other tool entirely: the number going into the file.

    **Each number has a dial or a bar and a box, and the bearing has both.** Not
    a redundancy: a round control and a straight one are good at different
    gestures. Swinging a plane through a quadrant to see where the cut goes is a
    turn, and the dial is the only one of the two that does it without hitting
    an end; nudging a bearing by a degree, or jumping from 90 to 270, is a
    distance, and the bar is the only one of the two that shows where you are in
    the range while you do it.

    **It reports into its own label and not into the status bar.** The bar is
    one line and shared with everything the panel says, and this fires on every
    step of a dial: a per-frame report there would be the mistake the claimed
    stretch already made once, scaled up by a factor of ninety.
    """

    steered = QtCore.pyqtSignal(float, float)
    take_asked = QtCore.pyqtSignal()
    armed_changed = QtCore.pyqtSignal(bool)
    unpin_asked = QtCore.pyqtSignal()
    start_asked = QtCore.pyqtSignal()

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

        # Fixed and not a floor, now that it sits beside something that wants
        # the width: left to stretch it would be drawn as an ellipse, and a
        # bearing read off an ellipse is read off the wrong angle.
        self.dial.setFixedSize(132, 132)

        # The dial alone steps by a whole degree and the convergence here is
        # 0.8, so without the tenth the correction would be finer than the
        # control meant to apply it. The box is what commands; the dial follows.
        self.dip_dir = QtWidgets.QDoubleSpinBox()
        self.dip_dir.setRange(0.0, 359.9)
        self.dip_dir.setDecimals(1)
        self.dip_dir.setSingleStep(0.1)
        self.dip_dir.setWrapping(True)
        self.dip_dir.setSuffix("°  dip dir")

        # A bar for the bearing too, beside the dial and not instead of it.
        #
        # The dial is the better control for the gesture this tool exists for --
        # swing the plane and watch the cut move, which is a turn and not a
        # distance -- but it is the worse one for the gesture around it: nudging
        # a bearing by a degree, or going to 270 from 90 without passing
        # through everything between. A bar does both, and the two cost one
        # widget to keep in step.
        #
        # **It cannot wrap, and that is the dial's half of the division.** The
        # range runs 0 to 360 with north at both ends, so the whole circle is
        # reachable from either side; dragged off the right-hand end it stops
        # there rather than coming round, and the bearing it reports is taken
        # modulo 360. Which is why the handle is not pulled back to the left
        # while a hand is on it: the next thing to set the plane puts it there.
        self.dip_dir_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.dip_dir_slider.setRange(0, 360)
        self.dip_dir_slider.setTickInterval(45)
        self.dip_dir_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.dip_dir_slider.setMinimumHeight(26)
        self.dip_dir_slider.setPageStep(45)

        self.dip_dir_scale = self._scale(self.dip_dir_slider)

        # Beside 132 px of dial, a slider at its natural height is a 20 px bar
        # between two spin boxes, which is the shape of a progress bar: it reads
        # as something being reported rather than something to take hold of. The
        # dial says what it is by being round and says its scale with notches
        # all the way round; this said neither. So it is given the height its
        # ticks need in order to draw at all -- at the natural 20 px there is no
        # room under the groove for them and they come out not drawn, which is
        # why setting `TicksBelow` here had been doing nothing -- and a page step
        # of a tick, so that clicking the groove lands on one.
        self.dip_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.dip_slider.setRange(0, 90)
        self.dip_slider.setTickInterval(15)
        self.dip_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.dip_slider.setMinimumHeight(26)
        self.dip_slider.setPageStep(15)

        self.dip_scale = self._scale(self.dip_slider)

        self.dip = QtWidgets.QDoubleSpinBox()
        self.dip.setRange(0.0, 90.0)
        self.dip.setDecimals(1)
        self.dip.setSingleStep(0.1)
        self.dip.setSuffix("°  dip")

        self.dip_dir.setValue(90.0)
        self.dip.setValue(30.0)
        self._sync_dial()
        self._sync_bars()

        self.label = QtWidgets.QLabel()
        self.label.setWordWrap(True)
        self.label.setStyleSheet("color: #6a6a6a; font-size: 10px;")
        self.label.setMinimumHeight(48)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        # What to do next, worked out from the state and not from the last thing
        # pressed. Its own label, above the per-frame report and below the
        # buttons it is about.
        #
        # **A line of its own and not a sentence appended to that report**,
        # because the two are different kinds of writing and would spoil each
        # other: the report changes ninety times a turn of the dial, so an
        # instruction inside it is an instruction moving about under the eye, and
        # a report with guidance stuck to it is read as one thing and skipped as
        # one thing. This also lets it be said with the steering switched *off*,
        # where there is no frame and no report -- which is where the first step
        # is, and the step a hand gets stuck on.
        #
        # Darker than the report, and that is the whole of the styling argument:
        # grey at ten pixels is what this window says about itself, and the one
        # line here that is addressed to somebody has to not look like it.
        self.step = QtWidgets.QLabel()
        self.step.setWordWrap(True)
        self.step.setStyleSheet("color: #30506a; font-size: 11px;")
        self.step.setMinimumHeight(30)
        self.step.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        # `Keep this plane`, and the word is borrowed on purpose from the button
        # six inches below it: `Keep the ticked ones`. Two producers in one window
        # and one verb between them, because what the two gestures do is the same
        # thing -- a plane that was being looked at becomes a plane the file
        # claims. It used to say `write it in the line`, which is exactly what the
        # code does and not what anybody is trying to do, and the line it names is
        # in another window.
        self.take = QtWidgets.QPushButton("Keep this plane")
        self.take.clicked.connect(lambda: self.take_asked.emit())

        # **The line to keep it in, startable from here.** The same button is in
        # the box, three windows away, and the step line kept pointing at it
        # there: *press `+ fit` in the box*. That is not advice anybody follows
        # with one hand on a dial and the other on the map, so the state it was
        # the way out of -- the caret parked on whatever line was last before the
        # path -- was the state every attempt was made from.
        #
        # First in the row, left of `Keep`, which is the order the two go in: this
        # one makes the line, that one fills it. Never disabled, because writing a
        # template is not a judgement about anything -- it is what a hand does
        # *instead* of pressing a button that is grey.
        self.start = QtWidgets.QPushButton("+ fit")
        self.start.setToolTip(
            "Write a new `fit` line above the path and put the caret on it, with "
            "its first end armed. Its two ends come up empty: shift-click the map "
            "for each, which is how the stretch this plane is about gets said."
        )
        self.start.clicked.connect(lambda: self.start_asked.emit())

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
        self.dip_dir_slider.valueChanged.connect(self._dip_dir_slid)
        self.dip_slider.valueChanged.connect(self._dip_slid)
        self.dip.valueChanged.connect(self._dip_typed)
        self.on.toggled.connect(self._toggled)

        # The dial beside the bars rather than above them, which is what the
        # move out of the map's column bought. Stacked in 190 px the two bars
        # would be 168 px long -- two degrees of bearing per pixel -- and the
        # block would be 400 px tall; across a window they get the width, and
        # the dial is level with the pair it belongs to.
        bars = QtWidgets.QVBoxLayout()
        bars.setSpacing(2)
        bars.addWidget(self.dip_dir_slider)
        bars.addWidget(self.dip_dir_scale)
        bars.addWidget(self.dip_dir)

        # A gap between the two, because this is four controls making two pairs
        # -- each a thing to take hold of with its number under it -- and at an
        # even spacing a bar sits as near the box above it as the one it belongs
        # to. Which is not a hypothetical: with one bar here and the dial above
        # it, the dip's slider was read as the dip direction's by the person who
        # asked for this one.
        bars.addSpacing(10)

        bars.addWidget(self.dip_slider)
        bars.addWidget(self.dip_scale)
        bars.addWidget(self.dip)

        turning = QtWidgets.QHBoxLayout()
        turning.setSpacing(10)
        turning.addWidget(self.dial, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        turning.addLayout(bars, 1)

        pressing = QtWidgets.QHBoxLayout()
        pressing.addWidget(self.start)
        pressing.addWidget(self.take)
        pressing.addWidget(self.release)
        pressing.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.on)
        layout.addLayout(turning)
        layout.addLayout(pressing)
        layout.addWidget(self.step)
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

    @staticmethod
    def _scale(slider):
        """
        The ends of a slider's range written under it, and the middle.

        The other half of what the notches do: a dial with no numbers on it is
        still obviously a bearing, and a bar with no numbers could be running to
        90 or to 360.

        **Read off the slider and not argued**, so that a range changed in one
        place does not leave a label behind saying what it used to be.

        Three of them and not five, because three is the most that can be placed
        exactly this way: equal cells with the outer two aligned outwards put
        marks at 0, 1/2 and 1 of the width, which is where those values are. A
        fourth would have to sit at 1/4, and equal cells would put it at 3/10 --
        twenty pixels out on a bar this wide, which is a scale that lies. The
        ends themselves are off by half a handle, the groove being inset by that
        much, and that is the price of not reimplementing
        `QStyle.sliderPositionFromValue`: affordable because of what they are
        for -- which end is which, and where the range stops. Neither is a
        reading taken off the bar; the number is in the box below, to a tenth.
        """

        low, high = slider.minimum(), slider.maximum()
        scale = QtWidgets.QWidget()

        marks = QtWidgets.QHBoxLayout(scale)
        marks.setContentsMargins(2, 0, 2, 0)
        marks.setSpacing(0)

        for degrees, side in (
            (low, QtCore.Qt.AlignmentFlag.AlignLeft),
            ((low + high) // 2, QtCore.Qt.AlignmentFlag.AlignHCenter),
            (high, QtCore.Qt.AlignmentFlag.AlignRight),
        ):
            mark = QtWidgets.QLabel(f"{degrees}°")
            mark.setStyleSheet("color: #6a6a6a; font-size: 10px;")
            mark.setAlignment(side | QtCore.Qt.AlignmentFlag.AlignTop)
            marks.addWidget(mark, 1)

        return scale

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
        self._sync_bars()

    def note(self, said):
        self.label.setText(said or "")

    def tell_next(self, said):
        """
        The one gesture that would take a steered plane nearer the file.

        Written as the imperative it is -- `next: ...` and then a thing to do --
        and set from the state every time the state could have moved, so that it
        is never the trace of a press that has already happened. `QLabel` drops
        a `setText` with the text it already has, which is what makes calling
        this per frame free.
        """

        self.step.setText(f"next: {said}" if said else "")

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
            why or "Put this attitude on the line the caret is on, with where it "
                   "came from, and into the document in the same press once both "
                   "ends have been clicked. Nothing reaches the file until Save."
        )

    # -- five controls saying two numbers, each following the others -------
    #
    # The boxes are what command, and everything else follows them: a gesture on
    # a dial or a bar writes its box with the box's signal blocked and then puts
    # the *other* representations of that box back in step, never the one under
    # the hand. A control re-set while it is being dragged fights the pointer,
    # and a bearing is the case where it would also lie -- 359.6 in the box
    # rounds to 360, which the bar would then snap to its other end.

    def _sync_dial(self):
        with QtCore.QSignalBlocker(self.dial):
            self.dial.setValue(
                int(round(self.dip_dir.value() - self.DIAL_NORTH_OFFSET)) % 360
            )

    def _sync_dip_dir_bar(self):
        with QtCore.QSignalBlocker(self.dip_dir_slider):
            self.dip_dir_slider.setValue(int(round(self.dip_dir.value())) % 360)

    def _sync_dip_bar(self):
        with QtCore.QSignalBlocker(self.dip_slider):
            self.dip_slider.setValue(int(round(self.dip.value())))

    def _sync_bars(self):
        self._sync_dip_dir_bar()
        self._sync_dip_bar()

    def _dial_moved(self, value):
        with QtCore.QSignalBlocker(self.dip_dir):
            self.dip_dir.setValue(float((value + self.DIAL_NORTH_OFFSET) % 360))

        self._sync_dip_dir_bar()
        self._answer()

    def _dip_dir_slid(self, value):
        with QtCore.QSignalBlocker(self.dip_dir):
            self.dip_dir.setValue(float(value % 360))

        self._sync_dial()
        self._answer()

    def _dip_dir_typed(self, value):
        self._sync_dial()
        self._sync_dip_dir_bar()
        self._answer()

    def _dip_slid(self, value):
        with QtCore.QSignalBlocker(self.dip):
            self.dip.setValue(float(value))

        self._answer()

    def _dip_typed(self, value):
        self._sync_dip_bar()
        self._answer()

    def _steering(self, on):
        for widget in (
            self.dial,
            self.dip_dir_slider,
            self.dip_dir_scale,
            self.dip_dir,
            self.dip_slider,
            self.dip_scale,
            self.dip,
            self.take,
        ):
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


class ReadingsHere(QtWidgets.QWidget):
    """
    The measurements claimed along one trace, and the one gesture that is new.

    `FitFromDem`'s sibling, and separate from it for the reason `fits_in` gives:
    a fit is what a computation returned and a reading is what a compass was
    pointed at, so one table over both would be one gesture over two kinds of
    claim with different grounds behind them. The division is also what makes
    this window's Detach safe to offer at all -- it is never a press away from a
    fit, which goes out through its own button with its own argument.

    **What it exists for is a case the file could not state.** S26 reads 140/35
    on `Mt. Alpi faults.2`, a `?transcurrent`: a 35-degree plane on a fault that
    cannot have one. Its three nearest neighbours read 107/35, 115/30 and
    120/30, the one of those that sits on a structure sits on a `?thrusts`, and
    that structure's own comment names a sovrascorrimento. The compass was
    right. What was wrong was `off=7.4` -- an importer snapping a point onto the
    nearest trace -- and nothing in this tool could undo an importer's guess.

    **The second column is the point of the table.** A fit carries the stretch it
    was computed over, written in the line; a reading carries a point, and the
    ground it answers for is `DEFAULT_MAX_GAP` either side of it, which is in no
    file anywhere. That is the number that made S26 a problem worth finding: it
    governs the last 150 m of its trace, and -- through the `misurata-lontana`
    tier, where nothing nearer answers -- the first 500 m as well, from 3.4 km
    away. Shown as metres along the trace, and lit on the map by the same band
    the fit window uses, because the band means one thing and this is a stretch
    under discussion like any other.

    **Detach, not Delete.** The word is the claim: what the press removes is the
    line saying this measurement belongs to this structure, and `detachment_note`
    is where the rest of that argument lives. The reading does not evaporate --
    it becomes the comment that says it was here and why it went.

    **Why is typed before the press, not after.** A reason box that could be left
    empty is a reason box that is left empty, and the comment is the whole
    justification for removing the line rather than deleting it. So the button is
    dead until there is a sentence, and the sentence goes in the file verbatim.
    """

    # The stretch to light on the map, as `(s0, s1)` or None -- the same signal
    # and the same sink as the fit window's, `_show_fitting` being the one place
    # that decides what the band means.
    showing = QtCore.pyqtSignal(object)

    # That the next shift-click on the map is for a measurement's point and not
    # for an anchor. A third meaning for one gesture, so it is a mode and it is
    # held by a button that stays down -- the alternative, a modifier nobody is
    # told about, is how a click disappears.
    point_wanted = QtCore.pyqtSignal(bool)

    # For the status bar, which belongs to the map.
    said = QtCore.pyqtSignal(str)

    # That the block changed under everything else looking at it.
    wrote = QtCore.pyqtSignal()

    def __init__(self, panel, parent=None):
        super().__init__(parent)

        self.panel = panel

        # The reading rows of the selected block, in file order, as `curation.Row`.
        # Each knows its line and where that line sits, which is what the splice
        # needs. The index into this list is the row number, and the table is
        # never sorted -- `rows_of`'s rule, kept here because the correspondence
        # is what Detach aims through.
        self._in_file = []

        self.about = QtWidgets.QLabel()
        self.about.setWordWrap(True)
        self.about.setStyleSheet("font-weight: bold;")

        self.carries = QtWidgets.QLabel()
        self.carries.setWordWrap(True)
        self.carries.setStyleSheet("color: #6a6a6a; font-size: 11px;")

        self.table = QtWidgets.QTableWidget(0, len(READING_COLUMNS))
        self.table.setHorizontalHeaderLabels(READING_COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(False)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.itemSelectionChanged.connect(self._picked)

        self.why = QtWidgets.QLineEdit()
        self.why.setPlaceholderText(
            "why this measurement is not of this structure"
        )
        self.why.setToolTip(
            "Goes into the file word for word, on the comment that stands where "
            "the reading stood. The next person to open the block reads this and "
            "nothing else about it, so a sentence naming what the plane belongs "
            "to is worth more than `wrong`."
        )
        self.why.textChanged.connect(self._tell)

        self.detach_button = QtWidgets.QPushButton("Detach this reading")
        self.detach_button.setEnabled(False)
        self.detach_button.clicked.connect(self.detach_picked)

        taking_out = QtWidgets.QGroupBox("Take one off this trace")
        out_laid = QtWidgets.QVBoxLayout(taking_out)
        out_laid.addWidget(self.why)
        out_laid.addWidget(self.detach_button)

        # -- and the other direction ---------------------------------------

        # Where the measurement was made, as `(x, y, s, off, aimed)`, or None.
        # `off` is what goes on the line and `aimed` is how far the click itself
        # landed from the trace: the two differ only when the point was snapped,
        # and then the second is the only place that number survives.
        self._point = None

        self.pick_point = QtWidgets.QPushButton("Point on the map")
        self.pick_point.setCheckable(True)
        self.pick_point.setToolTip(
            "Then shift-click the map where the measurement was made. How far "
            "that lands from the trace goes into the line as off=, which is what "
            "says whether this reading was taken on the fault or near it."
        )
        self.pick_point.toggled.connect(self._wanting)

        # Checked, because a fault plane is measured on the fault, and the trace
        # *is* the fault at the surface. The gesture was built unsnapped on the
        # argument that a station is where a person stood and `off=0.0` would lie
        # about it -- which holds for a bedding reading near a fault and is the
        # wrong way round for the fault itself, where the click off the line is
        # the artefact and the line is the record.
        #
        # Nothing about the answer turns on this. `attitude_at` reads `s`, never
        # the offset, and `s` is the same projection either way -- snapping moves
        # the point onto the path at the progressive it already had. What changes
        # is the anchor written in the file, the dot's place on the map, and the
        # statement `off=` makes.
        self.on_trace = QtWidgets.QCheckBox("measured on the trace itself")
        self.on_trace.setChecked(True)
        self.on_trace.setToolTip(
            "Puts the point on the trace at the progressive the click projects "
            "to, and writes off=0.0 -- the format's own example of a field "
            "reading. Leave it checked for a fault plane read on the fault, and "
            "clear it for a measurement made near the trace and not on it, where "
            "the distance is worth keeping.\n\n"
            "Which answer the reading gives does not depend on this: that is "
            "decided by where along the trace the click projects to, and "
            "snapping does not move it along."
        )
        self.on_trace.toggled.connect(self._tell)

        self.where = QtWidgets.QLabel()
        self.where.setStyleSheet("color: #6a6a6a; font-size: 11px;")

        self.dip_dir = QtWidgets.QSpinBox()
        self.dip_dir.setRange(0, 360)
        self.dip_dir.setSuffix("°")
        self.dip_dir.setWrapping(True)
        self.dip_dir.setToolTip(
            "Dip direction, in true azimuth -- a compass reading already "
            "corrected for declination, which is what this format writes. "
            "Nothing is corrected here and nothing about north goes on the "
            "line: a measurement was not computed from grid coordinates, so "
            "there is no convergence to undo."
        )

        self.dip = QtWidgets.QSpinBox()
        self.dip.setRange(0, 90)
        self.dip.setSuffix("°")

        self.station = QtWidgets.QLineEdit()
        self.station.setPlaceholderText("station")
        self.station.setToolTip(
            "The name the measurement is known by in the field notes, written as "
            "station=. Optional, and worth filling: it is how every reading in "
            "these files says which one it is."
        )

        dialling = QtWidgets.QHBoxLayout()
        dialling.addWidget(QtWidgets.QLabel("dip dir"))
        dialling.addWidget(self.dip_dir)
        dialling.addWidget(QtWidgets.QLabel("dip"))
        dialling.addWidget(self.dip)
        dialling.addWidget(self.station, stretch=1)

        self.add_button = QtWidgets.QPushButton("Add this reading")
        self.add_button.setEnabled(False)
        self.add_button.clicked.connect(self.add_reading)

        aiming = QtWidgets.QHBoxLayout()
        aiming.addWidget(self.pick_point)
        aiming.addWidget(self.on_trace, stretch=1)

        putting_in = QtWidgets.QGroupBox("Put one on it")
        in_laid = QtWidgets.QVBoxLayout(putting_in)
        in_laid.addLayout(aiming)
        in_laid.addWidget(self.where)
        in_laid.addLayout(dialling)
        in_laid.addWidget(self.add_button)

        # One Undo for both, because what it undoes is the last press that wrote
        # and there is no sense in which a window has two pasts.
        self.undo_button = QtWidgets.QPushButton("Undo")
        self.undo_button.setEnabled(False)
        self.undo_button.setToolTip(
            "Put the block back as it was before the last press that wrote in "
            "it. Nothing reaches the file until Save."
        )
        self.undo_button.clicked.connect(self.undo_last)

        self.step = QtWidgets.QLabel()
        self.step.setWordWrap(True)
        self.step.setStyleSheet("color: #6a6a6a; font-size: 11px;")

        back = QtWidgets.QHBoxLayout()
        back.addWidget(self.step, stretch=1)
        back.addWidget(self.undo_button)

        laid = QtWidgets.QVBoxLayout(self)
        laid.addWidget(self.about)
        laid.addWidget(self.carries)
        laid.addWidget(self.table, stretch=1)
        laid.addWidget(taking_out)
        laid.addWidget(putting_in)
        laid.addLayout(back)

        self.retarget()

    # -- what the file claims here -----------------------------------------

    def _structure(self):
        if self.panel.index is None:
            return None

        return self.panel.document.dataset.structures[self.panel.index]

    def retarget(self):
        """Points at whatever the panel has open, and reads the block again."""

        structure = self._structure()

        self._in_file = []
        self.table.clearContents()
        self.table.setRowCount(0)

        if structure is None:
            self.about.setText("Nothing selected")
            self.carries.setText("")
            self._tell()
            return

        gstruct = module()

        self.about.setText(
            f"{structure.ident} -- {gstruct.path_length(structure.path):.0f} m"
        )

        self._in_file = readings_in(
            self.panel.document.text_of(self.panel.index), structure.path
        )

        self.table.setRowCount(len(self._in_file))

        for row in range(len(self._in_file)):
            self._write_row(row)

        self.table.resizeColumnsToContents()
        self.carries.setText(self._carries_said())
        self._tell()

    def _carries_said(self):
        """What the block states, and the reach that is not written in any of it."""

        how_many = len(self._in_file)

        if not how_many:
            return (
                "No measurement in the file along this trace -- so whatever "
                "answers here, a fit made it."
            )

        return (
            f"{how_many} measurement{'' if how_many == 1 else 's'}, and each of "
            f"them outranks every fit for {DEFAULT_MAX_GAP:.0f} m either side of "
            f"where it sits. Past that, the nearest one still answers wherever "
            f"no fit does -- which is how a reading comes to speak for ground "
            f"kilometres away from it."
        )

    def _reach_of(self, row):
        """The stretch a reading answers over, clipped to the trace, or None."""

        structure = self._structure()

        if row.place is None or structure is None:
            return None

        gstruct = module()
        length = gstruct.path_length(structure.path)

        return (
            max(0.0, row.place - DEFAULT_MAX_GAP),
            min(length, row.place + DEFAULT_MAX_GAP),
        )

    def _write_row(self, row):
        claim = self._in_file[row]
        reach = self._reach_of(claim)
        attrs = claim.attrs or {}

        rest = " ".join(
            f"{key}={value}"
            for key, value in attrs.items()
            if key not in ("station", "src") and not key.startswith("raw.")
        )

        cells = (
            f"{claim.place:.0f} m" if claim.place is not None else "off the trace",
            (
                f"{claim.plane[0]:.0f}/{claim.plane[1]:.0f}"
                if claim.plane is not None
                else claim.word
            ),
            f"{reach[0]:.0f} to {reach[1]:.0f} m" if reach is not None else "",
            attrs.get("src", ""),
            rest,
        )

        for column, text in enumerate(cells):
            item = QtWidgets.QTableWidgetItem(text)

            if column == 0:
                # The station name rides on the first cell rather than taking a
                # column of its own: it is how a geologist says which reading
                # this is, and it is also absent from anything typed here.
                named = attrs.get("station")

                if named:
                    item.setText(f"{cells[0]}  ({named})")

                item.setToolTip(claim.line.strip())

            self.table.setItem(row, column, item)

    # -- the gesture --------------------------------------------------------

    def _picked_row(self):
        picked = self.table.selectionModel()
        rows = picked.selectedRows() if picked is not None else []

        if not rows:
            return None

        at = rows[0].row()

        return at if at < len(self._in_file) else None

    def _picked(self):
        at = self._picked_row()

        self.showing.emit(None if at is None else self._reach_of(self._in_file[at]))
        self._tell()

    def _tell(self):
        """What is missing before a press can happen, and both presses' state."""

        at = self._picked_row()
        why = self.why.text().strip()
        open_here = self.panel.index is not None

        self.detach_button.setEnabled(at is not None and bool(why))
        self.add_button.setEnabled(self._point is not None)
        self.pick_point.setEnabled(open_here)
        self.undo_button.setEnabled(self.panel.may_undo())
        self.where.setText(self._placed_said())
        self.step.setText(self._step_said(at, why, open_here))

    def _step_said(self, at, why, open_here):
        """
        One line about whichever gesture is in the middle of happening.

        **The gesture under way wins**, and that is the whole of the ordering.
        Two groups of controls could each have something to say at once, and a
        line reporting both would be a line reporting neither -- so a point
        waiting for its plane is said ahead of a row waiting for its reason,
        because the hand that is holding something is the hand to answer.
        """

        if not open_here:
            return ""

        if self.wanting_point():
            return (
                "Shift-click the map where the measurement was made"
                + (
                    " -- it goes on the trace at the progressive it projects to."
                    if self.snapping()
                    else ", and it stays where you click."
                )
            )

        if self._point is not None:
            dialled = (
                "Dial the plane, then `Add this reading`. The dip direction is a "
                "true azimuth, declination already taken off."
            )

            # Said and not refused, because the number has two readings and
            # neither is a mistake: the click may have missed the trace, or the
            # trace may be drawn tens of metres from the fault it stands for,
            # which at 1:50.000 is ordinary and is the case snapping exists for.
            # Which of the two it is, is not something this window can tell.
            if self.snapping() and self._point[3] > SAME_OUTCROP_M:
                return (
                    f"{dialled} The click sits {self._point[3]:.0f} m off the "
                    f"trace and the reading will be written on it -- either the "
                    f"aim missed, or the trace is not drawn where the fault is."
                )

            return dialled

        if at is not None:
            claim = self._in_file[at]

            if not why:
                return (
                    "Say why, and it goes in the file on the comment that "
                    "replaces the line."
                )

            return (
                f"`Detach this reading` takes {reading_said(claim)} out and "
                f"leaves the comment"
                + (
                    ""
                    if from_a_file(claim)
                    else " -- which will be the only copy of it, this one "
                         "carrying no `raw=`"
                )
            )

        if not self._in_file:
            return "Nothing measured along this trace yet."

        return "Pick the reading this trace should not carry."

    # -- and the other direction -------------------------------------------

    def wanting_point(self):
        """Whether the next shift-click on the map belongs to this window."""

        return self.pick_point.isChecked()

    def snapping(self):
        """Whether that click should be put on the trace rather than beside it."""

        return self.on_trace.isChecked()

    def _wanting(self, on):
        if not on:
            self.showing.emit(None)

        self.point_wanted.emit(bool(on))
        self._tell()

    def took_point(self, x, y, s, off):
        """
        Where the click landed, from the map, with `s` and `off` worked out there.

        Kept as it arrived even when the point is going to be snapped, and the
        snap is done at `Add this reading`: a click already taken is the one
        chance to change one's mind about which of the two statements the line
        should make, and a point snapped on arrival has thrown away the click
        that would have to be snapped back.

        `s` and `off` come worked out rather than recomputed because the map has
        a path and this window has a structure, and two places projecting a point
        onto a trace is two places to round it differently.
        """

        self._point = (float(x), float(y), float(s), float(off))
        self.pick_point.setChecked(False)
        self._tell()

    def _placed_said(self):
        """Where the next press would put it, which the checkbox can still move."""

        if self._point is None:
            return ""

        x, y, s, off = self._point

        if not self.snapping():
            return f"{x:.2f}, {y:.2f} -- {s:.0f} m along this trace, {off:.1f} m off it"

        structure = self._structure()
        on_it = point_on(structure.path, s) if structure is not None else (x, y)

        return (
            f"{on_it[0]:.2f}, {on_it[1]:.2f} -- {s:.0f} m along this trace, on it"
            + (f"; the click was {off:.1f} m off" if off > SAME_OUTCROP_M else "")
        )

    def _writing(self):
        """The anchor and the `off=` the next press would write, snapped or not."""

        x, y, s, off = self._point
        structure = self._structure()

        if not self.snapping() or structure is None:
            return (x, y), off

        return point_on(structure.path, s), 0.0

    def add_reading(self):
        """The dialled measurement into the block, at the point that was clicked."""

        if self._point is None:
            return False

        (x, y), off = self._writing()
        attrs = {}
        named = self.station.text().strip()

        if named:
            attrs["station"] = named

        # `src=field` because that is what this window is: a number out of a
        # notebook. The one other value an importer writes is `src=points`, which
        # means a row of a layer, and nothing typed here is that.
        attrs["src"] = "field"

        # Written for the reason `imports` writes it: `s` is derived and looks
        # exact whatever the distance, so without this the file cannot say
        # whether the compass was on the fault or near it. Nothing downstream
        # reads it -- `Anchored.resolve` recomputes the distance from the anchor
        # and `attitude_at` never asks -- so this is the line stating in words
        # what the coordinates already imply, which is the whole of its job.
        attrs["off"] = f"{off:.1f}"

        line = reading_line(x, y, self.dip_dir.value(), self.dip.value(), attrs)
        refused = self.panel.insert_claim(line)

        if refused is not None:
            self.said.emit(refused)

            return False

        self.said.emit(
            f"added {line.strip()} -- Undo takes it out, Save writes the file"
        )

        self._point = None
        self.station.clear()
        self.showing.emit(None)
        self.wrote.emit()
        self.retarget()

        return True

    def detach_picked(self):
        """The reading on the picked row out, with the reason in its place."""

        at = self._picked_row()
        why = self.why.text().strip()

        if at is None or not why:
            return False

        claim = self._in_file[at]
        note = detachment_note(claim, why, time.strftime("%d.%m.%Y"))
        refused = self.panel.comment_out(claim.at, claim.line, note)

        if refused is not None:
            self.said.emit(refused)

            return False

        self.said.emit(
            f"detached {reading_said(claim)} -- the comment says so in the "
            f"block, Undo puts the line back, Save writes the file"
        )
        self.why.clear()
        self.showing.emit(None)
        self.wrote.emit()
        self.retarget()

        return True

    def undo_last(self):
        """The block before the last press that wrote in it, put back."""

        if not self.panel.undo_applied():
            return False

        self.said.emit("the block is back as it was before the last press")
        self.wrote.emit()
        self.retarget()

        return True


class AmendReading(QtWidgets.QWidget):
    """
    One measurement already in the file, moved or corrected, against what it costs.

    `ReadingsHere`'s third gesture and a window of its own, because it is the
    only one here that writes **over** something. Add puts a line where there
    was none and Detach takes one out leaving the reason; both of those are
    decisions about a line's existence, and the file afterwards says what it
    said plus or minus a claim. This one changes what a claim *is*, with no
    second copy of the old one anywhere unless something puts it there.

    **The format decides what that something has to be, and it decides it
    differently for the two halves of a reading.** FORMAT.md's first rule keeps
    the source string beside the normalised value, and on all 44 readings in the
    AOI that string is the plane: `raw="dip_dir=140 dip=35"`, agreeing with the
    slot on every one of them. So rewriting the plane slot loses nothing -- the
    file goes on saying what the source said, a few characters further along the
    same line, and the difference between the two *is* the curation. The anchor
    has no such copy, ever, because the source geometry **is** the anchor: a
    point out of a layer, written down. Move it and nothing anywhere remembers
    where the importer put it.

    Hence: **a move is always recorded in a comment and a correction of the plane
    usually is not**, which is `owed_record`, and the comment carries the reason
    it exists so that a block holding three amendments with comments and a fourth
    without says which rule each of them fell under. The two exceptions are
    measured rather than assumed -- a reading typed in by `ReadingsHere`, which
    has no `raw=`, and a reading whose `raw=` an earlier amendment has already
    overtaken.

    **What it shows before the press is what the move does to the answer, and
    that is not the same question as where the dot goes.** A reading outranks
    every fit for `DEFAULT_MAX_GAP` either side of itself and, past that, still
    answers wherever no fit does. Moving it therefore redraws the provenance of
    the whole trace, which is a computation over several lines at once and the
    one thing reading a file cannot tell you. So the candidate line is parsed --
    `Document.reading_of`, the real parser under the file's own header -- and the
    trace is sampled before and after.

    Two numbers come out of that and they are different, which is the finding
    that shaped this box. Drag S26 1200 m back along `F0055` and **867 m of 3531
    change who answers while the plane changes over none of it**: S26 is the only
    measurement on that trace, so it answers everywhere either way and only the
    tier moves, `misurata` to `misurata-lontana`. Do the same to S20 on `F0074`,
    where a fit is competing, and 999 m change both. A window reporting only the
    first would call those two moves the same size, and one of them changes no
    answer at all.

    **`off=` is written, never typed.** It is the distance from the anchor to the
    trace, nothing downstream reads it -- `Anchored.resolve` recomputes the
    distance and `attitude_at` never asks -- which is exactly what makes a stale
    one pure misinformation: the only thing it can do is contradict the
    coordinates beside it. One reading in the file is already out by a tenth.
    `src` and the `raw.` keys are shown and not editable for their own reasons,
    written on `AMEND_KEPT`.

    **A wrapped line is refused rather than amended**, and that is `continued_at`:
    `loads` folds a line indented four or more into the record above it, `rows_of`
    reads one physical line at a time and cannot see that happening, so a rewrite
    built from a row's attributes would be silently overlaid by the continuation
    and the press would appear to have worked. No claim in either AOI file is
    wrapped; FORMAT.md's own example is.
    """

    # The stretch to light on the map, as `(s0, s1)` or None. The same sink as
    # the other three windows', `_show_fitting` being the one place that decides
    # what a band means -- and here it is the reach the reading would answer
    # over *after* the move, once there is a candidate, because the stretch
    # under discussion is the one the press would create.
    showing = QtCore.pyqtSignal(object)

    # That the next shift-click on the map is for this window's new place. The
    # fourth claimant on one gesture, so it is a mode held by a button that
    # stays down, and the exclusivity is settled in `_only_claimant`.
    point_wanted = QtCore.pyqtSignal(bool)

    # For the status bar, which belongs to the map.
    said = QtCore.pyqtSignal(str)

    # That the block changed under everything else looking at it.
    wrote = QtCore.pyqtSignal()

    def __init__(self, panel, parent=None):
        super().__init__(parent)

        self.panel = panel

        # The reading rows of the selected block, in file order, as
        # `curation.Row`. Unsorted, `rows_of`' rule: the index into this list is
        # the index the splice aims through.
        self._in_file = []

        # Where the press would put it, as `(x, y, s, off)` from the map, or
        # None for "not moved". Kept as the click arrived, snapped at the press
        # -- `ReadingsHere.took_point`'s argument, and the same reason: a click
        # already taken is the one chance to change one's mind about which of
        # the two statements the line should make.
        self._point = None

        # The consequence, cached on the candidate line it was measured for. A
        # parse of the block plus 400 provenance samples per keystroke would be
        # paid for nothing: the same line measures the same.
        self._measured = None
        self._measured_for = None

        self.about = QtWidgets.QLabel()
        self.about.setWordWrap(True)
        self.about.setStyleSheet("font-weight: bold;")

        self.table = QtWidgets.QTableWidget(0, len(READING_COLUMNS))
        self.table.setHorizontalHeaderLabels(READING_COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(False)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.itemSelectionChanged.connect(self._picked)

        # -- where it is ---------------------------------------------------

        self.was = QtWidgets.QLabel()
        self.was.setStyleSheet("color: #6a6a6a; font-size: 11px;")

        self.pick_point = QtWidgets.QPushButton("New place on the map")
        self.pick_point.setCheckable(True)
        self.pick_point.setEnabled(False)
        self.pick_point.setToolTip(
            "Then shift-click the map where the measurement belongs. What that "
            "costs is shown below before anything is written: moving a reading "
            "moves the ground it outranks every fit over, which is 250 m either "
            "side of it."
        )
        self.pick_point.toggled.connect(self._wanting)

        self.on_trace = QtWidgets.QCheckBox("put it on the trace itself")
        self.on_trace.setChecked(True)
        self.on_trace.setToolTip(
            "Puts the anchor on the trace at the progressive the click projects "
            "to and writes off=0.0. Checked for the reason `ReadingsHere` gives: "
            "a fault plane is measured on the fault, and the trace is the fault "
            "at the surface.\n\n"
            "Clear it where the distance is the measurement's own -- the five "
            "readings in these files with a large off= are the dip slopes "
            "FORMAT.md names, measured 44 to 70 m from the trace because that is "
            "where the surface is."
        )
        self.on_trace.toggled.connect(self._changed)

        self.drop_point = QtWidgets.QPushButton("Leave it where it is")
        self.drop_point.setEnabled(False)
        self.drop_point.setToolTip(
            "Forgets the picked place, so the press amends only what the line "
            "says and not where it sits."
        )
        self.drop_point.clicked.connect(self.forget_point)

        self.where = QtWidgets.QLabel()
        self.where.setWordWrap(True)
        self.where.setStyleSheet("font-size: 11px;")

        aiming = QtWidgets.QHBoxLayout()
        aiming.addWidget(self.pick_point)
        aiming.addWidget(self.on_trace, stretch=1)
        aiming.addWidget(self.drop_point)

        moving = QtWidgets.QGroupBox("Where it is")
        moving_laid = QtWidgets.QVBoxLayout(moving)
        moving_laid.addWidget(self.was)
        moving_laid.addLayout(aiming)
        moving_laid.addWidget(self.where)

        # -- what it says --------------------------------------------------

        self.dip_dir = QtWidgets.QSpinBox()
        self.dip_dir.setRange(0, 360)
        self.dip_dir.setSuffix("°")
        self.dip_dir.setWrapping(True)
        self.dip_dir.setEnabled(False)
        self.dip_dir.setToolTip(
            "Dip direction, in true azimuth. Whole degrees, which is "
            "`reading_line`'s rule and holds just as well for a correction: a "
            "compass already corrected for declination reads in the azimuth this "
            "format writes, so there is no `converg=` for a decimal to carry."
        )
        self.dip_dir.valueChanged.connect(self._changed)

        self.dip = QtWidgets.QSpinBox()
        self.dip.setRange(0, 90)
        self.dip.setSuffix("°")
        self.dip.setEnabled(False)
        self.dip.valueChanged.connect(self._changed)

        self.apart = QtWidgets.QLabel()
        self.apart.setStyleSheet("color: #6a6a6a; font-size: 11px;")

        dialling = QtWidgets.QHBoxLayout()
        dialling.addWidget(QtWidgets.QLabel("dip dir"))
        dialling.addWidget(self.dip_dir)
        dialling.addWidget(QtWidgets.QLabel("dip"))
        dialling.addWidget(self.dip)
        dialling.addWidget(self.apart, stretch=1)

        self.attrs = QtWidgets.QTableWidget(0, len(AMEND_COLUMNS))
        self.attrs.setHorizontalHeaderLabels(AMEND_COLUMNS)
        self.attrs.verticalHeader().setVisible(False)
        self.attrs.setSortingEnabled(False)
        self.attrs.setMaximumHeight(AMEND_TABLE_PX)
        self.attrs.horizontalHeader().setStretchLastSection(True)
        self.attrs.setToolTip(
            "The line's own attributes. Type in a value to change it, empty it "
            "to take the key off, and `Another attribute` to add one. The grey "
            "rows are not editable: `src` says who produced the line, `off` is "
            "derived from the anchor and written here, and `raw` is the source "
            "string the format conserves."
        )
        self.attrs.itemChanged.connect(self._attr_changed)

        self.add_attr = QtWidgets.QPushButton("Another attribute...")
        self.add_attr.setEnabled(False)
        self.add_attr.clicked.connect(self.add_attribute)

        saying = QtWidgets.QGroupBox("What it says")
        saying_laid = QtWidgets.QVBoxLayout(saying)
        saying_laid.addLayout(dialling)
        saying_laid.addWidget(self.attrs, stretch=1)
        saying_laid.addWidget(self.add_attr)

        # -- and what that costs -------------------------------------------

        self.cost = QtWidgets.QLabel()
        self.cost.setWordWrap(True)
        self.cost.setStyleSheet("font-size: 11px;")
        self.cost.setToolTip(
            "What the press would do to the answer along this trace, measured "
            "by parsing the candidate line and asking `attitude_at` everywhere "
            "before and after -- not by this window's idea of what the parser "
            "does."
        )

        costing = QtWidgets.QGroupBox("What that changes")
        costing_laid = QtWidgets.QVBoxLayout(costing)
        costing_laid.addWidget(self.cost)

        self.why = QtWidgets.QLineEdit()
        self.why.setPlaceholderText("why this reading is not as the file has it")
        self.why.setToolTip(
            "Goes into the file word for word. Required where the amendment owes "
            "a comment -- always for a move -- and written into it there; asked "
            "for anyway where it does not, and then it goes on the status bar "
            "and into nothing, which the window says."
        )
        self.why.textChanged.connect(self._tell)

        self.amend_button = QtWidgets.QPushButton("Amend this reading")
        self.amend_button.setEnabled(False)
        self.amend_button.clicked.connect(self.amend)

        self.undo_button = QtWidgets.QPushButton("Undo")
        self.undo_button.setEnabled(False)
        self.undo_button.setToolTip(
            "Put the block back as it was before the last press that wrote in "
            "it. Nothing reaches the file until Save."
        )
        self.undo_button.clicked.connect(self.undo_last)

        self.step = QtWidgets.QLabel()
        self.step.setWordWrap(True)
        self.step.setStyleSheet("color: #6a6a6a; font-size: 11px;")

        back = QtWidgets.QHBoxLayout()
        back.addWidget(self.step, stretch=1)
        back.addWidget(self.undo_button)

        laid = QtWidgets.QVBoxLayout(self)
        laid.addWidget(self.about)
        laid.addWidget(self.table, stretch=1)
        laid.addWidget(moving)
        laid.addWidget(saying)
        laid.addWidget(costing)
        laid.addWidget(self.why)
        laid.addWidget(self.amend_button)
        laid.addLayout(back)

        self.retarget()

    # -- what the file claims here -----------------------------------------

    def _structure(self):
        if self.panel.index is None:
            return None

        return self.panel.document.dataset.structures[self.panel.index]

    def retarget(self):
        """Points at whatever the panel has open, and reads the block again.

        Everything picked goes with it, the point included: a place clicked on
        the trace just left is a coordinate on another fault, and the controls
        below would be holding a plane belonging to a reading that is no longer
        on screen.
        """

        structure = self._structure()

        self._in_file = []
        self._point = None
        self._measured = self._measured_for = None
        self.table.clearContents()
        self.table.setRowCount(0)
        self._fill_attrs(None)

        if structure is None:
            self.about.setText("Nothing selected")
            self._tell()
            return

        gstruct = module()

        self.about.setText(
            f"{structure.ident} -- {gstruct.path_length(structure.path):.0f} m"
        )

        self._in_file = readings_in(
            self.panel.document.text_of(self.panel.index), structure.path
        )

        self.table.setRowCount(len(self._in_file))

        for row in range(len(self._in_file)):
            self._write_row(row)

        self.table.resizeColumnsToContents()
        self._tell()

    def _reach_of(self, row):
        """The stretch a reading answers over, clipped to the trace, or None.

        `ReadingsHere._reach_of` with one difference: `row` here can be a
        progressive rather than a `Row`, so that the reach of the place a click
        picked is worked out by the same arithmetic as the reach of the place the
        file holds. Two readings of `DEFAULT_MAX_GAP` would be two bands on one
        map meaning two slightly different things.
        """

        structure = self._structure()
        place = row if isinstance(row, float) else row.place

        if place is None or structure is None:
            return None

        gstruct = module()
        length = gstruct.path_length(structure.path)

        return (
            max(0.0, place - DEFAULT_MAX_GAP),
            min(length, place + DEFAULT_MAX_GAP),
        )

    def _write_row(self, row):
        """One row of the readings table. `ReadingsHere._write_row`'s columns."""

        claim = self._in_file[row]
        reach = self._reach_of(claim)
        attrs = claim.attrs or {}

        rest = " ".join(
            f"{key}={value}"
            for key, value in attrs.items()
            if key not in ("station", "src") and not key.startswith("raw.")
        )

        cells = (
            f"{claim.place:.0f} m" if claim.place is not None else "off the trace",
            (
                f"{claim.plane[0]:.0f}/{claim.plane[1]:.0f}"
                if claim.plane is not None
                else claim.word
            ),
            f"{reach[0]:.0f} to {reach[1]:.0f} m" if reach is not None else "",
            attrs.get("src", ""),
            rest,
        )

        for column, text in enumerate(cells):
            item = QtWidgets.QTableWidgetItem(text)

            if column == 0:
                named = attrs.get("station")

                if named:
                    item.setText(f"{cells[0]}  ({named})")

                item.setToolTip(claim.line.strip())

            self.table.setItem(row, column, item)

    # -- the row under the hand --------------------------------------------

    def _picked_row(self):
        picked = self.table.selectionModel()
        rows = picked.selectedRows() if picked is not None else []

        if not rows:
            return None

        at = rows[0].row()

        return at if at < len(self._in_file) else None

    def _picked(self):
        """A row picked loads its own values into the controls.

        **Its own, and not the last row's**, which is the trap `_show_held`
        found in the panel: a dial left on the previous reading's plane is a
        number nobody chose, sitting in a box that is about to write it. So
        everything is filled from the row, the picked place is dropped, and
        what is on screen is the file until somebody changes it.
        """

        at = self._picked_row()

        self._point = None
        self._measured = self._measured_for = None

        if at is None:
            self._fill_attrs(None)
            self.showing.emit(None)
            self._tell()
            return

        claim = self._in_file[at]

        if claim.plane is not None:
            for box, value in ((self.dip_dir, claim.plane[0]), (self.dip, claim.plane[1])):
                box.blockSignals(True)
                box.setValue(int(round(value)))
                box.blockSignals(False)

        self._fill_attrs(claim)
        self.showing.emit(self._reach_of(claim))
        self._tell()

    def _refused(self, claim):
        """Why this row cannot be amended at all, or None.

        Settled on the row rather than at the press, so that a reading this tool
        must not rewrite is one somebody is told about before they have typed a
        reason for changing it.
        """

        if claim.plane is None:
            return (
                f"a `{claim.word}` with no plane this tool can read: what it "
                f"holds in that slot is not two numbers, so there is nothing "
                f"here to dial and nothing to write back"
            )

        if anchor_of(claim.line) is None:
            return (
                "no anchor on the line, so it has no place on the trace to move "
                "from -- a `*` here is not an end of the path, it is a reading "
                "nobody has pinned"
            )

        wrapped = continued_at(self.panel.document.text_of(self.panel.index), claim.at)

        if wrapped:
            return (
                f"this line is continued on {len(wrapped)} more, and `loads` "
                f"merges their attributes into it. Rewriting the first would "
                f"leave those in place to be applied over the amendment, so the "
                f"press would appear to work and change nothing"
            )

        return None

    # -- where it is -------------------------------------------------------

    def wanting_point(self):
        """Whether the next shift-click on the map belongs to this window."""

        return self.pick_point.isChecked()

    def snapping(self):
        """Whether that click should be put on the trace rather than beside it."""

        return self.on_trace.isChecked()

    def _wanting(self, on):
        self.point_wanted.emit(bool(on))
        self._tell()

    def took_point(self, x, y, s, off):
        """Where the click landed, from the map, with `s` and `off` from there.

        `ReadingsHere.took_point` and the same contract: kept as it arrived, and
        the snap happens at the press. `s` and `off` come worked out rather than
        recomputed because the map has the path -- two places projecting a point
        onto a trace is two places to round it differently.
        """

        self._point = (float(x), float(y), float(s), float(off))
        self._measured = self._measured_for = None
        self.pick_point.setChecked(False)
        self.showing.emit(self._reach_of(float(s)))
        self._tell()

    def forget_point(self):
        """The picked place dropped, leaving the attributes still amendable."""

        at = self._picked_row()

        self._point = None
        self._measured = self._measured_for = None
        self.showing.emit(
            None if at is None else self._reach_of(self._in_file[at])
        )
        self._tell()

    def _writing(self):
        """The anchor and the `off=` the press would write, snapped or not."""

        if self._point is None:
            return None, None

        x, y, s, off = self._point
        structure = self._structure()

        if not self.snapping() or structure is None:
            return (x, y), off

        return point_on(structure.path, s), 0.0

    def _was_said(self, claim):
        """Where the file has it, in the units the move will be talked about in."""

        anchor = anchor_of(claim.line)

        if anchor is None or claim.place is None:
            return "no anchor on this line"

        off = (claim.attrs or {}).get("off")

        return (
            f"the file has it at {anchor[0]:.2f}, {anchor[1]:.2f} -- "
            f"{claim.place:.0f} m along this trace"
            + (f", off={off}" if off else "")
        )

    def _where_said(self, claim):
        """Where the press would put it, which the checkbox can still move."""

        if self._point is None:
            return ""

        anchor, off = self._writing()
        _, _, s, aimed = self._point
        moved = s - claim.place

        return (
            f"would go to {anchor[0]:.2f}, {anchor[1]:.2f} -- {s:.0f} m along, "
            + ("on the trace" if off == 0.0 else f"{off:.1f} m off it")
            + f"; {abs(moved):.0f} m "
            + ("further along" if moved > 0 else "back along")
            + (
                f". The click itself was {aimed:.0f} m off the trace"
                if off == 0.0 and aimed > SAME_OUTCROP_M
                else ""
            )
        )

    # -- what it says ------------------------------------------------------

    def _fill_attrs(self, claim):
        """The line's attributes into the table, the kept ones greyed.

        Rebuilt rather than patched, and the signal is held off while it is: a
        table whose cells are being written emits `itemChanged` for every one of
        them, and the handler's job is to notice that somebody typed.
        """

        self.attrs.blockSignals(True)
        self.attrs.clearContents()
        self.attrs.setRowCount(0)

        if claim is not None:
            held = list((claim.attrs or {}).items())
            self.attrs.setRowCount(len(held))

            for row, (key, value) in enumerate(held):
                kept = self._kept(key)

                name = QtWidgets.QTableWidgetItem(key)
                name.setFlags(name.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)

                cell = QtWidgets.QTableWidgetItem(value)

                if kept:
                    cell.setFlags(cell.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)

                    for item in (name, cell):
                        item.setForeground(QtGui.QColor("#9a9a9a"))
                        item.setToolTip(self._kept_why(key))

                self.attrs.setItem(row, 0, name)
                self.attrs.setItem(row, 1, cell)

        self.attrs.resizeColumnsToContents()
        self.attrs.blockSignals(False)

    @staticmethod
    def _kept(key):
        """Whether this window shows a key without letting anybody type over it."""

        return key in AMEND_KEPT or any(
            key == one or key.startswith(f"{one}.") for one in AMEND_RAW
        )

    @staticmethod
    def _kept_why(key):
        """The reason this key is not editable, which is different for each."""

        if key == "src":
            return (
                "Who produced the line. A curator editing this forges a "
                "provenance, and the whole precedence rests on telling a compass "
                "from a computation."
            )

        if key == "off":
            return (
                "Derived from the anchor, and written here from the anchor this "
                "press writes. Nothing downstream reads it, which is what makes "
                "a stale one pure misinformation: all it can do is contradict "
                "the coordinates beside it."
            )

        return (
            "The source string, which FORMAT.md's first rule conserves beside "
            "the normalised value -- and which `owed_record` reads to decide "
            "whether the old plane has anywhere to be. A conserved string "
            "somebody has edited is not one."
        )

    def _attr_changed(self, item):
        """A typed cell re-measures the candidate. Added keys come in empty."""

        if item.column() == 1:
            self._changed()

    def add_attribute(self):
        """An empty row for a key the line has not got.

        The key is typed as well as the value, because the format's attributes
        are open -- `note`, `date`, `site_note` and the `raw.` namespace are all
        an importer's inventions -- and a window offering a fixed list would be
        this tool deciding what a geologist is allowed to record.
        """

        at = self.attrs.rowCount()

        self.attrs.blockSignals(True)
        self.attrs.insertRow(at)
        self.attrs.setItem(at, 0, QtWidgets.QTableWidgetItem(""))
        self.attrs.setItem(at, 1, QtWidgets.QTableWidgetItem(""))
        self.attrs.blockSignals(False)

        self.attrs.editItem(self.attrs.item(at, 0))

    def _typed_attrs(self, claim):
        """The table against the line, as `{key: value or None}`: the differences.

        Differences and not the whole table, for `gstruct_writer`'s reason: a
        line rewritten from a dictionary comes back in the dictionary's spacing,
        and `with_values` splices only the keys that moved. An emptied value is
        `None`, which takes the key off.
        """

        held = dict(claim.attrs or {})
        out = {}

        for row in range(self.attrs.rowCount()):
            name = self.attrs.item(row, 0)
            cell = self.attrs.item(row, 1)

            if name is None:
                continue

            key = name.text().strip()
            value = (cell.text() if cell is not None else "").strip()

            if not key or self._kept(key):
                continue

            if key in held:
                if value != held[key]:
                    out[key] = value or None

                held.pop(key)
            elif value:
                out[key] = value

        # A row taken out of the table -- which nothing here offers yet, but the
        # difference has to be computed against the table and not against what
        # the table was built from, or a key would survive its own row.
        for key in held:
            if not self._kept(key):
                out[key] = None

        return out

    # -- the candidate, and what it costs ----------------------------------

    def _candidate(self, claim):
        """The line the press would write, with the anchor and plane it carries.

        Returns `(line, anchor, plane)`, the first None where there is nothing
        to write. Built by splicing the row's own line three times rather than
        by rebuilding it from `reading_line`: the comment at the end of a line,
        the order of its attributes and the spacing somebody lined up are all
        outside the slots being changed, and this tool's rule is that they stay
        where they are.

        **Whole degrees, as `reading_line` writes them**, and the decimal is
        taken off deliberately rather than left at `PLANE_DECIMALS`. That
        default exists because a fit's number was computed in grid and corrected
        by `converg=`, which at `.0f` would vanish while claiming to have
        happened. A compass has no correction to lose, every `attitude` in the
        AOI is written whole, and one written otherwise would look as though it
        had come from somewhere else.
        """

        anchor, off = self._writing()
        plane = (float(self.dip_dir.value()), float(self.dip.value()))
        line = claim.line

        if anchor is not None:
            line = with_anchor(line, *anchor)

            if line is None:
                return None, None, None

            # Written from the anchor that is about to go on the line, in the
            # same press, which is the only way the two cannot disagree.
            line = with_values(line, {"off": f"{off:.1f}"})

        if plane != claim.plane:
            line = with_plane(line, plane[0], plane[1], decimals=0)

            if line is None:
                return None, None, None

        typed = self._typed_attrs(claim)

        if typed:
            line = with_values(line, typed)

        if line == claim.line:
            return None, None, None

        return line, anchor, plane

    def _measure(self, claim, line):
        """What the candidate does to the answer along this trace, or None.

        Cached on the line, which is what the measurement is of. Returns
        `(moved_m, replanned_m, length, before, after)` in metres of trace.
        """

        if self._measured_for == line:
            return self._measured

        self._measured = self._measured_for = None

        structure = self._structure()

        if structure is None:
            return None

        text = self.panel.document.text_of(self.panel.index)
        lines = text.splitlines()

        if not 0 <= claim.at < len(lines):
            return None

        lines[claim.at] = line
        after = self.panel.document.reading_of("\n".join(lines))

        if after is None:
            return None

        max_gap = self.panel.max_gap
        before_p = provenance_of(structure, max_gap=max_gap)
        after_p = provenance_of(after, max_gap=max_gap)

        if not before_p or len(before_p) != len(after_p):
            return None

        length = before_p[-1][0]

        # What one sample stands for, and `length / samples` rather than
        # `length / (samples - 1)`: the second is the spacing *between* samples,
        # and summing it once per sample gives one spacing more than the trace
        # has. Measured, on `F0059`: 5712 m reported of a 5698 m fault, which is
        # a sentence that cannot be true of anything.
        step = length / len(before_p)

        # Who answers and what the answer is, counted apart. They are different
        # quantities and the difference is this box's whole point: on `F0055` a
        # move changes who answers over 109 m and the plane over none of it, S26
        # being the only measurement on that trace.
        #
        # **Keyed on the line and not on the tier**, which the AOI corrected
        # twice. The tier alone -- `misurata`, `fit`, `misurata-lontana` -- was
        # too coarse: on `F0058`, where S22 and S25 sit 16 m apart and disagree
        # by 14 degrees, one or the other is the nearest reading nearly
        # everywhere, so the tier moved over 34 m while *which of the two
        # answers* moved over 841. And `said` whole was too fine, because it
        # ends in the distance it answered from, which changes at every metre.
        # `answering` is the middle, and it is the identity the string carries.
        #
        # The plane goes into the first key as well as being counted on its own,
        # so that the two numbers nest the way the sentence reads them. Without
        # it they would not: `misurata-lontana` names no station, so two
        # different far readings compare equal there and the plane could change
        # over ground this said nobody new was answering.
        moved = sum(
            step for b, a in zip(before_p, after_p)
            if (answering(b[2]), b[1]) != (answering(a[2]), a[1])
        )
        replanned = sum(step for b, a in zip(before_p, after_p) if b[1] != a[1])

        self._measured = (moved, replanned, length, before_p, after_p)
        self._measured_for = line

        return self._measured

    def _cost_said(self, claim, line, anchor, plane):
        """What the press would change, in metres of this trace, and what it owes."""

        if line is None:
            return (
                "Nothing is different from what the file says, so there is "
                "nothing to write."
            )

        measured = self._measure(claim, line)

        if measured is None:
            return (
                "The candidate line does not parse, so what it would change "
                "cannot be measured -- `Amend this reading` will refuse it with "
                "the parser's own words."
            )

        moved, replanned, length, before_p, after_p = measured
        said = []

        if moved < AMEND_MOVED_M and replanned < AMEND_MOVED_M:
            said.append(
                f"No metre of these {length:.0f} answers differently afterwards."
                + (
                    " The line changes and the answer does not, which for an "
                    "attribute is the normal case: nothing downstream reads one."
                    if anchor is None
                    else " The reading moves and keeps on being the nearest one "
                    "everywhere, which is what a trace with a single "
                    "measurement on it does."
                )
            )
        else:
            said.append(
                f"{moved:.0f} m of {length:.0f} are answered by a different "
                f"line afterwards, and "
                + (
                    f"over {replanned:.0f} of those the plane is different too."
                    if replanned >= AMEND_MOVED_M
                    else "the plane is the same over all of them -- the same "
                    "reading still answers, from further off."
                )
            )

            changes = self._where_it_turns(before_p, after_p)

            if changes:
                said.append(changes)

        owed = owed_record(claim, anchor, plane)

        if owed is not None:
            said.append(f"The line as it stands goes into a comment: {owed}.")
        elif plane is not None and plane != claim.plane:
            said.append(
                f"No comment is owed: `raw=` on the line already reads "
                f"{claim.plane[0]:.0f}/{claim.plane[1]:.0f}, so the file goes on "
                f"saying what the source said."
            )
        else:
            # Said apart from the sentence above it, because the reason is a
            # different one and the `raw=` sentence over an attribute edit is a
            # non-sequitur: nothing about the measurement is being written over,
            # so there is nothing for the source string to be standing beside.
            said.append(
                "No comment is owed: the measurement itself is not being "
                "written over."
            )

        return " ".join(said)

    @staticmethod
    def _where_it_turns(before_p, after_p, most=3):
        """The first few stretches that change hands, named by both answers.

        The runs and not the samples: `runs_of`' argument, and it is stronger
        here -- what is being reported is where the answer changes, and between
        two changes there is one answer to report.

        Named by the station or the verdict and not by the tier, `_measure`'s
        correction and the same case behind it: on `F0058` the tier changes over
        14 m and which of two readings answers changes over 841, so a line
        reporting tiers reported the small one and left the large one out.
        """

        runs = []

        for before, after in zip(before_p, after_p):
            s = before[0]
            was = answering(before[2])
            now = answering(after[2])

            if was == now:
                continue

            if runs and runs[-1][1] == (was, now):
                runs[-1][0] = (runs[-1][0][0], s)
                continue

            runs.append([(s, s), (was, now)])

        if not runs:
            return ""

        said = ", ".join(
            f"{a:.0f}-{b:.0f} m {was} -> {now}"
            for (a, b), (was, now) in runs[:most]
        )

        return said + (f", and {len(runs) - most} more" if len(runs) > most else "")

    # -- telling -----------------------------------------------------------

    def _changed(self):
        """A control moved: the candidate is stale, so the measurement is too."""

        self._measured = self._measured_for = None
        self._tell()

    def _tell(self):
        """Every control's state, and one line about the gesture under way."""

        at = self._picked_row()
        claim = None if at is None else self._in_file[at]
        refused = None if claim is None else self._refused(claim)
        live = claim is not None and refused is None
        why = self.why.text().strip()

        line = anchor = plane = None

        if live:
            line, anchor, plane = self._candidate(claim)

        owed = None if not live else owed_record(claim, anchor, plane)

        for control in (self.pick_point, self.dip_dir, self.dip, self.add_attr):
            control.setEnabled(live)

        self.drop_point.setEnabled(live and self._point is not None)
        self.why.setEnabled(live)

        # Required where a comment will be written and offered where none will:
        # the comment is the whole justification for writing over a coordinate
        # nothing else holds, and a reason box that could be left empty is a
        # reason box that is left empty.
        self.amend_button.setEnabled(
            live and line is not None and (owed is None or bool(why))
        )

        self.undo_button.setEnabled(self.panel.may_undo())

        self.was.setText("" if claim is None else self._was_said(claim))
        self.where.setText("" if claim is None else self._where_said(claim))
        self.apart.setText("" if not live else self._apart_said(claim, plane))
        self.cost.setText(
            refused or ("" if not live else self._cost_said(claim, line, anchor, plane))
        )
        self.step.setText(self._step_said(claim, refused, line, owed, why))
        self._show_off()

    def _show_off(self):
        """The greyed `off` cell showing what the press would write, not what was.

        Because this window writes that number and the table shows it: left
        alone, a reading dragged onto the trace would sit above a row saying
        `off=44.2` while the press wrote `off=0.0`, which is the same stale copy
        the key is greyed out to prevent.
        """

        _, off = self._writing()

        for row in range(self.attrs.rowCount()):
            name = self.attrs.item(row, 0)

            if name is None or name.text().strip() != "off":
                continue

            cell = self.attrs.item(row, 1)

            if cell is None:
                continue

            at = self._picked_row()
            was = (
                "" if at is None
                else (self._in_file[at].attrs or {}).get("off", "")
            )

            self.attrs.blockSignals(True)
            cell.setText(was if off is None else f"{off:.1f}")
            self.attrs.blockSignals(False)

    @staticmethod
    def _apart_said(claim, plane):
        """How far the dialled plane is from the one the line holds."""

        if claim.plane is None:
            return ""

        if plane is None or plane == claim.plane:
            return f"what the line says: {claim.plane[0]:.0f}/{claim.plane[1]:.0f}"

        return (
            f"{between(claim.plane, plane):.0f}° from the "
            f"{claim.plane[0]:.0f}/{claim.plane[1]:.0f} the line says"
        )

    def _step_said(self, claim, refused, line, owed, why):
        """One line about whichever gesture is in the middle of happening."""

        if self.panel.index is None:
            return ""

        if not self._in_file:
            return "Nothing measured along this trace to amend."

        if claim is None:
            return "Pick the reading that is not as the file has it."

        if refused is not None:
            return "This one cannot be amended here -- see above."

        if self.wanting_point():
            return (
                "Shift-click the map where it belongs"
                + (
                    " -- it goes on the trace at the progressive it projects to."
                    if self.snapping()
                    else ", and it stays where you click."
                )
            )

        if line is None:
            return (
                "Move the point, dial the plane, or type over an attribute. "
                "What any of that costs is measured below before anything is "
                "written."
            )

        if owed is not None and not why:
            return (
                "Say why, and it goes into the comment that keeps the line as "
                "it stands -- which is owed here and is not owed for every "
                "amendment, so the comment carries the reason it exists."
            )

        if owed is None and not why:
            return (
                "`Amend this reading` writes it. A reason is not required here "
                "and is worth typing anyway: it goes on the status bar, where "
                "the next press overwrites it."
            )

        return "`Amend this reading` writes it, and Undo takes it back."

    # -- the press ---------------------------------------------------------

    def amend(self):
        """The candidate line into the block, with the comment where one is owed."""

        at = self._picked_row()

        if at is None:
            return False

        claim = self._in_file[at]

        if self._refused(claim) is not None:
            return False

        line, anchor, plane = self._candidate(claim)

        if line is None:
            return False

        why = self.why.text().strip()
        owed = owed_record(claim, anchor, plane)

        if owed is not None and not why:
            return False

        into = (
            line
            if owed is None
            else amend_note(claim, why, time.strftime("%d.%m.%Y"), owed)
            + "\n"
            + line
        )

        refused = self.panel.amend_claim(claim.at, claim.line, into)

        if refused is not None:
            self.said.emit(refused)

            return False

        self.said.emit(
            f"amended {reading_said(claim)}"
            + (f" -- {why}" if why else "")
            + (
                ", and the line as it stands is in the comment above it"
                if owed is not None
                else ""
            )
            + " -- Undo puts it back, Save writes the file"
        )

        self.why.clear()
        self._point = None
        self.showing.emit(None)
        self.wrote.emit()
        self.retarget()

        return True

    def undo_last(self):
        """The block before the last press that wrote in it, put back."""

        if not self.panel.undo_applied():
            return False

        self.said.emit("the block is back as it was before the last press")
        self.wrote.emit()
        self.retarget()

        return True


class ExposureHere(QtWidgets.QWidget):
    """
    Whether the contact crops out along a stretch, said by the only one who knows.

    **The axis the source cannot fill.** `certainty` and `exposure` are what a
    source says about a contact, and on these files one of them says nothing:
    `exposure` is `unknown` on all 393 structures of `montealpi_01.gstruct` and
    on all 393 of `merid_faults`, written `reason=assente-in-sorgente` because
    `geology.gpkg` has no such column. The five `exposed` spans in this project
    were typed into `curation.gstruct` by hand. That is the axis this window is
    for, and the reason it is the fourth operation here rather than a fifth: the
    `fit` on a dip slope, FORMAT.md's facet, is **licensed** by a span on it, and
    without one the calculation has nothing to run on.

    **What FORMAT.md asks of it.** `drape` -- the angle between a fitted plane
    and the hillside the trace lies on -- classifies and does not reject, and the
    document says why: the distinction between *a contact exhumed as a dip slope*
    and *a trace digitised along a break of slope* is not statistical, it is
    decided by this axis. 17 of the 27 fits in `merid_faults` came back
    `concorde-col-versante`, so this is not a corner case; it is the state of
    most of the file, and the thing that resolves it is a sentence a geologist
    writes.

    **So the hillside is shown and nothing is concluded from it.** `hillside_on`
    reads the ground 30 and 60 m either side of the picked stretch and fits a
    plane to it; the window prints that plane, how far it misses the cells, how
    much relief it was fitted through, and the angle from it to every measurement
    and every fit the stretch already carries, named one at a time. No threshold,
    no verdict, no greyed-out button where the angle is large -- a hillside
    parallel to a measured fault plane is exactly as consistent with the trace
    having been drawn along a scarp, and the whole point of the axis is that only
    a person can tell those apart. `vs_field` in `export_geology.py` is what
    happens when a script tries: the angle it checked was the angle to the very
    measurement that had grown the region, so the test could not fail.

    **A correction is a line added.** `span_at` returns the *last* span covering
    a metre, and the writing here goes above the `path`, which is below every
    assertion already in the block -- so declaring a stretch narrows whatever
    held over it without touching the line that held. Nothing in here removes a
    span, and that is the format's own idiom rather than a missing button: the
    general line stays in the file saying what the source said, and the reason it
    no longer answers is readable as the line that came after it. The `in force`
    column is where that is shown, and it is metres and not a tick, because a
    span can be shadowed over part of itself.

    **And it needs no DEM.** The declaration is a field observation; the hillside
    is evidence, and evidence that is often absent -- 13 of the 393 traces of
    `merid_faults` have no DEM under them at all. So the raster half of this
    window can be empty with the other half still working, which is the same rule
    the measurements window follows and for the same reason.
    """

    # The stretch to light on the map, as `(s0, s1)` or None. The same sink as
    # the other two windows', `_show_fitting` being the one place that decides
    # what a band on the map means.
    showing = QtCore.pyqtSignal(object)

    # That the next shift-click on the map is an end of this window's stretch.
    # The third claimant on that gesture, so it is a held mode with a button that
    # stays down, for the reason `ReadingsHere.point_wanted` gives.
    ends_wanted = QtCore.pyqtSignal(bool)

    said = QtCore.pyqtSignal(str)
    wrote = QtCore.pyqtSignal()

    def __init__(self, panel, parent=None):
        super().__init__(parent)

        self.panel = panel

        # The `span` rows of the open block that sit on this axis, in file order
        # -- which is the order the rule reads them in. Never sorted, for
        # `rows_of`'s reason.
        self._in_file = []

        # The picked ends as progressives, `(s0, s1)`, or None for the whole
        # trace. Two numbers and not two anchors: the anchors are computed when
        # the line is written, so that a second thought about which end is which
        # does not need the clicks taken again.
        self._ends = None

        # What the first of two clicks left, until the second arrives.
        self._first = None

        # The last hillside read, against the stretch it was read for, so that
        # retargeting does not reprint a plane measured somewhere else.
        self._hillside = None
        self._hillside_for = None

        self.about = QtWidgets.QLabel()
        self.about.setWordWrap(True)
        self.about.setStyleSheet("font-weight: bold;")

        self.rule = QtWidgets.QLabel(
            "The last span covering a metre is the one in force, so a correction "
            "on this axis is a line added and not a line changed -- what is "
            "written here goes below everything already in the block."
        )
        self.rule.setWordWrap(True)
        self.rule.setStyleSheet("color: #6a6a6a; font-size: 11px;")

        self.table = QtWidgets.QTableWidget(0, len(EXPOSURE_COLUMNS))
        self.table.setHorizontalHeaderLabels(EXPOSURE_COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(False)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.itemSelectionChanged.connect(self._picked)

        # -- the stretch -------------------------------------------------------

        self.pick_ends = QtWidgets.QPushButton("Ends on the map")
        self.pick_ends.setCheckable(True)
        self.pick_ends.setToolTip(
            "Then shift-click the two ends of the stretch on the map. Each click "
            "is projected onto this trace; the pair is put in order when it is "
            "written, because a span written backwards covers no ground."
        )
        self.pick_ends.toggled.connect(self._wanting)

        self.whole = QtWidgets.QCheckBox("the whole trace")
        self.whole.setToolTip(
            "Writes the two ends as `*`, which is the format's word for an end "
            "of the path and not a coordinate at it: redigitised past its old "
            "end, the trace keeps the claim over all of itself.\n\n"
            "This is what the source's own line says -- `* * unknown` -- so a "
            "declaration over the whole trace is a statement about the whole "
            "fault, which is rarely what a dip slope is."
        )
        self.whole.toggled.connect(self._whole_changed)

        self.where = QtWidgets.QLabel()
        self.where.setWordWrap(True)
        self.where.setStyleSheet("color: #6a6a6a; font-size: 11px;")

        picking = QtWidgets.QHBoxLayout()
        picking.addWidget(self.pick_ends)
        picking.addWidget(self.whole, stretch=1)

        stretching = QtWidgets.QGroupBox("The stretch")
        stretch_laid = QtWidgets.QVBoxLayout(stretching)
        stretch_laid.addLayout(picking)
        stretch_laid.addWidget(self.where)

        # -- what the ground there looks like ----------------------------------

        self.ground = QtWidgets.QLabel()
        self.ground.setWordWrap(True)
        self.ground.setStyleSheet("font-size: 11px;")

        self.angles = QtWidgets.QLabel()
        self.angles.setWordWrap(True)
        self.angles.setStyleSheet("font-size: 11px; color: #333333;")

        caveat = QtWidgets.QLabel(
            "A hillside parallel to the plane here is what an exhumed dip slope "
            "looks like, and also what a trace drawn along a break of slope "
            "looks like. The angle cannot tell them apart; that is the judgement "
            "this axis exists to record."
        )
        caveat.setWordWrap(True)
        caveat.setStyleSheet("color: #6a6a6a; font-size: 11px;")

        looking = QtWidgets.QGroupBox("The hillside beside that stretch")
        look_laid = QtWidgets.QVBoxLayout(looking)
        look_laid.addWidget(self.ground)
        look_laid.addWidget(self.angles)
        look_laid.addWidget(caveat)

        # -- and the declaration ----------------------------------------------

        self.value = QtWidgets.QComboBox()
        self.value.addItems(list(module().EXPOSURE))
        self.value.setToolTip(
            "exposed: the surface crops out -- the one value that licenses a "
            "plane fitted to the facet.\n"
            "covered: it exists and is hidden under soil, scree or vegetation.\n"
            "concealed: buried under younger units.\n"
            "unknown: nobody has looked, which is not the same as covered."
        )
        self.value.currentIndexChanged.connect(self._tell)

        self.why = QtWidgets.QLineEdit()
        self.why.setPlaceholderText("what was seen, and where")
        self.why.setToolTip(
            "Goes into the line as reason=, word for word. The value says what, "
            "and this says why -- which on this axis is the whole of the "
            "evidence, the source having nothing to say about it at all."
        )
        self.why.textChanged.connect(self._tell)

        self.declare_button = QtWidgets.QPushButton("Declare this stretch")
        self.declare_button.setEnabled(False)
        self.declare_button.clicked.connect(self.declare)

        saying = QtWidgets.QGroupBox("Say it")
        say_laid = QtWidgets.QVBoxLayout(saying)
        valued = QtWidgets.QHBoxLayout()
        valued.addWidget(QtWidgets.QLabel("exposure"))
        valued.addWidget(self.value)
        valued.addWidget(self.why, stretch=1)
        say_laid.addLayout(valued)
        say_laid.addWidget(self.declare_button)

        self.undo_button = QtWidgets.QPushButton("Undo")
        self.undo_button.setEnabled(False)
        self.undo_button.setToolTip(
            "Put the block back as it was before the last press that wrote in "
            "it. Nothing reaches the file until Save."
        )
        self.undo_button.clicked.connect(self.undo_last)

        self.step = QtWidgets.QLabel()
        self.step.setWordWrap(True)
        self.step.setStyleSheet("color: #6a6a6a; font-size: 11px;")

        back = QtWidgets.QHBoxLayout()
        back.addWidget(self.step, stretch=1)
        back.addWidget(self.undo_button)

        laid = QtWidgets.QVBoxLayout(self)
        laid.addWidget(self.about)
        laid.addWidget(self.rule)
        laid.addWidget(self.table, stretch=1)
        laid.addWidget(stretching)
        laid.addWidget(looking)
        laid.addWidget(saying)
        laid.addLayout(back)

        self.retarget()

    # -- what the file says on this axis -----------------------------------

    def _structure(self):
        if self.panel.index is None:
            return None

        return self.panel.document.dataset.structures[self.panel.index]

    def retarget(self):
        """Points at whatever the panel has open, and reads the axis again."""

        structure = self._structure()

        self._in_file = []
        self.table.clearContents()
        self.table.setRowCount(0)

        if structure is None:
            self.about.setText("Nothing selected")
            self._forget_stretch()

            return

        gstruct = module()

        self.about.setText(
            f"{structure.ident} -- {gstruct.path_length(structure.path):.0f} m"
        )

        self._in_file = [
            row
            for row in rows_of(
                self.panel.document.text_of(self.panel.index), structure.path
            )
            if row.word == "span" and row.sort == EXPOSURE_AXIS
        ]

        self.table.setRowCount(len(self._in_file))

        for row in range(len(self._in_file)):
            self._write_row(row)

        self.table.resizeColumnsToContents()
        self._forget_stretch()

    def _forget_stretch(self):
        """The picked ends dropped, which a change of trace has to do.

        Progressives on one trace are a different stretch on the next, and a
        stretch kept across a selection would be a claim about ground nobody
        picked -- silently, since the numbers would still read as metres.

        `the whole trace` is left alone, and the difference is what each of them
        is about: two progressives are about *that* trace, and the box is about
        whichever one is open. Working down a list of faults declaring the whole
        of each is a real session, and clearing the box on every row would be
        this window forgetting the one thing that was not picked anywhere.
        """

        self._ends = None
        self._first = None
        self._hillside = None
        self._hillside_for = None
        self.pick_ends.setChecked(False)
        self.showing.emit(None)
        self._tell()

    def _in_force(self, at):
        """
        How many metres of a span's own stretch are still answering, as
        `(in force, its own length)`, or None.

        `covered_metres` with the list turned round, which is that function's
        own note: a span is shadowed by the spans written *after* it, a fit by
        the ones written before. Taken by index and not by row, because two
        identical lines are a thing these files hold and `list.index` would
        answer about the first of them.
        """

        row = self._in_file[at]

        if row.ends is None:
            return None

        shadowed = covered_metres(
            row.ends, [one.ends for one in self._in_file[at + 1:]]
        )

        if shadowed is None:
            return None

        held = row.ends[1] - row.ends[0]

        return max(0.0, held - shadowed), held

    def _ends_said(self, claim):
        """
        The two ends as metres, with `*` marked where the file wrote one.

        The number is where the end resolves to and the star is what it was
        written as, and both are needed: `*` resolves to the path's end *now*,
        so a column of numbers alone cannot tell a claim that follows the trace
        from one pinned to where the trace happens to stop today.
        """

        if claim.ends is None:
            return "", ""

        written = claim.line.split()[ENDS_AT]

        return tuple(
            f"{metres:.0f} m"
            + (" *" if len(written) > n and written[n] == "*" else "")
            for n, metres in enumerate(claim.ends)
        )

    def _write_row(self, row):
        claim = self._in_file[row]
        attrs = claim.attrs or {}
        force = self._in_force(row)

        if force is None:
            said = ""
        elif force[1] <= 0.0:
            # A pair written the wrong way round: it parses, it sits in the file
            # looking like a decision, and `covers` is false everywhere on it.
            said = "over no ground"
        elif force[0] >= force[1] - FULLY_M:
            said = "all of it"
        elif force[0] <= FULLY_M:
            said = "none of it"
        else:
            said = f"{force[0]:.0f} of {force[1]:.0f} m"

        near, far = self._ends_said(claim)

        cells = (
            near,
            far,
            claim.value or "",
            said,
            attrs.get("reason", ""),
        )

        for column, text in enumerate(cells):
            item = QtWidgets.QTableWidgetItem(text)

            if column == 0:
                item.setToolTip(claim.line.strip())

            if column == 2 and claim.value == LICENSING:
                # The one value with a consequence beyond being read, so it is
                # the one value the table marks. Not a judgement about the line:
                # a tint on the word `exposed` says where the licence is.
                item.setBackground(QtGui.QColor(EXPOSED_TINT))

            self.table.setItem(row, column, item)

    def _picked(self):
        """A row read back as the stretch it claims, lit on the map."""

        picked = self.table.selectionModel()
        rows = picked.selectedRows() if picked is not None else []

        if not rows or rows[0].row() >= len(self._in_file):
            return

        claim = self._in_file[rows[0].row()]

        self.showing.emit(claim.ends)

    # -- the stretch -------------------------------------------------------

    def wanting_ends(self):
        """Whether the next shift-click on the map belongs to this window."""

        return self.pick_ends.isChecked()

    def half_picked(self):
        """Whether one end is in and the other is still being waited for."""

        return self._first is not None

    def _wanting(self, on):
        if on:
            # A fresh pair, because that is what the button says: pressing it
            # with one end already taken and keeping that end would make the
            # next click finish a stretch somebody had stopped picking.
            self._first = None
            self._ends = None
            self.whole.setChecked(False)
            self.showing.emit(None)

        self.ends_wanted.emit(bool(on))
        self._tell()

    def _whole_changed(self, on):
        if on:
            self.pick_ends.setChecked(False)
            self._first = None
            self._ends = None

        self._hillside = None
        self._hillside_for = None
        self.showing.emit(self._stretch_now())
        self._tell()

    def took_end(self, s):
        """One end of the stretch, as a progressive worked out on the map."""

        structure = self._structure()

        if structure is None:
            return

        if self._first is None:
            self._first = float(s)
            self._tell()

            return

        # Ordered here and not left to the writer, which is the one thing a click
        # can settle that reading the line cannot: `covers` is `s0 <= s <= s1`, so
        # a pair the wrong way round parses and holds over nothing. Two clicks
        # have no order to lose -- the first is wherever the hand started.
        self._ends = tuple(sorted((self._first, float(s))))
        self._first = None
        self._hillside = None
        self._hillside_for = None
        self.pick_ends.setChecked(False)
        self.showing.emit(self._ends)
        self._tell()

    def _stretch_now(self):
        """The stretch the next press would claim, as `(s0, s1)`, or None."""

        structure = self._structure()

        if structure is None:
            return None

        if self.whole.isChecked():
            return 0.0, module().path_length(structure.path)

        return self._ends

    def _placed_said(self):
        """Where the next press would claim, in metres and in the file's words."""

        structure = self._structure()

        if structure is None:
            return ""

        if self.whole.isChecked():
            return (
                f"the whole of {structure.ident}, written `* *` -- the two ends "
                f"of the path and not coordinates at them"
            )

        if self._first is not None:
            return f"one end at {self._first:.0f} m; shift-click the other"

        if self._ends is None:
            return ""

        s0, s1 = self._ends

        return (
            f"{s0:.0f} to {s1:.0f} m -- {s1 - s0:.0f} m of trace, anchored at "
            f"both ends"
        )

    # -- the ground there --------------------------------------------------

    def _read_hillside(self):
        """
        The plane of the ground beside the picked stretch, read once per stretch.

        Cached against the stretch rather than recomputed on every keystroke in
        the reason box: it is a raster read over the corridor's bounds, and the
        box is typed in a word at a time.
        """

        stretch_at = self._stretch_now()
        structure = self._structure()

        if stretch_at is None or structure is None:
            return None

        for_now = (self.panel.index, stretch_at)

        if for_now == self._hillside_for:
            return self._hillside

        self._hillside_for = for_now
        self._hillside = None

        if self.panel.dem is None or self.panel.dem_said:
            return None

        QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor)
        )
        try:
            self._hillside = hillside_on(
                stretch(structure.path, *stretch_at),
                self.panel.dem,
                convergence=self.panel.convergence,
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        return self._hillside

    def _ground_said(self):
        """The hillside in words, or why there is none to say anything about."""

        if self._stretch_now() is None:
            return ""

        if self.panel.dem is None or self.panel.dem_said:
            return (
                "No DEM here, so nothing about the ground. The declaration does "
                "not need one: this is evidence, not a condition."
            )

        got = self._read_hillside()

        if got is None:
            return (
                "No DEM under this stretch, or not enough of one to fit a plane "
                "to -- the corridor fell on nodata or off the raster."
            )

        return (
            f"the ground dips {got.dip_dir:.0f}/{got.dip:.0f}"
            + (
                f" ({got.converg:+.2f} of convergence taken off)"
                if got.north == "true" else " from grid north"
            )
            + f", fitted to {got.n} cells "
            + " and ".join(
                f"{offset:.0f}"
                for offset in sorted({abs(one) for one in got.across})
            )
            + " m either side of the trace.\n"
            f"It misses them by {got.rms:.1f} m, "
            + self._residual_said(got)
            + f", over {got.relief:.0f} m of relief."
        )

    @staticmethod
    def _residual_said(got):
        """
        What the residual is large *against*, which is the only way to read it.

        The ratio and not the two numbers side by side, because the division is
        the sentence: over the 27 traces of `merid_faults` the whole-trace
        corridor misses its own best plane by 7 to 210 times what the sampling
        costs, so every `drape=` in that file is an angle to a plane the ground
        does not hold. Which is a thing to see while picking a stretch, and the
        stretch is the one variable that moves it.
        """

        floor = got.sampling_rms

        if floor <= 0.05:
            # A slope near flat, where the quotient is meaningless rather than
            # large: the floor is what half a cell of plan error does to height,
            # and on level ground it does nothing.
            return (
                f"against the {floor:.2f} m that reading cells at their centres "
                f"costs on ground this flat"
            )

        ratio = got.rms / floor

        # A decimal below ten and none above it. `423.8x` is a tenth of a
        # multiple of a tenth of a metre, which is three digits of nothing, and
        # `1x` is the one place the decimal carries the answer -- a residual at
        # the floor is a hillside that is a plane.
        return (
            f"which is {ratio:.1f}x" if ratio < 10.0 else f"which is {ratio:.0f}x"
        ) + (
            f" the {floor:.1f} m that reading cells at their centres costs on "
            f"a slope this steep"
        )

    def _angles_said(self):
        """
        The angle from the hillside to each plane the stretch already carries.

        Named one at a time and not averaged, which is the lesson of S22 and S25:
        two readings 30 m apart differ by 14 degrees, so a mean of the planes
        along a stretch is a number no outcrop has, and an angle to that mean
        would be an angle to nothing.
        """

        got = self._read_hillside()
        stretch_at = self._stretch_now()
        structure = self._structure()

        if got is None or stretch_at is None or structure is None:
            return ""

        s0, s1 = stretch_at
        told = []

        for attitude in structure.attitudes:
            if attitude.s is None or attitude.plane is None:
                continue

            if s0 <= attitude.s <= s1:
                named = attitude.attrs.get("station") or "a reading"
                told.append(
                    f"{named} {attitude.plane} is "
                    f"{between(got.plane, (attitude.plane.dip_dir, attitude.plane.dip)):.0f}"
                    f"\N{DEGREE SIGN} off it"
                )

        for fit in structure.fits:
            if fit.plane is None:
                continue

            at0 = 0.0 if fit.s0 is None else fit.s0
            at1 = structure.length if fit.s1 is None else fit.s1

            if at1 < s0 or at0 > s1:
                continue

            told.append(
                f"the {fit.attrs.get('from', 'fit')} fit {fit.plane} is "
                f"{between(got.plane, (fit.plane.dip_dir, fit.plane.dip)):.0f}"
                f"\N{DEGREE SIGN} off it"
            )

        if not told:
            return "Nothing is claimed over this stretch to compare it with."

        return "; ".join(told) + "."

    # -- the press ---------------------------------------------------------

    def _tell(self):
        """What is missing before a press can happen, and the press's state."""

        stretch_at = self._stretch_now()
        why = self.why.text().strip()
        open_here = self.panel.index is not None

        self.pick_ends.setEnabled(open_here)
        self.whole.setEnabled(open_here)
        self.declare_button.setEnabled(
            open_here and stretch_at is not None and bool(why)
        )
        self.undo_button.setEnabled(self.panel.may_undo())
        self.where.setText(self._placed_said())
        self.ground.setText(self._ground_said())
        self.angles.setText(self._angles_said())
        self.step.setText(self._step_said(stretch_at, why, open_here))

    def _step_said(self, stretch_at, why, open_here):
        """One line about whichever half of the gesture is still missing."""

        if not open_here:
            return ""

        if self._first is not None:
            return "Shift-click the other end of the stretch."

        if self.wanting_ends():
            return "Shift-click the two ends of the stretch on the map."

        if stretch_at is None:
            return (
                "Pick the stretch this says something about -- two ends on the "
                "map, or the whole trace."
            )

        if not why:
            return (
                "Say what was seen. It goes in the line as reason=, and on this "
                "axis it is the only evidence there will ever be: the source has "
                "none."
            )

        value = self.value.currentText()

        if value == LICENSING:
            return (
                "`Declare this stretch` writes it, and a plane fitted to the "
                "facet becomes possible over that ground."
            )

        return "`Declare this stretch` writes it below the lines already there."

    def declare(self):
        """The picked stretch declared, as one `span` line in the block."""

        stretch_at = self._stretch_now()
        structure = self._structure()
        why = self.why.text().strip()

        if stretch_at is None or structure is None or not why:
            return False

        if self.whole.isChecked():
            # `*`, and not the coordinates of the path's two ends: the token goes
            # on meaning *the end of the path*, so the claim follows a trace that
            # is redigitised instead of stopping where it used to stop.
            start, end = None, None
        else:
            start, end = (
                point_on(structure.path, stretch_at[0]),
                point_on(structure.path, stretch_at[1]),
            )

        line = span_line(
            EXPOSURE_AXIS,
            self.value.currentText(),
            start,
            end,
            {"src": "gsurf", "reason": why},
        )
        refused = self.panel.insert_claim(line)

        if refused is not None:
            self.said.emit(refused)

            return False

        self.said.emit(
            f"declared {line.strip()} -- Undo takes it out, Save writes the file"
        )

        self.why.clear()
        self.showing.emit(None)
        self.wrote.emit()
        self.retarget()

        return True

    def undo_last(self):
        """The block before the last press that wrote in it, put back."""

        if not self.panel.undo_applied():
            return False

        self.said.emit("the block is back as it was before the last press")
        self.wrote.emit()
        self.retarget()

        return True


class FacetHere(QtWidgets.QWidget):
    """
    The attitude of the surface itself, where somebody has said it crops out.

    `ExposureHere` writes the licence and this spends it. FORMAT.md's rule is
    that a plane fitted to a facet runs only under `exposure=exposed`, and the
    reason is not procedural: the region is grown by taking cells whose own slope
    matches a measured plane, so on *any* hillside near *any* fault it will find
    something. What makes the answer a measurement of the fault rather than of
    the hill is a person having stood there and said the fault is what crops out.

    **The radius is part of the answer, not a frame around it.** This is the one
    thing the reference implementation hid, and the measurement is plain:
    `facets.RADIUS` is 500 m, and at 500 m five of the eight facets of this AOI
    are still growing when the window stops. Opened to 1000 m, S26 goes from 47
    to 95 hectares with its plane steady at 142/30 -- a bigger measurement of the
    same surface -- while S19 moves from 121/30 to 110/30, eleven degrees, toward
    the 107/35 its own compass reads. One of those is a surface being measured
    further and the other is a surface that was never one plane, and the only way
    to tell is to ask at several radii and watch. So `Grow it further` sweeps the
    ladder and the table reports **how far the plane moved**, with the rim flag
    beside it, because an area that reaches the edge of the window is a lower
    bound wearing a number.

    **The stretch comes from the licence and not from the region.** A `fit`
    claims an interval of trace and a facet is a region; the obvious repair to
    `export_geology.py`'s `* *` was to anchor the fit over the ground the facet
    covers, and the data refuses it. These facets sit a median of 47 to 296 m
    from their own trace and out to 714 m, because an exposed dip slope runs away
    down the dip and the trace is its up-dip edge: strict containment gives 20 m
    of a 27 hectare surface, and nothing at all on two of the eight. So the
    interval written is the `exposure=exposed` span in force at the seed --
    somebody's own statement of where this contact crops out -- and `off=` goes
    on the line beside it, so the next reader can see how far from the trace the
    surface measured actually lies without recomputing anything.

    **And the angle to the seed is not evidence.** It cannot be: the region is
    the cells within `tol` of that plane, so the result is inside `tol` of it by
    construction. `export_geology.py` wrote it as `vs_field` and read it as
    corroboration, which is the mistake this window is built not to repeat -- the
    seed's row in the angles table says `grew it`, and the readings that did not
    grow it are the ones worth reading.
    """

    showing = QtCore.pyqtSignal(object)
    said = QtCore.pyqtSignal(str)
    wrote = QtCore.pyqtSignal()

    # The facet to draw on the map, or None. A region and not a stretch, which is
    # why it does not go to the band's sink: see `EditorWindow._show_facet`.
    drawing = QtCore.pyqtSignal(object)

    def __init__(self, panel, parent=None):
        super().__init__(parent)

        self.panel = panel

        # The readings of the open block, as `curation.Row`, in file order.
        self._seeds = []

        # The facet last grown, and which row and radius it came from.
        self._facet = None
        self._grown_from = None

        # And how far its cells lie from the trace, measured once with it. Kept
        # rather than recomputed in `_measured_said`, which `_tell` calls on
        # every keystroke and every change of either dial: the quantity is a
        # distance from 600 cells to every segment of a path, and it cannot
        # change while the facet does not.
        self._offsets = None

        self.about = QtWidgets.QLabel()
        self.about.setWordWrap(True)
        self.about.setStyleSheet("font-weight: bold;")

        self.licence = QtWidgets.QLabel()
        self.licence.setWordWrap(True)
        self.licence.setStyleSheet("color: #6a6a6a; font-size: 11px;")

        self.seeds = QtWidgets.QTableWidget(0, len(SEED_COLUMNS))
        self.seeds.setHorizontalHeaderLabels(SEED_COLUMNS)
        self.seeds.verticalHeader().setVisible(False)
        self.seeds.setSortingEnabled(False)
        self.seeds.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.seeds.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.seeds.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.seeds.itemSelectionChanged.connect(self._picked)

        # -- the two numbers the method is made of -----------------------------

        self.radius = QtWidgets.QComboBox()

        for metres in FACET_RADII:
            self.radius.addItem(f"{metres:.0f} m", metres)

        self.radius.setToolTip(
            "How far from the site to look. Not a frame around the answer: at "
            "500 m five of the eight facets of this area are still growing when "
            "the window stops, and one of them turns eleven degrees when it is "
            "opened to 1000. `Grow it further` is how to tell a surface being "
            "measured further from a surface that was never one plane."
        )
        self.radius.currentIndexChanged.connect(self._tell)

        self.tolerance = QtWidgets.QSpinBox()
        self.tolerance.setRange(2, 40)
        self.tolerance.setValue(int(facets.TOL))
        self.tolerance.setSuffix("\N{DEGREE SIGN}")
        self.tolerance.setToolTip(
            "How far a cell's own slope may sit from the measured plane and "
            "still count as the same surface. This is the assertion the method "
            "is made of, so it is here to be moved -- and it is also why the "
            "angle between the result and the seed says nothing: the region is "
            "the cells within this many degrees of that plane."
        )
        self.tolerance.valueChanged.connect(self._tell)

        self.grow_button = QtWidgets.QPushButton("Grow the facet")
        self.grow_button.setEnabled(False)
        self.grow_button.clicked.connect(self.grow)

        self.sweep_button = QtWidgets.QPushButton("Grow it further")
        self.sweep_button.setEnabled(False)
        self.sweep_button.setToolTip(
            "The same site at every radius on the ladder, so that how much the "
            "plane moves is on screen beside how much bigger the surface got."
        )
        self.sweep_button.clicked.connect(self.sweep)

        dialling = QtWidgets.QHBoxLayout()
        dialling.addWidget(QtWidgets.QLabel("look out to"))
        dialling.addWidget(self.radius)
        dialling.addWidget(QtWidgets.QLabel("within"))
        dialling.addWidget(self.tolerance)
        dialling.addStretch(1)
        dialling.addWidget(self.grow_button)
        dialling.addWidget(self.sweep_button)

        growing = QtWidgets.QGroupBox("Grow it")
        grow_laid = QtWidgets.QVBoxLayout(growing)
        grow_laid.addLayout(dialling)

        self.measured = QtWidgets.QLabel()
        self.measured.setWordWrap(True)
        self.measured.setStyleSheet("font-size: 11px;")
        grow_laid.addWidget(self.measured)

        self.sweep_table = QtWidgets.QTableWidget(0, len(SWEEP_COLUMNS))
        self.sweep_table.setHorizontalHeaderLabels(SWEEP_COLUMNS)
        self.sweep_table.verticalHeader().setVisible(False)
        self.sweep_table.setSortingEnabled(False)
        self.sweep_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.sweep_table.setMaximumHeight(SWEEP_TABLE_PX)
        grow_laid.addWidget(self.sweep_table)

        # -- what else was measured along the same ground ----------------------

        self.angles = QtWidgets.QLabel()
        self.angles.setWordWrap(True)
        self.angles.setStyleSheet("font-size: 11px; color: #333333;")

        against = QtWidgets.QGroupBox("Against what else was read there")
        against_laid = QtWidgets.QVBoxLayout(against)
        against_laid.addWidget(self.angles)

        # -- and the line ------------------------------------------------------

        self.keep_button = QtWidgets.QPushButton("Keep this fit")
        self.keep_button.setEnabled(False)
        self.keep_button.clicked.connect(self.keep)

        self.undo_button = QtWidgets.QPushButton("Undo")
        self.undo_button.setEnabled(False)
        self.undo_button.clicked.connect(self.undo_last)

        self.step = QtWidgets.QLabel()
        self.step.setWordWrap(True)
        self.step.setStyleSheet("color: #6a6a6a; font-size: 11px;")

        keeping = QtWidgets.QHBoxLayout()
        keeping.addWidget(self.keep_button)
        keeping.addStretch(1)
        keeping.addWidget(self.undo_button)

        laid = QtWidgets.QVBoxLayout(self)
        laid.addWidget(self.about)
        laid.addWidget(self.licence)
        laid.addWidget(self.seeds, stretch=1)
        laid.addWidget(growing)
        laid.addWidget(against)
        laid.addWidget(self.step)
        laid.addLayout(keeping)

        self.retarget()

    # -- what is here to grow from -----------------------------------------

    def _structure(self):
        if self.panel.index is None:
            return None

        return self.panel.document.dataset.structures[self.panel.index]

    def refusal(self):
        """Why no facet can be grown in this session at all, or None."""

        if self.panel.dem is None:
            return (
                "No DEM in this session. A facet is a region of topography, so "
                "this is the one thing here that cannot be done without one."
            )

        return self.panel.dem_said

    def retarget(self):
        """Points at whatever the panel has open, and reads its readings again."""

        structure = self._structure()

        self._seeds = []
        self._forget()
        self.seeds.clearContents()
        self.seeds.setRowCount(0)

        if structure is None:
            self.about.setText("Nothing selected")
            self.licence.setText("")
            self._tell()

            return

        gstruct = module()

        self.about.setText(
            f"{structure.ident} -- {gstruct.path_length(structure.path):.0f} m"
        )

        self._seeds = [
            row
            for row in readings_in(
                self.panel.document.text_of(self.panel.index), structure.path
            )
            if row.plane is not None and row.place is not None
        ]

        self.seeds.setRowCount(len(self._seeds))

        for row in range(len(self._seeds)):
            self._write_seed(row)

        self.seeds.resizeColumnsToContents()
        self.licence.setText(self._licence_said())
        self._tell()

    def _licensed(self, row):
        """The `exposure=exposed` span in force where a reading sits, or None.

        `span_at` and not a search of the lines, because the question is which
        span *answers* there: the axis allows any number of overlapping spans and
        the last one covering wins, so a file can hold `exposed` over a stretch
        and `covered` written under it over part of that stretch. The licence is
        what holds, and that is one call.
        """

        structure = self._structure()

        if structure is None or row.place is None:
            return None

        span = structure.span_at(EXPOSURE_AXIS, row.place)

        return span if span is not None and span.value == LICENSING else None

    def _licence_said(self):
        """What this trace has been declared to be, in one line."""

        structure = self._structure()

        if structure is None:
            return ""

        exposed = [
            span
            for span in structure.spans
            if span.axis == EXPOSURE_AXIS and span.value == LICENSING
        ]

        if not exposed:
            return (
                "Nothing on this trace is declared exposed, so there is nothing "
                "to grow: the calculation is licensed by an "
                "`exposure=exposed` span, and that is written in `Exposure "
                "along this trace` (Ctrl+E)."
            )

        metres = sum(
            (span.s1 - span.s0)
            for span in exposed
            if span.s0 is not None and span.s1 is not None and span.s1 > span.s0
        )

        return (
            f"{len(exposed)} stretch{'' if len(exposed) == 1 else 'es'} declared "
            f"exposed, {metres:.0f} m of trace in all. A fit grown here is "
            f"written over the one in force at the site, which is the claim "
            f"being capped by its own licence."
        )

    def _write_seed(self, row):
        claim = self._seeds[row]
        span = self._licensed(claim)
        attrs = claim.attrs or {}
        named = attrs.get("station")

        cells = (
            f"{claim.place:.0f} m" + (f"  ({named})" if named else ""),
            f"{claim.plane[0]:.0f}/{claim.plane[1]:.0f}",
            (
                f"{span.s0:.0f} to {span.s1:.0f} m"
                if span is not None and span.s0 is not None
                else "not declared"
            ),
            "",
        )

        for column, text in enumerate(cells):
            item = QtWidgets.QTableWidgetItem(text)

            if column == 0:
                item.setToolTip(claim.line.strip())

            if column == 2 and span is not None:
                item.setBackground(QtGui.QColor(EXPOSED_TINT))

            self.seeds.setItem(row, column, item)

    def _picked_row(self):
        picked = self.seeds.selectionModel()
        rows = picked.selectedRows() if picked is not None else []

        if not rows:
            return None

        at = rows[0].row()

        return at if at < len(self._seeds) else None

    def _picked(self):
        self._forget()
        self._tell()

    def _forget(self):
        """The grown facet dropped, which any change of question has to do."""

        self._facet = None
        self._grown_from = None
        self._offsets = None
        self.sweep_table.clearContents()
        self.sweep_table.setRowCount(0)
        self.drawing.emit(None)
        self.showing.emit(None)

    # -- growing it --------------------------------------------------------

    def _grow_at(self, radius):
        """The facet at one radius for the picked row, or None. Nothing shown."""

        at = self._picked_row()
        structure = self._structure()

        if at is None or structure is None or self.refusal():
            return None

        claim = self._seeds[at]
        anchor = anchor_of(claim.line)

        if anchor is None:
            return None

        return facets.facet_on(
            anchor,
            claim.plane,
            self.panel.dem,
            convergence=self.panel.convergence,
            radius=float(radius),
            tol=float(self.tolerance.value()),
        )

    def _keep_grown(self, at, facet, radius):
        """One grown facet taken as the window's answer: measured, drawn, shown."""

        structure = self._structure()

        self._facet = facet
        self._grown_from = (at, radius, self.tolerance.value())
        self._offsets = (
            facets.offsets_on(facet, structure.path)
            if facet is not None and structure is not None
            else None
        )

        self.drawing.emit(facet)

        if facet is None:
            return

        span = self._licensed(self._seeds[at])

        if span is not None and span.s0 is not None:
            self.showing.emit((span.s0, span.s1))

        self._fill_seed_grown(at)

    def grow(self):
        """The facet at the chosen radius, measured and drawn and not written."""

        at = self._picked_row()

        if at is None:
            return False

        radius = self.radius.currentData()

        QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor)
        )
        try:
            self._keep_grown(at, self._grow_at(radius), radius)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        if self._facet is None:
            self.said.emit(self._nothing_grew())
            self._tell()

            return False

        self.said.emit(
            f"grown {self._facet.area_ha:.0f} ha at "
            f"{self._facet.dip_dir:.0f}/{self._facet.dip:.0f}"
            + (
                " -- still growing where the window stops, so this is a lower "
                "bound: try a longer radius"
                if self._facet.at_the_rim else ""
            )
        )
        self._tell()

        return True

    def _nothing_grew(self):
        """
        Why a press came back with nothing, as one of the things it can be.

        Said rather than left as an empty box, because the reasons are different
        facts about the ground: a site whose own cell and whose neighbours all
        fail the tolerance is a surface that does not crop out as morphology at
        all, and no method reading a DTM will find its attitude. That is an
        answer, and `facets.facet_on` returning None is how it arrives.
        """

        return (
            f"nothing grew from here within "
            f"{self.tolerance.value()}\N{DEGREE SIGN}: either the site's own "
            f"cells do not match the plane that was measured, or what does "
            f"match is under "
            f"{facets.MIN_AREA_HA:.0f} ha. The surface is not morphology here, "
            f"and a DTM cannot give its attitude"
        )

    def sweep(self):
        """The same site at every radius, with how far the plane moved."""

        at = self._picked_row()

        if at is None:
            return False

        QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor)
        )
        try:
            grown = [(metres, self._grow_at(metres)) for metres in FACET_RADII]
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        rows = [one for one in grown if one[1] is not None]

        self.sweep_table.setRowCount(len(rows))

        first = rows[0][1] if rows else None
        chosen = self.radius.currentData()

        for row, (metres, facet) in enumerate(rows):
            moved = between(first.plane, facet.plane) if first is not None else 0.0

            cells = (
                f"{metres:.0f} m",
                f"{facet.area_ha:.0f} ha",
                f"{facet.dip_dir:.0f}/{facet.dip:.0f}",
                f"{moved:.0f}\N{DEGREE SIGN}",
                "at the rim" if facet.at_the_rim else "",
            )

            for column, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)

                if column == 4 and facet.at_the_rim:
                    item.setForeground(QtGui.QColor("#b2182b"))

                self.sweep_table.setItem(row, column, item)

        self.sweep_table.resizeColumnsToContents()

        if not rows:
            self.said.emit(self._nothing_grew())

            return False

        # The chosen radius's own facet is kept out of the sweep rather than
        # recomputed, so that the press leaves the window pointing at a
        # measurement and not at a table of them -- and so that the sentence
        # below is the last thing said, where a second `grow()` would have
        # overwritten it with its own.
        at_chosen = next(
            (facet for metres, facet in rows if metres == chosen), None
        )

        if at_chosen is not None:
            self._keep_grown(at, at_chosen, chosen)

        self._tell()

        widest = max(
            between(rows[0][1].plane, facet.plane) for _, facet in rows
        )

        # The sentence is about the spread and not about the largest, because the
        # question the sweep answers is whether there is one plane here: a
        # surface measured further is the same answer over more ground, and a
        # surface that was never one plane is a different answer each time.
        self.said.emit(
            f"{len(rows)} radii: the plane moves "
            f"{widest:.0f}\N{DEGREE SIGN} across them"
            + (
                " -- one surface, measured further out"
                if widest < FACET_ONE_PLANE
                else " -- more than a measurement of one surface moves, so the "
                     "radius is choosing the answer"
            )
        )

        return True

    def _measured_said(self):
        """The facet in words: the plane, how big, how planar, and how far off."""

        refused = self.refusal()

        if refused:
            return refused

        got = self._facet

        if got is None:
            return ""

        away = self._offsets

        said = (
            f"{got.area_ha:.0f} ha over {got.n} cells, dipping "
            f"{got.dip_dir:.0f}/{got.dip:.0f}"
            + (
                f" ({got.converg:+.2f} of convergence taken off)"
                if got.north == "true" else " from grid north"
            )
            + f", {got.span:.0f} m across.\n"
            f"It misses its own best plane by {got.rms:.1f} m over those "
            f"{got.span:.0f} m -- {got.waviness:.1f}\N{DEGREE SIGN} of "
            f"waviness -- through {got.relief:.0f} m of relief. None of that is "
            f"sampling: a facet is fitted to the cells themselves."
        )

        if away is not None and len(away):
            said += (
                f"\nThe surface lies a median of {np.median(away):.0f} m from "
                f"the trace, out to {away.max():.0f} m -- which is what an "
                f"exposed dip slope does, running away down the dip from its "
                f"own trace."
            )

        if got.at_the_rim:
            said += (
                f"\nAnd it is still growing where the window stops, so "
                f"{got.area_ha:.0f} ha is a lower bound and not a measurement. "
                f"A longer radius is the next thing to try."
            )

        return said

    def _angles_said(self):
        """
        Every reading on the licensed stretch, with its angle, the seed marked.

        Marked and not excluded, which is the point: the seed has to be visible
        *as* the seed, because its angle is bounded by the tolerance by
        construction and a reader who does not know which row grew the region
        would read the smallest number in the table as the best agreement.
        """

        got = self._facet
        at = self._picked_row()
        structure = self._structure()

        if got is None or at is None or structure is None:
            return ""

        seed = self._seeds[at]
        span = self._licensed(seed)
        told = []

        for attitude in structure.attitudes:
            if attitude.s is None or attitude.plane is None:
                continue

            if span is not None and span.s0 is not None:
                if not (span.s0 <= attitude.s <= span.s1):
                    continue

            named = attitude.attrs.get("station") or "a reading"
            apart = between(
                got.plane, (attitude.plane.dip_dir, attitude.plane.dip)
            )
            same = (
                seed.place is not None
                and abs(attitude.s - seed.place) < SAME_OUTCROP_M
            )

            told.append(
                f"{named} {attitude.plane} is {apart:.0f}\N{DEGREE SIGN} off it"
                + ("  <- grew it, so this angle is inside the tolerance by "
                   "construction" if same else "")
            )

        if not told:
            return "Nothing else is measured on the licensed stretch."

        return "\n".join(told)

    # -- the line ----------------------------------------------------------

    def _tell(self):
        """What is missing before each press can happen."""

        at = self._picked_row()
        refused = bool(self.refusal())
        stale = self._grown_from != (
            at, self.radius.currentData(), self.tolerance.value()
        )

        self.radius.setEnabled(not refused)
        self.tolerance.setEnabled(not refused)
        self.grow_button.setEnabled(at is not None and not refused)
        self.sweep_button.setEnabled(at is not None and not refused)
        self.keep_button.setEnabled(
            self._facet is not None and not stale and self._licence_for() is not None
        )
        self.undo_button.setEnabled(self.panel.may_undo())
        self.measured.setText(self._measured_said())
        self.angles.setText(self._angles_said())
        self.step.setText(self._step_said(at, stale))

    def _licence_for(self):
        """The span a kept fit would be written over, or None if there is none."""

        at = self._picked_row()

        return None if at is None else self._licensed(self._seeds[at])

    def _step_said(self, at, stale):
        """One line about whichever half of the gesture is still missing."""

        refused = self.refusal()

        if refused:
            return refused

        if self.panel.index is None:
            return ""

        if not self._seeds:
            return (
                "Nothing is measured along this trace, and a facet is grown "
                "from a measurement: the plane says which cells belong to the "
                "same surface."
            )

        if at is None:
            return "Pick the reading to grow from."

        if self._licensed(self._seeds[at]) is None:
            return (
                "That reading is not on ground declared exposed. Grow it to "
                "look, by all means -- but a fit cannot be written, because the "
                "licence is the stretch it would be written over (Ctrl+E)."
            )

        if self._facet is None or stale:
            return "`Grow the facet`, and look at what came out."

        span = self._licensed(self._seeds[at])

        return (
            f"`Keep this fit` writes it over the declared stretch, "
            f"{span.s0:.0f} to {span.s1:.0f} m"
            if span.s0 is not None
            else "`Keep this fit` writes it over the whole trace, which is what "
                 "the licence says"
        )

    def _fill_seed_grown(self, at):
        """The last column of the row that was grown, so the table remembers."""

        got = self._facet

        if got is None or at >= self.seeds.rowCount():
            return

        self.seeds.setItem(at, 3, QtWidgets.QTableWidgetItem(
            f"{got.area_ha:.0f} ha {got.dip_dir:.0f}/{got.dip:.0f}"
            + (" (rim)" if got.at_the_rim else "")
        ))
        self.seeds.resizeColumnsToContents()

    def keep(self):
        """The grown facet as one anchored `fit` line in the block."""

        got = self._facet
        at = self._picked_row()
        span = self._licence_for()

        if got is None or at is None or span is None:
            return False

        away = self._offsets if self._offsets is not None else np.zeros(0)
        seed = self._seeds[at]
        attrs = {
            "from": facets.FROM_FACET,
            "src": "gsurf",
            # The licence written on the line, which FORMAT.md asks for: a fit
            # whose precondition is a curatorial act has to name the act.
            "licence": facets.LICENCE,
            "north": got.north,
            "dem": self.panel.dem.path.name,
            "seed": (seed.attrs or {}).get("station", "")
                    or f"{seed.place:.0f}m",
            "tol": f"{got.tol:.0f}",
            "radius": f"{got.radius:.0f}",
            "ncell": str(got.n),
            "area_ha": f"{got.area_ha:.1f}",
            # `across` and not the reference's `span`, which on a `fit` line
            # would sit beside the interval the fit claims and read as a second
            # statement about it. This is the region's own diagonal in plan,
            # which is a different quantity: on F0058 it is 987 m next to a
            # trace 981 m long, and the two numbers are not comparable.
            "across": f"{got.span:.0f}",
            "res": f"{got.rms:.1f}",
            "wavy": f"{got.waviness:.1f}",
            "relief": f"{got.relief:.0f}",
        }

        if got.north == "true":
            attrs["converg"] = f"{got.converg:+.2f}"

        if len(away):
            # How far the measured surface lies from the line the fit is written
            # on. Not a quality: a dip slope is *supposed* to run away down the
            # dip. It is the number that says what kind of claim this is.
            attrs["off"] = f"{np.median(away):.0f}"
            attrs["offmax"] = f"{away.max():.0f}"

        if got.at_the_rim:
            # Said on the line, because the window closes and the number does
            # not: an area measured against the edge of the window is a lower
            # bound, and a reader cannot tell that from the figure alone.
            attrs["rim"] = "reached"

        line = as_line(
            module().Fit(
                plane=module().Plane(got.dip_dir, got.dip),
                start=span.start,
                end=span.end,
                attrs=attrs,
            )
        )
        refused = self.panel.insert_claim(line)

        if refused is not None:
            self.said.emit(refused)

            return False

        self.said.emit(
            f"kept {line.strip()[:80]}... -- Undo takes it out, Save writes the "
            f"file"
        )
        self.wrote.emit()
        self.retarget()

        return True

    def undo_last(self):
        """The block before the last press that wrote in it, put back."""

        if not self.panel.undo_applied():
            return False

        self.said.emit("the block is back as it was before the last press")
        self.wrote.emit()
        self.retarget()

        return True


class FitFromDem(QtWidgets.QWidget):
    """
    Reading the topography along one trace: press, look at what came out, keep it.

    The window the fitting was asked for, and what it replaces is not a button
    but a habit. `fit_off_dem` wrote its answer into the box as lines, and the
    box is going: a curator reading `fit plane @583458.91,4439774.76
    @582408.83,4441315.77 118.4/42.1 from=trace-dem` is reading coordinates to
    find out which piece of fault a plane was claimed over, which is a thing
    nobody can do. Here the same answer is four numbers in a row, the two that
    matter are **progressives along the trace**, and pointing at the row lights
    the ground on the map.

    **What the file already claims is above what was just read**, and those are
    two tables because they are two different kinds of thing: one is in the file
    and the other is a list nobody has decided about yet. The order is the
    argument -- you see what is claimed along this fault before you add to it.

    That is not a convenience. `montealpi_01.gstruct` carries three fits over
    `2887.500..2937.503 m` of `L0071`, two of them byte-identical, because
    nothing on screen ever said the first one was there; and the defence against
    it today is that Keep spends its list, which stops the second press and not
    the second session. With the file's own fits on screen the duplicate is
    visible before it is made, and a candidate that an earlier line already
    covers arrives unticked -- `attitude_at` takes the first fit covering a
    progressive, so such a line would parse, apply, save, and never be asked
    anything.

    **Every fit, whatever made it.** A file's fits come from the sweep, from the
    plane steered against the topography, from a table, from a reach; a table
    showing only this window's own output would say *nothing is claimed here*
    about a trace that carries a fit off a table, which is the lie by omission
    worth avoiding. So the producer is a column, and this window is really about
    the planes claimed along one trace, with reading them off the DEM as the way
    to make new ones.

    **Everything is read, and the ticks decide what is kept.** The other
    arrangement -- choose a stretch first, then read only that -- was the obvious
    one and is worse, because it asks the question in the wrong order: which part
    of a fault the topography can answer for is the thing being found out, so
    fixing it in advance is guessing at the answer. Reading the whole trace costs
    about ten milliseconds and returns typically four to eight stretches, most of
    them somewhere a curator would not have thought to look.

    **One thing is set, and it is the window length.** It started as no control
    at all, on the argument that `fits.Sweep`'s four numbers are chosen together
    -- which is true of the step and the fallback and not of the length, and the
    file said so: `montealpi_01.gstruct` read through the sweep gives one trace
    in five something to keep, and the lengths that would have answered for the
    rest sit inside the sweep unreported. So `read over` is a choice, and the
    honest way to describe it is not "try until something comes out": a length
    is the distance over which this fault is claimed to hold a single
    orientation, it goes into the file as `window=`, and the reading says `the
    length you asked for` rather than pretending the trace picked it.

    Which also says how to pick one, because there is a cost and it is not
    subtle. A long window covers a bend wherever it is put, so it finds the bend
    and then cannot say where along the fault the answer held -- on a 2683 m
    trace with one bend, 250 m puts the fit on the bend and 2500 m puts it on
    everything. The length to ask for is the length the fault looks like it
    holds over, not the one that fills the table.

    **And where a reading gives nothing, the other lengths are named.** `nothing
    held` is a verdict about one window, and by itself it cannot be told apart
    from a trace that is simply straight. The ladder already knows the
    difference and used to throw it away.

    **The step, the fallback and the gate stay uncontrolled.** The first two
    belong to the ladder; the gate is measured off *every* path in the file on
    purpose, so that two verdicts in one file answer the same question, and a
    spin box for it would be a spin box for making them not. What the gate came
    to is shown, because `too straight to carry a plane` is a comparison and this
    is the number it is against.

    **Not modal**, which is the one thing about it that is structural. The trace
    is on the map and the map is the other window: a modal dialog would put the
    evidence behind the thing asking about it. It is also what lets the checks
    drive it, a modal box being something a headless run sits down in front of
    for ever.
    """

    # What to light on the map, as `(s0, s1)` along the selected trace, or None.
    # The map's own `_show_claimed`, which the caret used to feed and which is
    # about to have nothing else feeding it.
    showing = QtCore.pyqtSignal(object)

    # For the status bar, which is the map's and not this window's.
    said = QtCore.pyqtSignal(str)

    # How many were kept, for anything that wants to know without asking.
    kept = QtCore.pyqtSignal(int)

    def __init__(self, panel, parent=None):
        super().__init__(parent)

        self.panel = panel

        # The last `Read`, or None. Held rather than rebuilt from the table,
        # because what is kept is the *lines* and the table shows a rendering of
        # them: recovering a line from four formatted cells would be parsing our
        # own display, and the display rounds.
        self._read = None

        # The `fit` rows the document holds for the selected trace, in file
        # order, as `curation.Row`. Each knows the line it came from and where
        # that line sits in the block, which is what a splice needs -- so this is
        # also where removing one will be aimed from. Named for what it holds and
        # not for the table showing it, `self.carried` being that table.
        self._in_file = []

        # One band and two tables with an opinion about it. Picking a row in
        # either clears the other, and clearing a selection is itself a signal,
        # so the two would hand the band back and forth. See `_only_here`.
        self._picking = False

        # And the same again for the ticks: colouring a cell after it is in the
        # table is a change by `itemChanged`'s reckoning, so a fill would be read
        # as somebody ticking things.
        self._filling = False

        self.about = QtWidgets.QLabel()
        self.about.setWordWrap(True)
        self.about.setStyleSheet("font-weight: bold;")

        # What the file says along this trace, said in words as well as drawn in
        # rows, because the one thing the rows cannot show is the rule that makes
        # their order meaning.
        self.carries = QtWidgets.QLabel()
        self.carries.setWordWrap(True)
        self.carries.setStyleSheet("color: #6a6a6a; font-size: 11px;")

        self.carried = QtWidgets.QTableWidget(0, len(CARRIED_COLUMNS))
        self.carried.setHorizontalHeaderLabels(CARRIED_COLUMNS)
        self.carried.verticalHeader().setVisible(False)
        self.carried.setSortingEnabled(False)
        self.carried.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.carried.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.carried.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.carried.itemSelectionChanged.connect(self._carried_picked)

        # The two presses the file's own table affords, on a row under it rather
        # than in a menu behind a right-click: nothing else in this window is
        # hidden that way, and a delete nobody finds is a window that looks as
        # though it cannot delete. Both start dead -- one wants a row, the other
        # wants something to have happened.
        self.delete_button = QtWidgets.QPushButton("Delete this fit")
        self.delete_button.setEnabled(False)
        self.delete_button.clicked.connect(self.delete_carried)

        # Named for the gesture and not for the thing -- `Undo` and not `put the
        # line back` -- because what it undoes is either of the two presses in
        # this window that write: a Keep and a Delete. Beside the delete because
        # that is the press it will be wanted after, right-aligned because it is
        # about the window rather than about the table.
        self.undo_button = QtWidgets.QPushButton("Undo")
        self.undo_button.setEnabled(False)
        self.undo_button.setToolTip(
            "Put the block back as it was before the last press that wrote in "
            "it -- a Keep or a Delete. The box's own Ctrl+Z cannot: applying "
            "re-reads the block and the box's history goes with it. Nothing "
            "reaches the file until Save."
        )
        self.undo_button.clicked.connect(self.undo_last)

        self.removing = QtWidgets.QWidget()

        taking_out = QtWidgets.QHBoxLayout(self.removing)
        taking_out.setContentsMargins(0, 0, 0, 0)
        taking_out.addWidget(self.delete_button)
        taking_out.addStretch(1)
        taking_out.addWidget(self.undo_button)

        self.read_button = QtWidgets.QPushButton("Read the topography")

        # Kept, because the button's tooltip is either this or the refusal, and
        # `retarget` runs more than once: reading the refusal back off the widget
        # to decide whether to restore it would make the widget the place the
        # sentence lives, and then a session that refused once would go on
        # refusing in words after the reason had gone.
        self._how_to_read = (
            "Sweep a window along this trace and work out the plane over every "
            "stretch that turns enough to determine one. Nothing is written: "
            "what comes out is listed below, to be looked at and ticked."
        )
        self.read_button.setToolTip(self._how_to_read)
        self.read_button.clicked.connect(self.read)

        self.length = QtWidgets.QComboBox()
        self.length.addItem("the length this trace holds", None)

        for metres in FIT_LENGTHS:
            self.length.addItem(f"{metres:.0f} m", float(metres))

        self.length.setToolTip(
            "How long a stretch of trace each plane is worked out over. Left to "
            "itself the trace picks its own -- the length at which the held "
            "share peaks, which is a measurement of the fault. Picking one "
            "instead asserts that this fault holds a single orientation over "
            "that distance; the file records it as window= either way."
        )

        # Re-reads on change, but only over a reading that is already on screen.
        # Two gestures per length would make trying four of them eight presses,
        # and the reading costs ten milliseconds. Never after a Keep, which
        # leaves `_read` None on purpose: a combo that refilled a list somebody
        # had just spent would put the same fits back within reach of one click.
        self.length.currentIndexChanged.connect(self._length_changed)

        self.outcome = QtWidgets.QLabel()
        self.outcome.setWordWrap(True)

        # The lengths that would have answered, where this one did not. Its own
        # label and not a second line of `outcome`, because it is about windows
        # that were not used and has to read as an aside rather than as part of
        # the verdict.
        self.elsewhere = QtWidgets.QLabel()
        self.elsewhere.setWordWrap(True)
        self.elsewhere.setStyleSheet("color: #6a6a6a; font-size: 11px;")

        self.table = QtWidgets.QTableWidget(0, len(FIT_COLUMNS))
        self.table.setHorizontalHeaderLabels(FIT_COLUMNS)
        self.table.verticalHeader().setVisible(False)

        # Never sorted, and for a reason rather than an omission Qt would have
        # filled in: the rows come out in order along the trace, which is the
        # order somebody walking the fault would meet them in, and a click on
        # `plane` would shuffle a fault into a ranking of dip directions. The
        # file order matters twice over -- `attitude_at` takes the first fit that
        # covers a metre, so a header click that reordered these would change
        # what the file on screen says without changing the file.
        self.table.setSortingEnabled(False)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.itemSelectionChanged.connect(self._row_picked)
        self.table.itemChanged.connect(self._ticks_changed)

        # How much of what was just read is already claimed. Said and not left to
        # the tooltips, which are the one part of a window nobody can be told to
        # look at: a row arriving unticked is a decision this window made, and a
        # decision made silently is indistinguishable from a bug.
        self.already = QtWidgets.QLabel()
        self.already.setWordWrap(True)
        self.already.setStyleSheet("color: #8a5000; font-size: 11px;")

        self.gate = QtWidgets.QLabel()
        self.gate.setWordWrap(True)
        self.gate.setStyleSheet("color: #6a6a6a; font-size: 11px;")

        self.keep_button = QtWidgets.QPushButton("Keep the ticked ones")
        self.keep_button.setToolTip(
            "Write the ticked fits into this structure and put them through the "
            "parser, in one step -- the looking that Apply used to stand for has "
            "been done here. Nothing reaches the file until Save."
        )
        self.keep_button.clicked.connect(self.keep)

        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.clicked.connect(self._close_asked)

        pressing = QtWidgets.QHBoxLayout()
        pressing.addWidget(self.read_button)
        pressing.addStretch(1)
        pressing.addWidget(QtWidgets.QLabel("read over:"))
        pressing.addWidget(self.length)

        deciding = QtWidgets.QHBoxLayout()
        deciding.addStretch(1)
        deciding.addWidget(self.keep_button)
        deciding.addWidget(self.close_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self.about)

        # What the file says, then the press, then what came back. The reading
        # gets the larger share of the height because it is the list being worked
        # through; the file's fits are usually one or two lines and are being
        # consulted, not sorted.
        layout.addWidget(self.carries)
        layout.addWidget(self.carried, stretch=1)
        layout.addWidget(self.removing)
        layout.addLayout(pressing)
        layout.addWidget(self.outcome)
        layout.addWidget(self.elsewhere)
        layout.addWidget(self.table, stretch=2)
        layout.addWidget(self.already)
        layout.addWidget(self.gate)
        layout.addLayout(deciding)

        self.retarget()

    def put_steering(self, steering):
        """
        Hangs the hand-steered plane between what the file claims and the sweep.

        Built by the window and not here, because what it steers is drawn on the
        map and this panel has never known that there is a map. Placed here
        because of what it *makes*: a `fit` line on this trace, the same kind of
        thing the sweep below it makes, which the table above it already listed
        under `plane-dem` while living in another window entirely.

        Between the two and not above either, so the window reads downwards as
        what is already claimed, then the two ways to add to it -- a plane laid
        by hand, and a window swept along the trace.
        """

        at = self.layout().indexOf(self.removing) + 1

        for offset, widget in enumerate((self._rule(), steering, self._rule())):
            self.layout().insertWidget(at + offset, widget)

    @staticmethod
    def _rule():
        """A line across the window, which is all the sectioning this needs."""

        rule = QtWidgets.QFrame()
        rule.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        rule.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)

        return rule

    # -- what it is pointed at ---------------------------------------------

    def retarget(self):
        """
        Points at whatever the panel has open, and throws any preview away.

        Thrown away and not kept against the trace it was read off, which would
        be the friendlier thing and the wrong one: these rows are anchored at
        coordinates on one path, the window says which trace it is about in one
        line at the top, and a list surviving a change of selection is a list
        that can be ticked and kept onto a fault it was never read from.

        **The window length is not thrown away with them**, and the difference
        is what each belongs to. A list of fits belongs to one trace. A decision
        that these faults read at 900 m belongs to the sheet, and re-picking it
        for every fault in turn would make a session's worth of identical
        choices out of one.

        **And the file's own fits are read again here**, which is what keeps this
        window from showing a fit that no longer exists. It is enough because of
        where this is called from: a change of selection, and `_on_applied`,
        which goes through `select` -- and `Document` only changes under an
        Apply. Anything that comes to write a block without applying it will have
        to say so here.
        """

        self._read = None
        self._empty()
        self._fill_carried()

        structure = self._structure()

        if structure is None:
            self.about.setText("Nothing selected")
            self.outcome.setText("")
            self.read_button.setEnabled(False)
            self.keep_button.setEnabled(False)
            return

        gstruct = module()

        self.about.setText(
            f"{structure.ident} -- "
            f"{gstruct.path_length(structure.path):.0f} m"
        )

        refused = self.refusal()

        self.read_button.setEnabled(refused is None)
        self.read_button.setToolTip(refused or self._how_to_read)
        self.outcome.setText(refused or "Nothing read yet.")
        self._plainly()

    def _structure(self):
        if self.panel.index is None:
            return None

        return self.panel.document.dataset.structures[self.panel.index]

    def refusal(self):
        """Why the topography cannot be read for this file at all, or None."""

        if self.panel.dem is None:
            return (
                "No DEM in this session, and a plane read off the topography "
                "needs one."
            )

        return self.panel.dem_said

    # -- what the file already claims --------------------------------------

    def _fill_carried(self):
        """
        The fits the document holds along this trace, in the order it holds them.

        **Read out of the document and not out of the panel's box**, which is the
        one choice here with a consequence. The box can hold a `fit` typed and
        not applied, and this table is about what the file claims -- a line
        waiting in the box claims nothing yet, and showing it here would make the
        count disagree with the file for as long as somebody was in the middle of
        typing. It is also the arrangement that survives the box going away.

        The row number is the index into `self._in_file`, because the two are
        built in one pass and this table is never sorted. Which is `rows_of`'s
        rule arriving here as a convenience: order in this format is meaning, so
        there was never going to be a second order to keep track of.
        """

        self._in_file = []
        self.carried.clearContents()
        self.carried.setRowCount(0)

        structure = self._structure()

        if structure is None:
            self.carries.setText("")
            self._offer_removal()
            return

        self._in_file = fits_in(
            self.panel.document.text_of(self.panel.index), structure.path
        )

        self.carried.setRowCount(len(self._in_file))

        for row in range(len(self._in_file)):
            self._write_carried(row)

        self.carried.resizeColumnsToContents()
        self.carries.setText(self._carries_said())
        self._offer_removal()

    def _carries_said(self):
        """What the file claims here, including the rule that orders the claims."""

        how_many = len(self._in_file)

        if not how_many:
            return "No fit in the file along this trace."

        told = [
            f"The file carries {how_many} fit"
            f"{'' if how_many == 1 else 's'} along this trace"
        ]

        # Said whenever there is more than one, and not only where two of them
        # overlap. It is the rule the order of these rows stands for, a curator
        # has no way of guessing it from a table, and the case where it matters
        # is exactly the case where nothing looks wrong.
        if how_many > 1:
            told.append(
                "and where two cover the same metre, the first of them is what "
                "answers there"
            )

        inert = sum(1 for at in range(how_many) if self._inert(at) is not None)

        if inert == how_many == 1:
            told.append("and it answers nowhere")
        elif inert == 1:
            told.append("1 of them answers nowhere")
        elif inert:
            told.append(f"{inert} of them answer nowhere")

        return "; ".join(told) + "."

    def _inert(self, at):
        """
        Why the fit on this row is asked nothing, or None if it answers somewhere.

        Three ways a `fit` line can sit in a file, parse, and mean nothing, all
        three of them in `montealpi_01.gstruct`:

        * its anchors do not read as a stretch at all;
        * they read as one written backwards, which `covers` holds over no ground;
        * every metre of its stretch is claimed by a fit above it.

        None of these is a judgement about the geology, which is why they are
        here and `0.0/0.0` is not: those are statements about what the format
        does with the line, and a curator cannot check any of them by reading.
        """

        claim = self._in_file[at]

        if claim.ends is None:
            return "the stretch this line claims cannot be read off it"

        s0, s1 = claim.ends

        if s0 > s1:
            return (
                f"this line runs from {s0:.0f} m back to {s1:.0f} m: the pair is "
                f"read as written and `covers` is `s0 <= s <= s1`, so it holds "
                f"over no part of the trace"
            )

        if s1 - s0 <= FULLY_M:
            return None

        covered = covered_metres(claim.ends, [one.ends for one in self._in_file[:at]])

        if covered is not None and covered >= (s1 - s0) - FULLY_M:
            return (
                f"every metre of {s0:.0f} to {s1:.0f} m is claimed by a fit "
                f"above this one, and the first one covering a metre is what "
                f"answers there: nothing ever reads this line"
            )

        return None

    def _write_carried(self, at):
        """One fit the file holds, and what the format does with it."""

        claim = self._in_file[at]
        ends, plane = claim.ends, claim.plane
        inert = self._inert(at)
        backwards = ends is not None and ends[0] > ends[1]

        for column, written in enumerate((
            "?" if ends is None else f"{ends[0]:.0f} m",
            "?" if ends is None else f"{ends[1]:.0f} m",
            "?" if plane is None else f"{plane[0]:.0f}/{plane[1]:.0f}",
            f"{claim.attrs['window']} m" if "window" in claim.attrs else "--",
            claim.attrs.get("from", "--"),
        )):
            cell = QtWidgets.QTableWidgetItem(written)
            cell.setFlags(
                QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsSelectable
            )

            # The line itself, which is the thing being shown in cells: a row
            # that reads oddly is a row somebody wants to see the text of, and
            # until the box goes there is nowhere else to look it up.
            cell.setToolTip(claim.line.strip())

            if inert is not None:
                cell.setForeground(QtGui.QColor("#6a6a6a"))
                cell.setToolTip(f"{claim.line.strip()}\n\n{inert}")

            # On the two cells the mistake is in, which is where the ends are.
            # The panel says the same thing in words -- `_show_backwards`, which
            # is what is left of the claims table -- and it says it for the
            # claims no window watches; this is the fits' own copy, in the
            # window that lists them. Neither AOI file has one to show today,
            # the table having found the ones they had.
            if backwards and column in (0, 1):
                cell.setForeground(QtGui.QColor("#b2182b"))

            self.carried.setItem(at, column, cell)

        # Who made it and with what, off the row. `from=` is the column, and the
        # rest of the provenance is what the column can be asked.
        self.carried.item(at, 4).setToolTip(
            " ".join(
                f"{key}={claim.attrs[key]}"
                for key in ("from", "src", "dem", "windows", "step", "north",
                            "converg", "span_verdict")
                if key in claim.attrs
            )
            or claim.line.strip()
        )

    def _carried_row(self):
        """Which fit the file's table is pointing at, or None."""

        rows = {index.row() for index in self.carried.selectedIndexes()}

        if len(rows) != 1:
            return None

        at = rows.pop()

        return at if at < len(self._in_file) else None

    def _offer_removal(self):
        """
        Whether there is a fit to take out, and a press to put back.

        Both are facts about now and neither is about the last gesture, so they
        are settled here and called from everywhere either could have moved --
        the table being refilled under a selection is one of those, which is why
        this is not hung on `itemSelectionChanged` alone.
        """

        at = self._carried_row()

        self.delete_button.setEnabled(at is not None)
        self.delete_button.setToolTip(
            "Take the fit on the selected row out of this structure, through "
            "the parser, in one press. A fit is a plane some producer computed, "
            "with the producer written on it: removed, the file says what it "
            "said before it was computed. Undo puts it back, and nothing "
            "reaches the file until Save."
            if at is None else
            f"Take this line out of the structure:\n\n"
            f"{self._in_file[at].line.strip()}\n\n"
            f"Undo puts it back. Nothing reaches the file until Save."
        )

        self.undo_button.setEnabled(self.panel.may_undo())

    def _carried_picked(self):
        """The stretch of the selected file row, for the map to light."""

        if self._picking:
            return

        self._offer_removal()

        at = self._carried_row()

        if at is None or not self._in_file:
            self.showing.emit(None)
            return

        self._only_here(self.table)
        self.showing.emit(
            self._in_file[at].ends if at < len(self._in_file) else None
        )

        # And the row that has no band says so, which is the rule `claim_said`
        # follows: a picture cannot show a stretch that covers nothing, and a
        # line nothing ever reads looks exactly like a line that answers. Only
        # these, because a row with a band on the map has already been answered
        # and a sentence per click would spend the status bar on saying what is
        # already drawn.
        inert = self._inert(at)

        if inert is not None:
            self.said.emit(inert)

    def delete_carried(self):
        """
        The fit on the selected row out of the file, through the parser at once.

        The press the window has been read for since the file's own fits went
        into it. `montealpi_01.gstruct` carries three fits over the same fifty
        metres of `L0071`, two of them byte-identical, and one more whose ends
        are written backwards so that it holds over no ground at all -- all four
        legal, all four parsing, and until now the only way to be rid of one was
        to find it among the coordinates in the box.

        **Applied in the press**, which is `keep`'s rule and the same argument:
        Apply stands for having looked, and a row that says which stretch it
        claims, which producer made it and that nothing ever reads it has been
        that. It takes the reading with it where there is one -- the file has
        moved under those candidates, and which of them an earlier line still
        covers is now a different answer -- and says so, a list that vanished
        quietly being indistinguishable from one that crashed.
        """

        at = self._carried_row()

        if at is None:
            return False

        claim = self._in_file[at]
        reading = self._read is not None
        refused = self.panel.drop_line(claim.at, claim.line)

        if refused is not None:
            self.said.emit(refused)

            return False

        said = (
            f"removed: {claim.line.strip()} -- Undo puts it back, Save writes "
            f"the file"
        )

        if reading:
            said += "; the reading went with it, the file having changed under it"

        self.said.emit(said)

        return True

    def undo_last(self):
        """The block before the last press that wrote in it, put back."""

        said = self.panel.undo_applied()

        if said is None:
            return False

        self.said.emit(f"{said} -- Save writes the file")

        return True

    def _only_here(self, other):
        """Clears the other table's selection without it taking the band back."""

        self._picking = True

        try:
            other.clearSelection()
        finally:
            self._picking = False

    # -- reading -----------------------------------------------------------

    def _length_changed(self):
        """A new window length, over a reading that is already on screen."""

        if self._read is not None:
            self.read()

    def read(self):
        """The topography along the selected trace, listed and not written."""

        got = self.panel.read_off_dem(self.length.currentData())

        if got is None:
            return None

        self._read = got
        self._fill(got)

        structure = self._structure()
        ident = "" if structure is None else f"{structure.ident}: "

        self.outcome.setText(got.reading.describe())
        self._plainly(bool(got.lines))
        self.gate.setText(f"gate: {self.panel.gate_said()}")
        self._name_the_others(got)

        self.keep_button.setEnabled(bool(self.ticked()))
        self.said.emit(f"{ident}{got.reading.describe()}")

        return got

    def _name_the_others(self, got):
        """
        Which other window lengths would have answered, where this one did not.

        Said only over a `silent` reading -- read, long enough, and nothing held
        -- because that is the verdict the question follows from. A trace off the
        DEM or shorter than the shortest window was never asked, and offering it
        a ladder would be offering to not ask it six more times.

        The length just used is left out of the list by construction, it having
        given nothing; and a trace that answers nowhere gets a sentence rather
        than a blank, because *no window between 150 and 2500 m* closes the
        question and an empty label leaves it open.

        **Said while it happens, because it is not always instant.** A reading is
        ten milliseconds and seven of them usually are too, but the cost is the
        windows inverted rather than the trace: on the AOI's 260 silent traces
        the median scan is under 50 ms, 31 are over 100, and `L0003` -- 19.8 km
        of lineament, 791 window positions at the shortest length -- takes a full
        second. Sampling the DEM once instead of seven times would take that to
        740 ms, which is the same problem with more plumbing, so what is done
        about it is to say so: the label is written and repainted before the work
        starts, and the cursor is the waiting one while it runs.

        `repaint` and not `processEvents`, which matters here: this runs inside
        the combo's own signal, and pumping the event loop would let a second
        change of length re-enter `read` on top of the first.
        """

        if got.lines or not got.reading.silent:
            self.elsewhere.setText("")
            return

        self.elsewhere.setText("Looking at the other window lengths...")
        self.elsewhere.repaint()

        QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor)
        )

        try:
            others = self.panel.lengths_that_hold()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        if not others:
            self.elsewhere.setText(
                f"No window between {FIT_LENGTHS[0]:.0f} and "
                f"{FIT_LENGTHS[-1]:.0f} m gives anything on this trace."
            )
            return

        self.elsewhere.setText(
            "Other windows do: "
            + ", ".join(f"{metres:.0f} m gives {count}" for metres, count in others)
            + "."
        )

    def _plainly(self, held=None):
        """
        The outcome in the colour of what it is: an answer, or nothing to keep.

        `nothing held` is the commonest outcome by far -- 27 of 185 traces on the
        AOI pass the gate -- and it is a verdict rather than a failure, so it is
        not red. Grey where there is nothing, and plain where there is something
        to decide about.
        """

        self.outcome.setStyleSheet(
            "font-size: 11px;" if held else "color: #6a6a6a; font-size: 11px;"
        )

    def _fill(self, got):
        """The stretches, in the order they run along the trace."""

        self._empty()
        self.table.setRowCount(len(got.lines))
        self._filling = True

        try:
            for row, (fit, span) in enumerate(zip(got.reading.fits, got.spans)):
                self._write(row, fit, span)
        finally:
            self._filling = False

        self.table.resizeColumnsToContents()
        self._say_already(got)

    def _say_already(self, got):
        """
        How much of this reading the file has already claimed, in one sentence.

        The summary of what `_write` did row by row, and it exists because the
        rows say it in a tick state and a tooltip. A row arriving unticked is a
        decision this window made on the curator's behalf; made without a word it
        is indistinguishable from a tick that failed to take.
        """

        spent = covered = 0

        for span in got.spans:
            claimed = covered_metres(span, [one.ends for one in self._in_file])

            if claimed is None or claimed <= 0.0 or span is None:
                continue

            if claimed >= (span[1] - span[0]) - FULLY_M:
                spent += 1
            else:
                covered += 1

        told = []

        if spent:
            told.append(
                f"{spent} of these cover ground a fit in the file already "
                f"claims to the last metre: kept, nothing would ever read "
                f"{'them' if spent > 1 else 'it'}, so "
                f"{'they are' if spent > 1 else 'it is'} not ticked"
            )

        if covered:
            told.append(
                f"{covered} overlap{'' if covered > 1 else 's'} a fit in the "
                f"file in part, and answer{'' if covered > 1 else 's'} over the "
                f"rest"
            )

        self.already.setText("; ".join(told) + ("." if told else ""))

    def _empty(self):
        self.table.clearContents()
        self.table.setRowCount(0)
        self.elsewhere.setText("")
        self.already.setText("")
        self.showing.emit(None)

    def _write(self, row, fit, span):
        # What a fit in the file already claims over this same ground, which is
        # the one thing about a candidate that is not a property of the
        # candidate. The lines are kept by appending, so an earlier fit wins
        # every metre the two share: fully covered, this row is a line that
        # parses and is read by nothing, and it arrives unticked.
        claimed = covered_metres(span, [one.ends for one in self._in_file])
        spent = (
            span is not None
            and claimed is not None
            and claimed >= (span[1] - span[0]) - FULLY_M
            and span[1] - span[0] > FULLY_M
        )

        tick = QtWidgets.QTableWidgetItem()
        tick.setFlags(
            QtCore.Qt.ItemFlag.ItemIsEnabled
            | QtCore.Qt.ItemFlag.ItemIsSelectable
            | QtCore.Qt.ItemFlag.ItemIsUserCheckable
        )
        tick.setCheckState(
            QtCore.Qt.CheckState.Unchecked if spent
            else QtCore.Qt.CheckState.Checked
        )

        # Ticked on arrival, which is a claim and worth saying: everything in
        # this list has already been through the gate, so the default is not
        # "probably fine" but "the topography determined this". Untick is for
        # the curator who knows something the topography does not -- a stretch
        # where the trace is a road cutting, a bend that is a digitising
        # artefact.
        #
        # **Unticked and not missing**, where the file already covers the ground.
        # The row is a true thing the topography said and the curator may want it
        # -- the way to have it is to remove the fit above it, which is a decision
        # about the file and not about this list -- so it is offered with the tick
        # off rather than withheld.
        tick.setToolTip(
            "Not ticked: the file already claims this ground, and the first fit "
            "covering a metre is what answers there. Tick it to write a line "
            "nothing will read."
            if spent else
            "Ticked: this one is written when Keep is pressed."
        )

        self.table.setItem(row, 0, tick)

        for column, written in enumerate((
            "?" if span is None else f"{span[0]:.0f} m",
            "?" if span is None else f"{span[1]:.0f} m",
            f"{fit.plane.dip_dir:.0f}/{fit.plane.dip:.0f}",
            f"{fit.attrs.get('window', '?')} m",
        ), start=1):
            cell = QtWidgets.QTableWidgetItem(written)
            cell.setFlags(
                QtCore.Qt.ItemFlag.ItemIsEnabled
                | QtCore.Qt.ItemFlag.ItemIsSelectable
            )
            self.table.setItem(row, column, cell)

        # The diagnostics, off the row and into what the row can be asked. They
        # are the producer's own attributes and go into the file either way;
        # what they are not is five more columns of a window whose argument is
        # that it fits on one screen and reads at a glance.
        self.table.item(row, 4).setToolTip(
            f"averaged over {fit.attrs.get('windows', '?')} windows "
            f"{fit.attrs.get('window', '?')} m long, stepped "
            f"{fit.attrs.get('step', '?')} m, off "
            f"{fit.attrs.get('dem', 'the DEM')}"
        )

        # And which north, on the plane it qualifies. The one attribute here
        # that changes what the number means.
        self.table.item(row, 3).setToolTip(
            f"from {fit.attrs.get('north', 'grid')} north"
            + (
                f", corrected by {fit.attrs['converg']} deg"
                if "converg" in fit.attrs else ""
            )
        )

        # And where the file has some of this ground, that on the two cells it is
        # about. Amber and not red: an overlap is a fact about the file, and the
        # curator may well want the row anyway.
        if claimed:
            how_much = (
                "all of it" if spent
                else f"{claimed:.0f} m of its {span[1] - span[0]:.0f}"
            )

            for column in (1, 2):
                self.table.item(row, column).setForeground(
                    QtGui.QColor("#6a6a6a" if spent else "#8a5000")
                )
                self.table.item(row, column).setToolTip(
                    f"a fit in the file already claims {how_much}, and the "
                    f"first fit covering a metre is what answers there"
                )

    # -- pointing and keeping ----------------------------------------------

    def _row_picked(self):
        """The stretch of the selected row, for the map to light."""

        if self._picking:
            return

        rows = {index.row() for index in self.table.selectedIndexes()}

        if self._read is None or len(rows) != 1:
            self.showing.emit(None)
            return

        row = rows.pop()

        self._only_here(self.carried)
        self.showing.emit(
            self._read.spans[row] if row < len(self._read.spans) else None
        )

    def _ticks_changed(self, _item=None):
        """
        Keep follows the ticks, rather than following there being rows.

        It followed the rows until a row could arrive unticked, and then the two
        came apart in both directions: a reading whose every row is already
        claimed would offer a Keep that writes nothing, and a curator who ticked
        one of them back would find the button dead. The button means *there is
        something to write*, so it is wired to that.
        """

        if self._filling:
            return

        self.keep_button.setEnabled(bool(self.ticked()))

    def ticked(self):
        """The lines the ticks have left, in the order they were read."""

        if self._read is None:
            return []

        return [
            line for row, line in enumerate(self._read.lines)
            if row < self.table.rowCount()
            and self.table.item(row, 0) is not None
            and self.table.item(row, 0).checkState()
            == QtCore.Qt.CheckState.Checked
        ]

    def keep(self):
        """
        The ticked lines into the structure, and the list spent.

        **Spent, and that is not tidiness.** The lines are written by appending,
        so a second press would append them a second time -- two `fit` lines over
        the same stretch, both legal, both parsing, and `attitude_at` taking the
        first. A window whose Keep button is still live after keeping is a window
        that hands you that with one stray click.
        """

        lines = self.ticked()

        if not lines:
            return False

        structure = self._structure()

        if not self.panel.keep_fits(lines):
            return False

        self._read = None
        self._empty()
        self.keep_button.setEnabled(False)

        ident = "" if structure is None else f" on {structure.ident}"

        self.outcome.setText(f"{len(lines)} fit(s) kept{ident}. Save to write them.")
        self._plainly()
        self.said.emit(f"{len(lines)} fit(s) kept{ident}; Save to write the file")

        self.kept.emit(len(lines))

        return True

    def _close_asked(self):
        self.window().close()


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

        # A fourth window, and unlike the other three it is not in the group.
        # The group's satellites come up with the map and are remembered where
        # they were left, which is right for a picture you work beside and wrong
        # for one you open to do a thing and shut: a fit window on screen at
        # start-up would be a window covering the map on behalf of a question
        # nobody has asked yet. `SatelliteWindow` is still what it is made of --
        # a real window, parented so it neither counts towards the last one
        # closed nor outlives the tool, and closing hides it rather than
        # destroying what is in it.
        self.fit_panel = FitFromDem(self.panel)
        self.fit_panel.showing.connect(self._show_fitting)
        self.fit_panel.said.connect(self.say)

        # Built here rather than in the panel, because every one of these four
        # wires ends on the map: what it steers is drawn, and `FitFromDem` has
        # never known that there is a map. Where it is *shown* is the panel's
        # business, and the panel puts it between the file's fits and the sweep.
        self.steering = PlaneSteering(refusal=self._steering_refusal())
        self.steering.steered.connect(self._steer)
        self.steering.take_asked.connect(self._take_plane)
        self.steering.armed_changed.connect(self._steering_armed)
        self.steering.unpin_asked.connect(self.unpin)
        self.steering.start_asked.connect(self._start_fit)

        self.fit_panel.put_steering(self.steering)

        self.fit_window = SatelliteWindow(
            "gSurf - fits along this trace", self.fit_panel, FIT_WINDOW_PX, parent=self
        )
        self.fit_window.visibility_changed.connect(self._fitting_visible)
        self._fit_placed = False

        # The readings beside the fits, and a second window rather than a second
        # table in that one: `fits_in` refuses to mix what a computation returned
        # with what a compass was pointed at, and the gesture here -- taking a
        # measurement off a trace -- must never be one row away from the gesture
        # that deletes a fit. Its band goes to the same sink, `_show_fitting`
        # being the one place that decides what a band on the map means.
        self.readings_panel = ReadingsHere(self.panel)
        self.readings_panel.showing.connect(self._show_fitting)
        self.readings_panel.said.connect(self.say)
        self.readings_panel.wrote.connect(self._readings_wrote)
        self.readings_panel.point_wanted.connect(self._readings_want_point)

        self.readings_window = SatelliteWindow(
            "gSurf - measurements along this trace",
            self.readings_panel,
            READING_WINDOW_PX,
            parent=self,
        )
        self._readings_placed = False

        # And the one that writes over what that one lists, which is why it is a
        # third window and not a third group inside it. Add and Detach are
        # decisions about whether a line exists; this changes what a line claims,
        # and the old claim survives only where something puts it somewhere --
        # see `AmendReading`, where the format decides what that something is.
        self.amend_panel = AmendReading(self.panel)
        self.amend_panel.showing.connect(self._show_fitting)
        self.amend_panel.said.connect(self.say)
        self.amend_panel.wrote.connect(self._amend_wrote)
        self.amend_panel.point_wanted.connect(self._amend_wants_point)

        self.amend_window = SatelliteWindow(
            "gSurf - amend a measurement",
            self.amend_panel,
            AMEND_WINDOW_PX,
            parent=self,
        )
        self._amend_placed = False

        # The third of these, on the axis neither of the other two can touch. Its
        # band goes to the same sink for the same reason, and it writes through
        # the same `insert_claim`: what is different is only that what it claims
        # is a stretch of ground rather than a plane.
        self.exposure_panel = ExposureHere(self.panel)
        self.exposure_panel.showing.connect(self._show_fitting)
        self.exposure_panel.said.connect(self.say)
        self.exposure_panel.wrote.connect(self._exposure_wrote)
        self.exposure_panel.ends_wanted.connect(self._exposure_wants_ends)

        self.exposure_window = SatelliteWindow(
            "gSurf - exposure along this trace",
            self.exposure_panel,
            EXPOSURE_WINDOW_PX,
            parent=self,
        )
        self._exposure_placed = False

        # And the one that spends what that one writes. Its band is the licensed
        # stretch, so it goes to the same sink; its region does not, a stretch of
        # trace and a patch of hillside being two different pictures.
        self.facet_panel = FacetHere(self.panel)
        self.facet_panel.showing.connect(self._show_fitting)
        self.facet_panel.drawing.connect(self._show_facet)
        self.facet_panel.said.connect(self.say)
        self.facet_panel.wrote.connect(self._facet_wrote)

        self.facet_window = SatelliteWindow(
            "gSurf - the surface where it crops out",
            self.facet_panel,
            FACET_WINDOW_PX,
            parent=self,
        )
        self._facet_placed = False

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
        What is left on the map's own frame once the steering has gone: the
        legend's two boxes.

        The steering was here, on the argument that what it steers is a picture
        -- curves swinging about a pin -- and that the judgement it serves is
        made by looking at them against the trace, so a dial on the other
        monitor would be steering by feel. That argument was not wrong and it
        was outweighed. A dial bolted to the map's frame is 190 px of map gone
        for good, whether or not anybody is steering; and the thing it makes is
        a `fit` line, which the fit window now lists, marks and will delete --
        so leaving the steering out of it meant reaching a `plane-dem` line
        through a window named after another producer.

        What it costs is real and is left to the hand: the fit window can be
        put on the other screen, and then the feel is gone. It opens at the
        map's own corner, which is where the trade stays paid.
        """

        beside = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(beside)
        layout.setContentsMargins(0, 0, 0, 0)
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
        """Everything that can be done to the open trace, and the way back to a
        window that has been closed.

        **The first menu is named for its subject and not for its first entry.**
        It was `Fit` while the fit window was the only thing in it; four more
        doors moved in, the label did not move, and five of the six things this
        tool can do to a trace spent that time filed under the name of the
        sixth. The one who could not find them was the person who put them
        there.

        Singular, because every entry acts on the selected block and on no
        other -- the fit sweep included, which reads one trace at a time.
        `Traces` would promise something over the file, and nothing here does
        that.

        The entries lost `along this trace` with the rename: four of the six
        were saying it each, under a menu that now says it once.
        """

        trace = self.menuBar().addMenu("&Trace")

        # Without this, `QMenu` shows none of the tooltips set below -- which is
        # how the two refusals at the foot of this method came to be written for
        # nobody. They are the reason an entry is grey, they are set on the
        # entry, and Qt's default is to drop them on the floor.
        #
        # Worth checking on a *disabled* entry when this is touched, because
        # that is the whole case: a grey door with no reason beside it is the
        # thing being fixed, and some styles do not hover a disabled item.
        trace.setToolTipsVisible(True)

        self.fit_action = QtGui.QAction("&Fits...", self)
        self.fit_action.setShortcut("Ctrl+D")
        self.fit_action.triggered.connect(self.open_fitting)
        trace.addAction(self.fit_action)

        # In the same menu and never greyed, which is the one difference from the
        # entry above it: this window needs no DEM, only the block. A file opened
        # without any raster at all can still be asked what it measures and told
        # that a measurement is not of this fault.
        self.readings_action = QtGui.QAction("&Measurements...", self)
        self.readings_action.setShortcut("Ctrl+M")
        self.readings_action.triggered.connect(self.open_readings)
        trace.addAction(self.readings_action)

        # Beside it and never greyed either, for the same reason: what it needs
        # is the block. It is a separate entry rather than a button inside that
        # window because of what it does -- the only gesture in this tool that
        # writes over a claim somebody already made.
        self.amend_action = QtGui.QAction("Amend a measu&rement...", self)
        self.amend_action.setShortcut("Ctrl+R")
        self.amend_action.triggered.connect(self.open_amend)
        trace.addAction(self.amend_action)

        # Never greyed either, and the argument is stronger here than for the
        # readings: this one asserts a field observation, and the DEM in it is
        # evidence beside the assertion rather than a condition on it. A session
        # with no raster can still record that a contact crops out.
        self.exposure_action = QtGui.QAction("&Exposure...", self)
        self.exposure_action.setShortcut("Ctrl+E")
        self.exposure_action.triggered.connect(self.open_exposure)
        trace.addAction(self.exposure_action)

        # Greyed with the fit window and off the same answer, for that entry's
        # reason: a facet is a region of topography, so unlike the two above it
        # this one cannot be done at all without a DEM these traces may be read
        # against.
        self.facet_action = QtGui.QAction(
            "The surface where it crops &out...", self
        )
        self.facet_action.setShortcut("Ctrl+T")
        self.facet_action.triggered.connect(self.open_facet)
        trace.addAction(self.facet_action)

        # Under a separator, and with no ellipsis, because it is the one entry
        # here that does not ask: the five above open a window and this one
        # writes a line straight into the box. That is Qt's own convention for
        # the difference, and it is the distinction the panel used to make with
        # three `+` buttons beside one ending in dots.
        #
        # It is in this menu at all because a `span` is a claim about the open
        # trace like the other five, and `use` is the one claim in the format
        # with no window to make it in -- so this is the last abbreviation left
        # for somebody to finish by hand.
        trace.addSeparator()

        self.span_action = QtGui.QAction("&Span", self)
        self.span_action.setShortcut("Ctrl+U")
        self.span_action.setToolTip(
            f"Write `{SPAN_TEMPLATE.strip()}` above the path, with the first "
            f"end selected: shift-click the map to fill it in, and the next one "
            f"is selected in turn. `use` is what decides -- `certainty` and "
            f"`exposure` describe the contact."
        )
        self.span_action.triggered.connect(self._add_span)
        trace.addAction(self.span_action)

        facet_refused = self.facet_panel.refusal()

        if facet_refused:
            self.facet_action.setEnabled(False)
            self.facet_action.setToolTip(facet_refused)

        # Greyed off the window's own answer rather than a second copy of it: a
        # menu entry that opens a window with a refusal in it costs the gesture
        # before it answers, and two places deciding separately whether there is
        # a DEM is two places to disagree. There used to be a second place -- the
        # panel's own fit button, working it out for itself.
        refused = self.fit_panel.refusal()

        if refused:
            self.fit_action.setEnabled(False)
            self.fit_action.setToolTip(refused)

        # Every door to the open trace, kept as a list because they are handed
        # round the windows in `_build_shortcuts`: a `QAction` on this window
        # reaches only this window, and the box these aim through is in another
        # one.
        self.trace_actions = (
            self.fit_action,
            self.readings_action,
            self.amend_action,
            self.exposure_action,
            self.facet_action,
            self.span_action,
        )

        menu = self.menuBar().addMenu("&Windows")

        self.window_actions = self.group.actions_into(
            menu, {"panel": "&Structures", "net": "Stereo&net"}
        )

        menu.addSeparator()

        front = QtGui.QAction("Bring all to &front", self)
        front.triggered.connect(self.group.raise_all)
        menu.addAction(front)

    def open_fitting(self):
        """
        Brings up the fit window on the selected trace, wherever it was left.

        **It is not retargeted when it is already open**, and that is the whole
        of the guard: pressing the button a second time is somebody bringing a
        window to the front, and a list of fits thrown away by a click meant to
        raise the thing holding it is the kind of loss that has no undo and no
        trace. It retargets on a change of selection, which is the gesture that
        actually makes the list about the wrong fault -- see `select`.
        """

        if not self.fit_window.isVisible():
            self.fit_panel.retarget()
            self._place_fitting()

        self.fit_window.show()
        self.fit_window.raise_()
        self.fit_window.activateWindow()

    def _fitting_visible(self, shown):
        """
        Takes the steered plane off the map when the window holding its dial goes.

        The steering is a mode, and a mode whose way out has just been hidden is
        a trap: closing this window used to be impossible, the dial being bolted
        to the map's frame, and now it is one click. What would be left on the
        map is a cut and a band nothing on screen can turn or switch off.

        The numbers stay in the boxes, so coming back by Ctrl+D and ticking the
        box again puts the same plane back. What is dropped is the drawing, and
        it is said rather than done quietly: a picture that vanishes on its own
        is a picture somebody goes looking for.
        """

        if shown or not self.steering.armed():
            return

        self.steering.on.setChecked(False)
        self.say(
            "The steered plane is off the map: its dial went with the window "
            "(Ctrl+D brings both back, on the same plane)."
        )

    def open_readings(self):
        """
        Brings up the measurements window on the selected trace.

        Retargeted on every opening, unlike the fit window, and the difference is
        what each holds. That one carries a list nobody has decided about yet and
        throwing it away on a click meant to raise a window is a real loss; this
        one holds nothing but what the file says, so reading the block again is
        free and is the only way it cannot be stale.
        """

        self.readings_panel.retarget()
        self._place_readings()
        self.readings_window.show()
        self.readings_window.raise_()
        self.readings_window.activateWindow()

    def open_exposure(self):
        """
        Brings up the exposure window on the selected trace.

        Retargeted on every opening, like the readings window and unlike the fit
        window: what it holds is the file's own spans, so reading them again is
        free, and the picked stretch is the one thing worth keeping -- which
        `retarget` drops, because a stretch picked before the window was last
        closed is two progressives nobody is looking at any more.
        """

        self.exposure_panel.retarget()
        self._place_exposure()
        self.exposure_window.show()
        self.exposure_window.raise_()
        self.exposure_window.activateWindow()

    def _exposure_wrote(self):
        """A declared stretch reaches everything else that draws the block."""

        # The band, the lanes and the row all read the model, and a span changes
        # what the lanes draw on this axis -- which is the picture this window
        # exists to put something into.
        self.fit_panel.retarget()
        self.readings_panel.retarget()
        self._tell_next()

    def _only_claimant(self, armed):
        """Arming one claimant on the shift-click disarms every other.

        Two modes held at once would leave `_on_map_pressed` deciding which of
        them a click belongs to by the order of its branches, which is the thing
        that function's own comment refuses to do. So the exclusivity is here,
        where it is a sentence: the last button pressed is the one that is armed.

        **One list, written once**, which it was not when there were two of
        these: a pair of methods each disarming the other is a rule that is
        correct for two claimants and silently incomplete for three. The third
        arrived and the pair still compiled.
        """

        for button in (
            self.readings_panel.pick_point,
            self.exposure_panel.pick_ends,
            self.amend_panel.pick_point,
        ):
            if button is not armed:
                button.setChecked(False)

    def _exposure_wants_ends(self, on):
        if on:
            self._only_claimant(self.exposure_panel.pick_ends)

    def _readings_want_point(self, on):
        if on:
            self._only_claimant(self.readings_panel.pick_point)

    def _amend_wants_point(self, on):
        if on:
            self._only_claimant(self.amend_panel.pick_point)

    def open_facet(self):
        """
        Brings up the facet window on the selected trace.

        Retargeted on every opening, like the other two readers: what it holds
        is the file's readings and the licence over them, so reading them again
        is free. The grown facet goes with the retarget, which is right -- a
        region measured before the window was last closed is a picture of ground
        nobody is looking at, and leaving it would leave it drawn on the map.
        """

        self.facet_panel.retarget()
        self._place_facet()
        self.facet_window.show()
        self.facet_window.raise_()
        self.facet_window.activateWindow()

    def _facet_wrote(self):
        """A kept facet fit reaches everything else that draws the block."""

        self.fit_panel.retarget()
        self.readings_panel.retarget()
        self._tell_next()

    def _show_facet(self, facet):
        """
        The grown region on the map, as its own cells, or taken off.

        Through `on_map` cell by cell rather than as a raster laid on an extent:
        the map can be in a projection the file is not, and a region is the one
        thing here that would come out wrong rather than merely shifted -- an
        image placed by its corners in another projection is skewed in between,
        and a skewed facet would look like a measurement of a surface that is
        not there.
        """

        if facet is None:
            self.facet_drawn.set_data([], [])
            self.map_view.blit()

            return

        drawn = np.asarray(
            self.on_map(facets.cells_of(facet, cap=FACET_CELLS_DRAWN)),
            dtype=float,
        )

        if not len(drawn):
            self.facet_drawn.set_data([], [])
        else:
            self.facet_drawn.set_data(drawn[:, 0], drawn[:, 1])

        self.map_view.blit()

    def _place_facet(self):
        """Offset further in again, so the four do not land as one."""

        if self._facet_placed:
            return

        self._facet_placed = True

        available = self.screen().availableGeometry()
        frame = self.frameGeometry()

        self.facet_window.move(
            min(
                frame.left() + FIT_OFFSET_PX[0] * 4,
                available.right() - self.facet_window.width(),
            ),
            min(
                frame.top() + FIT_OFFSET_PX[1] * 4,
                available.bottom() - self.facet_window.height(),
            ),
        )

    def _place_exposure(self):
        """Offset further in again, so the three do not land as one."""

        if self._exposure_placed:
            return

        self._exposure_placed = True

        available = self.screen().availableGeometry()
        frame = self.frameGeometry()

        self.exposure_window.move(
            min(
                frame.left() + FIT_OFFSET_PX[0] * 3,
                available.right() - self.exposure_window.width(),
            ),
            min(
                frame.top() + FIT_OFFSET_PX[1] * 3,
                available.bottom() - self.exposure_window.height(),
            ),
        )

    def _readings_wrote(self):
        """A detachment reaches everything else that draws the block."""

        # The map's bands and the row's `holds` fraction both come off the model,
        # and a reading taken out moves the second: a trace answering for its
        # last 150 m through a measurement answers for none of it afterwards.
        self.fit_panel.retarget()
        self.amend_panel.retarget()
        self._tell_next()

    def _amend_wrote(self):
        """An amended reading reaches everything else that draws the block."""

        # Everything a detachment reaches, and the facet window as well: its
        # seeds are the readings, their licence is read at the progressive each
        # of them sits at, and a moved reading is a seed at another place -- so
        # a grown region left on screen would belong to a site nothing claims
        # any more.
        self.fit_panel.retarget()
        self.readings_panel.retarget()
        self.facet_panel.retarget()
        self._tell_next()

    def open_amend(self):
        """
        Brings up the amend window on the selected trace.

        Retargeted on every opening, like the other three readers and for their
        reason: what it holds is the file's own readings. The picked place goes
        with the retarget, which matters more here than in any of them -- a
        coordinate clicked before this window was last closed is a place on the
        map that a press would write into a line, and the one thing it must not
        do is still be held when somebody comes back to a different reading.
        """

        self.amend_panel.retarget()
        self._place_amend()
        self.amend_window.show()
        self.amend_window.raise_()
        self.amend_window.activateWindow()

    def _place_amend(self):
        """Offset further in again, so the five do not land as one."""

        if self._amend_placed:
            return

        self._amend_placed = True

        available = self.screen().availableGeometry()
        frame = self.frameGeometry()

        self.amend_window.move(
            min(
                frame.left() + FIT_OFFSET_PX[0] * 5,
                available.right() - self.amend_window.width(),
            ),
            min(
                frame.top() + FIT_OFFSET_PX[1] * 5,
                available.bottom() - self.amend_window.height(),
            ),
        )

    def _place_readings(self):
        """Offset further in than the fit window, so the two do not land as one."""

        if self._readings_placed:
            return

        self._readings_placed = True

        available = self.screen().availableGeometry()
        frame = self.frameGeometry()

        self.readings_window.move(
            min(
                frame.left() + FIT_OFFSET_PX[0] * 2,
                available.right() - self.readings_window.width(),
            ),
            min(
                frame.top() + FIT_OFFSET_PX[1] * 2,
                available.bottom() - self.readings_window.height(),
            ),
        )

    def _place_fitting(self):
        """
        Down and in from the map's corner, the first time, and never after.

        A palette's placement and not a satellite's. The group puts its windows
        down the right edge of the desktop and remembers where they were dragged
        to; this one is opened against a trace that is on screen *now*, so the
        rule it has to obey is the one Qt's own default for a child window
        breaks -- it must not come up centred over the map. After the first time
        the window keeps the position it was left at within the session, which is
        what `_fit_placed` is for.
        """

        if self._fit_placed:
            return

        self._fit_placed = True

        available = self.screen().availableGeometry()
        frame = self.frameGeometry()

        self.fit_window.move(
            min(
                frame.left() + FIT_OFFSET_PX[0],
                available.right() - self.fit_window.width(),
            ),
            min(
                frame.top() + FIT_OFFSET_PX[1],
                available.bottom() - self.fit_window.height(),
            ),
        )

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

        **And the fit window, which is not in the group**, so iterating the group
        had left it out. That was invisible while the window was a list to tick
        and shut; it stopped being invisible when the steering moved in, because
        then the press that keeps a plane and the key that writes the file were in
        two different windows for no reason anybody chose. What keeps that window
        out of the group is where it is *shown* -- it must not come up at
        start-up -- which has nothing to do with what a key does in it.

        **And the `Trace` menu's six**, which had the same failure and kept it
        longer, because nothing about them looked like a shortcut: they are menu
        entries, the menu bar is the map window's, and a `QAction` living there
        is a `WindowShortcut` on that one window. `SatelliteWindow` sets
        `Qt.WindowType.Window`, so the panel is a window and not a pane --
        meaning Ctrl+M did nothing from the panel, which is the window holding
        the box and the caret that all six of those doors aim through. The keys
        worked from the one window where there was nothing to aim with.
        """

        mine = []

        for label, shortcut, slot in (
            ("Save", "Ctrl+S", self.save),
            ("Apply", "Ctrl+Return", self.panel.apply_block),
        ):
            action = QtGui.QAction(label, self)
            action.setShortcut(shortcut)
            action.triggered.connect(slot)

            self.addAction(action)
            mine.append(action)

        # Every window this tool owns, and the list is one place on purpose: a
        # window added here and forgotten there is a Ctrl+S that works from four
        # windows out of six, which fails silently and only sometimes -- the same
        # failure the net was argued into this list for, and the one the menu's
        # entries were quietly in until they were handed round too.
        for satellite in (
            *self.group.satellites.values(),
            self.fit_window,
            self.readings_window,
            self.amend_window,
            self.exposure_window,
            self.facet_window,
        ):
            for action in (*mine, *self.trace_actions):
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

        # The grown facet, as a stipple of its own cells. Lowest of everything
        # that is about a claim, because it is the only artist here that is
        # about a *region*: a band is drawn along the trace and a cut crosses
        # it, and both have to stay legible over this.
        #
        # Square markers, so the stipple reads as cells of a grid rather than as
        # a scatter of measurements -- there are already round dots on this map
        # and they are stations, which is the one thing this must not look like.
        self.facet_drawn = self.map_view.add_animated(
            axes.add_line(
                Line2D(
                    [], [], color=FACET_TINT, marker="s",
                    markersize=FACET_CELL_PX, markeredgewidth=0.0,
                    alpha=FACET_ALPHA, linestyle="none", zorder=5.1,
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

        # And the fit window follows the selection, because what is in it is
        # anchored on the path that was just left. Here and not on
        # `panel.selected`, which is the request: the panel is asked first and
        # can refuse -- a block typed and not applied keeps the selection where
        # it is -- and a list thrown away on a move that did not happen would be
        # a list lost to a click that changed nothing.
        self.fit_panel.retarget()
        self.readings_panel.retarget()

        # And this one drops a picked coordinate as well as a list, which is the
        # one of these that would be dangerous left behind: a place clicked on
        # the trace just left, still held, is a point on another fault that the
        # next press would write into a line as a correction.
        self.amend_panel.retarget()

        # And this one has more than a list to drop: its picked ends are
        # progressives, which on the next trace are a different stretch. See
        # `_forget_stretch`.
        self.exposure_panel.retarget()

        # The facet window drops the region it grew, which takes it off the map:
        # a patch of hillside measured from a station on the trace just left
        # would otherwise sit there under a different fault.
        self.facet_panel.retarget()

        # Said last, because every line of it is about this trace: which one is
        # open is the first thing the next step depends on.
        self._tell_next()

    def _show_fitting(self, interval):
        """
        The stretch the fit window is pointing at, or back to what the block says.

        One band and two things with an opinion about it, which is what the
        second one arriving made visible. The band means one thing -- the stretch
        under discussion -- and while a row in the fit window is selected that is
        the row; when none is, it is whatever the block claims, which is where
        the band came from and what it goes back to.

        **Back to, and not blank**, and the check that caught it is the one about
        the steering: pointing at nothing is not the same statement as there
        being nothing to point at, and a window emitting the first on every
        change of selection was quietly erasing the second. A picture that goes
        empty when a window that is not even open changes its mind is a picture
        that cannot be trusted while it is full.
        """

        if interval is not None:
            # The band is no longer showing what the panel last said, so the
            # panel has to stop believing that it is. See `forget_covering`.
            self.panel.forget_covering()

        self._show_claimed(
            self.panel._covering_now() if interval is None else interval
        )

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

        `000/00` is not taken, because it is not a plane anybody wrote: it is
        what the `+ fit` template carries until somebody fills it in, and taking
        it means the caret arriving at the line you are about to write undoes
        the steering you did to have something to write. Found on a file --
        `0.0/0.0 from=plane-dem`, a provenance stamped on a decision never made.
        The general question of who wins between the hand and the line is not
        answered here and is not meant to be: it stops being a question when the
        box stops being free text.
        """

        self.steering.set_writable(self.panel.has_plane_slot())

        if plane is not None and plane != (0.0, 0.0):
            self.steering.show_plane(*plane)

        if self.steering.armed():
            self._steer(*self.steering.plane())

        self._tell_next()

    def _steering_armed(self, on):
        """The box turning the steering on and off."""

        self.steering.set_writable(self.panel.has_plane_slot())
        self._tell_next()

        if on:
            self._steer(*self.steering.plane())
            return

        self._clear_steering()
        self.map_view.blit()

    def _tell_next(self):
        """
        The next gesture, into the steering's own line. Cheap enough per frame.

        And whether the press is live, worked out from the same state at the same
        moment. It used to be set on the panel's `holding` signal alone, which
        fires when the caret's *plane* changes -- so moving from a reading to a
        `fit` carrying the same two numbers changed nothing and left the button in
        the state the other line put it in. `QLabel.setText` and
        `QPushButton.setEnabled` both drop a value they already have, so asking
        both questions on every frame of the dial costs nothing.
        """

        self.steering.set_writable(self.panel.has_plane_slot())
        self.steering.tell_next(self._next_step())

    def _add_span(self):
        """A `use` for somebody to finish, asked for from the menu.

        `_start_fit`'s sibling, and what the panel's `+ span` button used to do.
        The words it says are the button's tooltip turned into news, because a
        menu entry has nowhere to put a tooltip at the moment it is triggered:
        the line is written in the other window, and from the map that is the
        only way anybody learns the press landed.
        """

        if self.panel.index is None:
            self.say("nothing selected to put a span on")

            return

        self.panel.add_line(SPAN_TEMPLATE)
        self.say(
            "a new `span` above the path, with its first end armed -- "
            "shift-click the map for each end, then say why in `reason=`"
        )

    def _start_fit(self):
        """A `fit` for the steered plane to go in, asked for from its own window."""

        if self.panel.index is None:
            self.say("nothing selected to put a fit on")

            return

        self.panel.add_line(FIT_TEMPLATE)
        self.say(
            "a new `fit` above the path, with its first end armed -- shift-click "
            "the map for each end, then `Keep this plane`"
        )
        self._tell_next()

    def _by_clicking(self, said):
        """
        A step that is a click on the map, or what is keeping one from landing.

        The navigation bar owns the mouse while one of its buttons is down, and
        those buttons stay down: `MapView._on_press` hands nothing on, so the pin
        and the ends cannot be clicked and the line below went on asking for them
        anyway. This is the state that had somebody unable to write a
        `from=plane-dem` fit three evenings running, and nothing on screen was
        wrong -- the advice was right, the gesture was right, and the click was
        going somewhere else.

        Returned instead of the step and not in front of it: the rule here is
        that the answer is the earliest thing still missing, and a mode that eats
        the gesture is earlier than the gesture.
        """

        named = self.map_view.navigating_as()

        if named is None:
            return said

        return (
            f"release `{named}` in the map's toolbar: while it is on, a click "
            f"drives the map instead of reaching the trace"
        )

    def _next_step(self):
        """
        The one gesture that would take a steered plane nearer the file, or None.

        **Worked out from the state, every time it is asked.** The alternative is
        a counter stepped on by each press, and it would be wrong within two
        gestures of anybody doing something out of order -- which is the whole
        point of a tool where the plane can be pinned before the stretch is
        decided, and where a `fit` already in the file can be clicked into and
        re-steered. There is no sequence to be at step 3 of; there is a line, a
        pin and a dial, and a next thing to do follows from the three of them.

        Asked in the order the work goes in, so the answer is always the earliest
        thing still missing. That ordering is the content: `shift-click the two
        ends` is useless advice to somebody whose caret is on a `path` line, and
        both are useless to somebody who has not switched the steering on.

        None where there is nothing to say -- a session with no DEM, where the
        controls are grey and carry the reason as a tooltip. A next step under
        dead controls would be an instruction nobody can follow.
        """

        if self.steering.refusal:
            return None

        if self.index is None:
            return "pick a trace, on the map or in the list"

        if not self.steering.armed():
            return "switch `plane on the DEM` on, above"

        # Both ends written is not the same thing as a stretch, and this is the
        # state the AOI was left in: `Mt. Alpi faults.2` carries a plane steered
        # by hand over `@583458.91,4439774.76 @582408.83,4441315.77`, which is
        # 2689 m back to 791 m, and `covers` being `s0 <= s <= s1` it holds over
        # no metre of anything. It was kept in one press -- legally, both ends
        # being written -- and saved, and the only thing on screen that ever said
        # so was a grey row in a table.
        #
        # **In front of the pin and not after it**, which is not an ordering
        # preference: a pair the wrong way round has no middle, so `pinned_at`
        # answers None for it and the advice about pinning would arrive first --
        # true, useless, and about the wrong line.
        claimed = interval_of(
            self.panel.line_now(), self.document.dataset.structures[self.index].path
        )

        if claimed is not None and claimed[0] > claimed[1]:
            return self._by_clicking(
                f"turn this line's ends round -- written as they are it runs "
                f"from {claimed[0]:.0f} m back to {claimed[1]:.0f} m and holds "
                f"over no ground: select one end and shift-click the map"
            )

        if self._pin is None:
            return self._by_clicking(
                "ctrl-click the trace to pin the plane on it, or put the caret "
                "on a line that claims a stretch"
            )

        # `press + fit` first, and the button is now in this window. It used to
        # name the box's one -- three windows away from a hand on a dial -- and
        # then offered the caret as the alternative, which is the gesture that
        # parked somebody on a compass reading and let the dial be written into
        # it. The caret stays offered, because clicking into a computed `fit` and
        # re-steering it is half of what this window is for; it stays second.
        if not self.panel.has_plane_slot():
            word = (self.panel.line_now().split() or [""])[0]
            whose = f" -- the caret is on `{word}`" if word else ""

            return (
                f"press `+ fit` to start a line for this plane, or put the caret "
                f"on a `fit` in the box{whose}"
            )

        line = self.panel.line_now()
        carried = plane_of(line)
        dialled = tuple(round(one, PLANE_DECIMALS) for one in self.steering.plane())
        elsewhere = carried is None or tuple(
            round(one, PLANE_DECIMALS) for one in carried
        ) != dialled

        # A `fit` the file already has, with the dial somewhere else: the press
        # would replace that plane, in one press, and it is the state this window
        # *opens* in -- the caret parks on the last line before the path, which on
        # a trace with fits is one of them. Legitimate, and not a thing to be
        # walked into: what the line said here was `turn the dial, then press Keep
        # this plane`, which is an instruction to overwrite somebody's answer with
        # no mention that that is what it is.
        if elsewhere and anchors_written(line) and self.panel.in_document(line):
            return (
                f"press `+ fit` for a new line, or `Keep this plane` to replace "
                f"the {carried[0]:.0f}/{carried[1]:.0f} the file has on this one"
                if carried is not None
                else "press `+ fit` for a new line to put this plane on"
            )

        # The line not yet saying what the dial says, which covers the two cases
        # that are the same case: a template still at `000/00`, and a plane
        # steered somewhere else since it was last written.
        if elsewhere:
            return (
                "turn the dial until the cut runs along the trace, then press "
                "`Keep this plane`"
            )

        if not anchors_written(line):
            return self._by_clicking(
                "shift-click the two ends of the stretch it holds over, then "
                "press `Keep this plane` again"
            )

        # Whether the *line* is claimed, and not whether the document has
        # unsaved changes. The two were the same test here once and it said `the
        # file has it` about a line finished in the box and kept by nothing --
        # the state two clicks into the sequence, where the one press left is the
        # one this was supposed to be asking for.
        if not self.panel.in_document(line):
            return (
                "press `Keep this plane` again: the line is finished, and this "
                "is the press that puts it in"
            )

        if self.document.dirty:
            return "Ctrl+S to write it into the file"

        # Nothing about this line, which is not nothing to do: this window is for
        # adding claims to a trace, and the state it opens in on a file with fits
        # in it is the caret sitting on one of them, finished and saved.
        return (
            "nothing on this line -- `+ fit` here starts another, or "
            "ctrl-click the trace to steer a plane somewhere else"
        )

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
        self._tell_next()

    def unpin(self):
        """Gives the pin back to the line under the caret."""

        if self._free_pin is None:
            return

        self._free_pin = None
        self.steering.set_pinned(False)
        self.say("the plane is back on the line the caret is on")
        self._resteer()
        self._tell_next()

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
            self._tell_next()

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
        self._tell_next()

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

        written, kept = self.panel.keep_plane(dip_dir, dip, {
            "from": FROM_STEERED,
            "src": "gsurf",
            "dem": self.session.dem.path.name,
            "north": "true",
            "converg": f"{convergence:+.2f}",
            "at": f"@{self._pin[0]:.2f},{self._pin[1]:.2f}",
        })

        if written is None:
            # Naming the line, because the press is right and the line is wrong,
            # and a complaint about the press sends the hand back to the dial.
            word = (self.panel.line_now().split() or ["nothing"])[0]

            self.say(
                f"a steered plane goes into a `fit` line, and the caret is on "
                f"`{word}`: press `+ fit` to start one -- it comes up with its "
                f"two ends empty, which is the stretch this is about"
            )

            return

        # Which of the two happened, in the sentence, because the difference is
        # whether the file now claims this. The unkept half ends in what to do
        # next rather than in a complaint: the plane is on the line, the ends are
        # still the whole trace, and clicking two of them is the missing step and
        # not a correction of this one.
        if kept:
            self.say(f"kept: {written.strip()} -- Save to put it in the file")
        else:
            self.say(
                f"on the line: {written.strip()} -- shift-click its two ends, "
                f"then press again; as it stands it would claim the whole trace"
            )

        self._tell_next()

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

        shifted = bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier)

        # Before `pick`, because while that button is down this *is* what a
        # shift-click means. Decided here with the other two for the reason
        # written above: which modifier wins is a sentence somebody can read, and
        # not the order of the branches inside `pick`.
        if shifted and self.readings_panel.wanting_point():
            self._point_for_reading(x, y)

            return

        # The third and fourth claimants, and no two of them can be armed at
        # once -- see `_only_claimant`, which is where that is enforced rather
        # than here, so the order of these branches decides nothing.
        if shifted and self.exposure_panel.wanting_ends():
            self._end_for_exposure(x, y)

            return

        if shifted and self.amend_panel.wanting_point():
            self._point_for_amendment(x, y)

            return

        self.pick(x, y, anchor=shifted)

    def _end_for_exposure(self, x, y):
        """A click sent to the exposure window as a progressive on this trace.

        The progressive and not the point, which is the difference from
        `_point_for_reading`: a reading is *at* a place and its distance off the
        trace is part of what it says, where an end of a span is a place *along*
        one and has no off-trace meaning at all -- `Span.resolve` projects it and
        keeps nothing else. So the projection happens here and the anchor is
        written from it at the press.
        """

        if self.index is None:
            self.say("nothing selected to pick a stretch on")
            return

        structure = self.document.dataset.structures[self.index]

        if len(structure.path) < 2:
            self.say(f"{structure.ident} has no path to pick a stretch on")
            return

        s, off = place_on(structure.path, *self.in_file(x, y))
        first = not self.exposure_panel.half_picked()

        self.exposure_panel.took_end(s)

        self.say(
            f"{'one end' if first else 'the other end'} at {s:.0f} m along "
            f"{structure.ident}"
            + (f", {off:.0f} m from where you clicked" if off > SAME_OUTCROP_M else "")
            + ("  -- shift-click the other" if first else "")
        )

    def _point_for_reading(self, x, y):
        """A click sent to the measurements window as it landed, with its distance.

        Projected here and snapped there, if at all: this is the half that needs
        the path, and whether the anchor ends up on the trace is a statement about
        what was measured, which the window holds and a click does not.
        """

        if self.index is None:
            self.say("nothing selected to measure on")
            return

        structure = self.document.dataset.structures[self.index]

        if len(structure.path) < 2:
            self.say(f"{structure.ident} has no path to measure against")
            return

        here = self.in_file(x, y)
        s, off = place_on(structure.path, *here)

        self.readings_panel.took_point(*here, s, off)
        self.say(
            f"the measurement goes at {s:.0f} m along {structure.ident}, "
            + ("on the trace" if self.readings_panel.snapping()
               else f"{off:.1f} m off it")
            + " -- dial the plane and press `Add this reading`"
        )

    def _point_for_amendment(self, x, y):
        """A click sent to the amend window, projected the same way.

        `_point_for_reading`'s twin and deliberately the same arithmetic: a
        reading added and a reading moved end up in the same slot of the same
        kind of line, so two projections rounding differently would make the
        two gestures write two slightly different anchors for one click.

        What it says afterwards is not the same, and that is the difference
        worth having here: an addition has only to be dialled, where a move is
        about to overwrite the one copy of where the reading already was, and
        the metres it changes hands over are measured in the window.
        """

        if self.index is None:
            self.say("nothing selected to amend on")
            return

        structure = self.document.dataset.structures[self.index]

        if len(structure.path) < 2:
            self.say(f"{structure.ident} has no path to amend against")
            return

        here = self.in_file(x, y)
        s, off = place_on(structure.path, *here)

        self.amend_panel.took_point(*here, s, off)
        self.say(
            f"it would go at {s:.0f} m along {structure.ident}, "
            + ("on the trace" if self.amend_panel.snapping()
               else f"{off:.1f} m off it")
            + " -- what that changes is in the window"
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

            turned = self.panel.insert_anchor(*snapped)

            # Nothing written, and the panel has said why. No dot either: the
            # green dot is where the anchor went, and drawn over a refusal it
            # would be the one half of the report that looked like success.
            if turned is None:
                return

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
            # Said and not done quietly. Turning the pair round is a change to
            # what somebody clicked, and the one place it could matter is the
            # one where it was not a mis-click: an end deliberately put past the
            # other, to be moved afterwards. That is rare and this is one
            # Ctrl-Z, where a silent swap would be a line that does not say what
            # the hand said and nothing anywhere admitting it.
            if turned:
                said += (
                    "; the ends were the other way round and have been turned "
                    "-- this trace runs the other way"
                )

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

        # The file having it is a state like any other, and the one the next step
        # was pointing at: said here, or the line would go on asking for a Save
        # that has happened.
        self._tell_next()

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
