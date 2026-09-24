"""
The boundary with gstruct, where a plane changes library.

`gstruct` is the text format the curation is written in and read back from --
one fact per line, anchored to coordinates rather than to progressives, and
readable without GDAL, which is the property it was built for. It lives in its
own repository and gSurf imports it; this module is the only place that does,
so that everything the two projects have to agree about is agreed in one file.

What they have to agree about, first of all, is which way a normal points.

Both libraries name a plane the same way -- dip direction and dip angle, in
degrees -- so the conversion carries two numbers across and invents nothing.
The agreement that matters is one level down. FORMAT.md is explicit about it:
the normal points *upward*, and its horizontal component points *toward* the
dip, not against it. So does geogst, through `norm_direct_up`; a plane dipping
30 to the east has the normal (0.5, 0, 0.866) in both, east-north-up.

The trap is that `norm_direct_up` is not the method gSurf reaches for. What it
uses everywhere else is `Plane.normal_axis`, which points *down*, and which is
right where it is used: the orientation tensor is axial, a bed and the same bed
overturned are one pole, and a sign there would be a distinction without a
difference. Measured over eight planes, `norm_direct_up` agrees with gstruct
eight times and `normal_axis` once -- the once being the vertical plane, where
up and down are both horizontal and the question does not arise.

Which is exactly why the convention is pinned in `checks/check_curation.py`
rather than trusted. An error of 180 degrees in a normal does not show up when
two planes are compared, because the angle between them is taken through the
absolute value of the dot product and the sign cancels; it shows up much later,
somewhere a plane is compared against a normal, as a result that is wrong by a
hemisphere and looks like a result.
"""

from __future__ import annotations


def gstruct_plane(plane):
    """
    A geogst plane as a gstruct one.

    `Plane` carries its azimuth as a dip direction whatever it was built from:
    `is_rhr_strike` is a way of answering the constructor, not a state the
    object keeps, so there is nothing to ask here and nothing to convert.
    """

    import gstruct

    return gstruct.Plane(float(plane.dipazim), float(plane.dipang))


def geogst_plane(plane):
    """
    A gstruct plane as a geogst one.

    `is_rhr_strike=False` and not the source's setting, because what is coming
    in is a dip direction: the format writes `140/31` and says in its own
    grammar that the first number is the dip direction. A layer read as RHR
    strike was converted on the way in, and converting again on the way back
    would turn a right angle into a fact about nothing.
    """

    from geogst.core.geology.orientations import Plane

    return Plane(float(plane.dip_dir), float(plane.dip), is_rhr_strike=False)
