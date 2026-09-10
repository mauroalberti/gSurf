"""
Located attitudes: the measurements a structural tool reads.

This is not the backdrop layer of `vectors.py`, which is drawn and no more. Here
the points *are* the data: each carries an orientation, and what a tool does
with them depends on that orientation being right. So the reading is strict
where it has to be and forgiving where the geology says it should be, and it
says out loud what it would not take.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class AttitudeSource:
    """
    A point layer read as planar attitudes, with the poles taken once.

    The poles are built at load and kept: they are what every window is a
    subset of, and rebuilding a few dozen of them per frame would be paying
    for the same arithmetic hundreds of times a second.
    """

    # What the azimuth field means. Dip direction is what an Italian survey
    # writes down (`Immersione`); the right-hand-rule strike is what an
    # English-language one usually does.
    CONVENTIONS = (
        ("dip direction", False),
        ("strike, right-hand rule", True),
    )

    def __init__(
        self,
        path,
        crs,
        layer=None,
        dip_dir_field=None,
        dip_field=None,
        is_rhr_strike=False,
        bounds=None,
    ):
        import geopandas as gpd

        self.path = Path(path)
        self.layer = layer
        self.dip_dir_field = dip_dir_field
        self.dip_field = dip_field
        self.is_rhr_strike = bool(is_rhr_strike)
        self.problem = None
        self.dropped = {}
        self.outside = 0

        self.xy = np.empty((0, 2))
        self.azimuths = np.empty(0)
        self.dips = np.empty(0)
        self.poles = []

        try:
            frame = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
        except Exception as err:
            self.problem = str(err).split("\n")[0]
            return

        if frame.crs is None:
            self.problem = "no CRS"
            return

        for field in (dip_dir_field, dip_field):
            if field not in frame.columns:
                self.problem = f"no field '{field}'"
                return

        frame = frame.to_crs(crs)
        frame = frame[frame.geometry.notna() & (frame.geometry.geom_type == "Point")]

        if frame.empty:
            self.problem = "no point geometry"
            return

        azimuth = self._numeric(frame[dip_dir_field])
        dip = self._numeric(frame[dip_field])

        keep = self._admissible(azimuth, dip)

        if not keep.any():
            self.problem = "no readable attitude"
            return

        frame = frame[keep]
        azimuth, dip = azimuth[keep], dip[keep]

        # A horizontal bed has no dip direction: whatever the field holds --
        # 999 in the CARG sheets, a blank, last measurement's leftover -- it is
        # not a bearing, and the pole is vertical whichever way it is read.
        azimuth = np.where(dip == 0.0, 0.0, azimuth)

        self.frame = frame
        self.xy = np.c_[frame.geometry.x.to_numpy(), frame.geometry.y.to_numpy()]
        self.azimuths = azimuth
        self.dips = dip
        self.poles = self._pole_axes(azimuth, dip)

        if bounds is not None:
            left, bottom, right, top = bounds
            inside = (
                (self.xy[:, 0] >= left)
                & (self.xy[:, 0] <= right)
                & (self.xy[:, 1] >= bottom)
                & (self.xy[:, 1] <= top)
            )
            self.outside = int((~inside).sum())

    # -- reading -----------------------------------------------------------

    @staticmethod
    def _numeric(column):
        """The column as floats, with whatever will not convert becoming NaN."""

        import pandas as pd

        return pd.to_numeric(column, errors="coerce").to_numpy(dtype=float)

    def _admissible(self, azimuth, dip):
        """
        Which rows carry an attitude, and a tally of why the others do not.

        The dip decides first, because it decides whether the azimuth means
        anything. Out of 0-90 there is no plane: the CARG sheets write 99 for
        contorted bedding measured as a mean, and taken at face value that is a
        plane overturned past the vertical. Only once the dip is a real one,
        and not zero, does the azimuth have to be a bearing.
        """

        dropped = {}

        dip_known = np.isfinite(dip)
        dip_sane = dip_known & (dip >= 0.0) & (dip <= 90.0)

        dropped["dip missing"] = int((~dip_known).sum())
        dropped["dip outside 0-90"] = int((dip_known & ~dip_sane).sum())

        # Zero dip is exempt: the azimuth is not read, so it cannot be wrong.
        needs_azimuth = dip_sane & (dip > 0.0)
        azimuth_known = np.isfinite(azimuth)
        azimuth_sane = azimuth_known & (azimuth >= 0.0) & (azimuth < 360.0)

        dropped["dip direction missing"] = int((needs_azimuth & ~azimuth_known).sum())
        dropped["dip direction outside 0-360"] = int(
            (needs_azimuth & azimuth_known & ~azimuth_sane).sum()
        )

        self.dropped = {reason: count for reason, count in dropped.items() if count}

        return dip_sane & (~needs_azimuth | azimuth_sane)

    def _pole_axes(self, azimuth, dip):
        """
        The poles as geogst axes, downward-pointing and sign-blind.

        An axis and not a direction: the orientation tensor is the axial
        counterpart of a mean, and a bed dipping 30 to the east and the same bed
        overturned are the same pole. That is also why overturned bedding needs
        no special case here -- it does for polarity, which is not this
        question.
        """

        from geogst.core.geology.orientations import Plane

        return [
            Plane(float(a), float(d), is_rhr_strike=self.is_rhr_strike).normal_axis()
            for a, d in zip(azimuth, dip)
        ]

    @staticmethod
    def numeric_fields(path, layer=None):
        """The numeric fields of the layer, which are the ones worth offering."""

        import pyogrio

        info = pyogrio.read_info(path, layer=layer) if layer else pyogrio.read_info(path)

        return [
            str(field)
            for field, dtype in zip(info["fields"], info["dtypes"])
            if str(dtype) != "object"
        ]

    # -- windows -----------------------------------------------------------

    @property
    def is_loaded(self):
        return len(self.poles) > 0

    def __len__(self):
        return len(self.poles)

    def within(self, x, y, radius):
        """
        The stations inside a circle, as indices.

        A circle and not a cell: a cell would make the answer depend on which
        way the grid happens to be turned, and there is no north in the question
        being asked.

        Scanned outright rather than through a spatial index. One window on 1757
        stations costs 0.05 ms against 0.013 with a KD-tree, and neither is what
        a frame is spent on. This once read that a tree would earn its place at
        ten thousand windows; measured, it does not -- a 16289-cell grid spends
        0.89 s searching against 0.046 with a tree, out of 6.9 s in total, so the
        tree would buy 13% of a field in exchange for a dependency. It would earn
        its place if the tensor stopped dominating, which is the opposite of what
        happened.
        """

        if not self.is_loaded:
            return np.empty(0, dtype=int)

        offsets = self.xy - np.array([x, y])

        return np.flatnonzero((offsets * offsets).sum(axis=1) <= radius * radius)

    def poles_at(self, indices):
        return [self.poles[i] for i in indices]

    def dip_directions(self):
        """The azimuths as dip directions, whichever way they were written."""

        if not self.is_rhr_strike:
            return self.azimuths

        return (self.azimuths + 90.0) % 360.0

    def summary(self):
        if self.problem:
            return f"{self.layer or self.path.stem}: {self.problem}"

        text = f"{self.layer or self.path.stem}: {len(self)} attitudes"

        if self.dropped:
            detail = ", ".join(f"{count} {reason}" for reason, count in self.dropped.items())
            text += f" ({detail})"

        if self.outside:
            text += f"; {self.outside} outside the map"

        return text
