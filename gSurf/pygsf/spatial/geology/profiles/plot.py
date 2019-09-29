
from warnings import warn

from functools import singledispatch


from matplotlib.figure import Figure


import matplotlib.pyplot as plt


from pygsf.spatial.geology.profiles.geoprofiles import *


z_padding = 0.2


@singledispatch
def plot(
    obj,
    **kargs
) -> Figure:
    """

    :param obj:
    :param kargs:
    :return:
    """

    fig = kargs.get("fig", None)
    aspect = kargs.get("aspect", 1)
    width = kargs.get("width", 18.5)
    height = kargs.get("height", 10.5)

    if fig is None:

        fig, ax = plt.subplots()
        fig.set_size_inches(width, height)

        ax.set_aspect(aspect)

    else:

        ax = plt.gca()

    return fig


@plot.register(GeoProfile)
def _(
    geoprofile: GeoProfile,
    **kargs
) -> Figure:
    """
    Plot a single geological profile.

    :param geoprofile: the geoprofile to plot
    :type geoprofile: GeoProfile
    :return: the figure.
    :rtype: Figure
    """

    fig = kargs.get("fig", None)
    plot_z_min = kargs.get("plot_z_min", None)
    plot_z_max = kargs.get("plot_z_max", None)

    if plot_z_min is None or plot_z_max is not None:
        z_range = geoprofile.z_max() - geoprofile.z_min()
        plot_z_min = geoprofile.z_min() - z_padding * z_range
        plot_z_max = geoprofile.z_max() + z_padding * z_range

    if fig is None:

        aspect = kargs.get("aspect", 1)
        width = kargs.get("width", 18.5)
        height = kargs.get("height", 10.5)

        fig, ax = plt.subplots()
        fig.set_size_inches(width, height)

        ax.set_aspect(aspect)
        ax.set_ylim([plot_z_min, plot_z_max])

    else:

        ax = plt.gca()

    if geoprofile.topo_profile:

        topo_color = kargs.get("topo_color", "blue")

        ax.plot(
            geoprofile.topo_profile.s_arr(),
            geoprofile.topo_profile.z_arr(),
            color=topo_color
        )

    if geoprofile.attitudes:

        attits = geoprofile.attitudes

        attitude_color = kargs.get("attitude_color", "red")
        section_length = geoprofile.length_2d()

        projected_z = [structural_attitude.z for structural_attitude in attits if
                       0.0 <= structural_attitude.s <= section_length]

        projected_s = [structural_attitude.s for structural_attitude in attits if
                       0.0 <= structural_attitude.s <= section_length]

        projected_ids = [structural_attitude.id for structural_attitude in attits if
                         0.0 <= structural_attitude.s <= section_length]

        axes = fig.gca()
        vertical_exaggeration = axes.get_aspect()

        axes.plot(projected_s, projected_z, 'o', color=attitude_color)

        # plot segments representing structural data

        for structural_attitude in attits:
            if 0.0 <= structural_attitude.s <= section_length:

                structural_segment_s, structural_segment_z = structural_attitude.create_segment_for_plot(
                    section_length,
                    vertical_exaggeration)

                fig.gca().plot(structural_segment_s, structural_segment_z, '-', color=attitude_color)

    if geoprofile.lines_intersections:

        if not geoprofile.topo_profile:

            warn('Topographic profile is not defined, so intersections cannot be plotted')

        else:

            for profile_part in geoprofile.lines_intersections:

                prof_part_id = profile_part.id
                id_color = "orange"
                parts = profile_part.parts

                for s_range in parts:

                    s_start, s_end = s_range[0], s_range[1] if len(s_range) > 1 else None
                    s_vals = geoprofile.topo_profile.s_subset(s_start, s_end)
                    z_vals = geoprofile.topo_profile.zs_from_s_range(s_start, s_end)
                    fig.gca().plot(s_vals, z_vals, 'o', color=id_color)

    if geoprofile.polygons_intersections:

        pass

    return fig


@plot.register(GeoProfileSet)
def _(
    geoprofiles: GeoProfileSet,
    **kargs
) -> List[Figure]:
    """
    Plot a set of geological profiles.

    :param geoprofiles: the geoprofiles to plot
    :type geoprofiles: GeoProfiles
    :return: the figures.
    :rtype: List[Figure]
    """

    num_subplots = geoprofiles.num_profiles()

    z_range = geoprofiles.z_max() - geoprofiles.z_min()
    plot_z_min = geoprofiles.z_min() - z_range * z_padding
    plot_z_max = geoprofiles.z_max() + z_range * z_padding

    figs = []
    for ndx in range(num_subplots):
        geoprofile = geoprofiles.extract_geoprofile(ndx)
        fig = plot(
            geoprofile,
            plot_z_min=plot_z_min,
            plot_z_max=plot_z_max
        )
        figs.append(fig)

    return figs