"""Test compatibility of converted object with echoshader's echogram functions"""

import holoviews
from echoshader.elements.echogram import (
    echogram,
    tricolor_echogram,
)


def test_echogram(ds):

    channel = ds.channel.values[0]

    plot = echogram(
        ds,
        channel=channel,
        vert_dim="depth",
    )

    assert isinstance(plot, holoviews.Image)


def test_echogram_with_accessor(ds):

    channel = ds.channel.values[0]

    plot = ds.eshader.echogram(
        channel=channel,
        vert_dim="depth",
    )

    assert isinstance(plot, holoviews.Image)


def test_multichannel_echogram(ds):

    channels = ds.sortby("frequency_nominal").channel.values.tolist()

    plot = echogram(
        ds,
        channel=channels,
        vert_dim="depth",
    )

    assert isinstance(plot, holoviews.Layout)


def test_tricolor(ds):

    rgb_channels = [
        ds.channel.sel(channel=ds.frequency_nominal == frequency).item()
        for frequency in (120000, 38000, 18000)
    ]

    plot = tricolor_echogram(
        ds,
        channel=rgb_channels,
        vmin=-80,
        vmax=-30,
        vert_dim="depth",
    )

    assert isinstance(plot, holoviews.RGB)
