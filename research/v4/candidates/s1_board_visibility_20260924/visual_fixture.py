"""Isolated S1 wood-board display correction; original collision remains untouched.

The #156 board was defined only in geom group 0. LIBERO's offscreen policy
views show the group's visual counterpart (group 1), as for the table. This
function adds a noncontact textured board twin without changing the original
board's body, dimensions, material, contact flags or task semantics.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from research.v4.candidates.s1_scene_20260924.fixture import add_fixtures

HERE = Path(__file__).resolve().parent
TEXTURE = HERE / "maple_board_1k.png"


def make_texture() -> None:
    """Create an original warm maple texture, distinct from the dark table."""
    n = 1024
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    grain = (5.8 * np.sin(.037 * y + 1.1 * np.sin(.009 * x))
             + 2.0 * np.sin(.16 * y + .006 * x))
    grain += np.random.default_rng(156).normal(0, 1.2, (n, n))
    rgb = np.clip(np.array([204, 164, 111])[None, None, :]
                  + grain[:, :, None], 0, 255).astype(np.uint8)
    Image.fromarray(rgb, "RGB").save(TEXTURE)


def add_board_visual(root: ET.Element) -> ET.Element:
    board = root.find(".//body[@name='s1_board']")
    collision = root.find(".//geom[@name='s1_board_collision']")
    if board is None or collision is None:
        raise ValueError("Original S1 physical board must exist first")
    if board.find("geom[@name='s1_board_visible']") is not None:
        raise ValueError("S1 board visual already installed")
    asset = root.find("asset")
    if asset is None:
        raise ValueError("MuJoCo asset element missing")
    ET.SubElement(asset, "texture", name="s1_maple_board_tex", type="2d", file=str(TEXTURE))
    ET.SubElement(asset, "material", name="s1_maple_board", texture="s1_maple_board_tex",
                  rgba="1 1 1 1", specular=".12", shininess=".12")
    # The 0.3 mm visual offset avoids coplanar flicker if group 0 is rendered.
    ET.SubElement(board, "geom", name="s1_board_visible", type="box",
                  pos="0 0 .90415", size=".135 .130 .00615",
                  material="s1_maple_board", group="1", contype="0",
                  conaffinity="0", mass="0")
    return root


def add_fixtures_visible(root: ET.Element, hob_xy=None, board_xy=None) -> ET.Element:
    kwargs = {}
    if hob_xy is not None:
        kwargs["hob_xy"] = hob_xy
    if board_xy is not None:
        kwargs["board_xy"] = board_xy
    add_fixtures(root, **kwargs)
    return add_board_visual(root)
