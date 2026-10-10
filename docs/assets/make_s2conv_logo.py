#!/usr/bin/env python
"""Logo of s2conv: a hexagon-tiled sphere in the style of the sax logo of s2fft and
s2scat, with the fibre of frames over one point, a cone whose rim carries the phase
colours, i.e. a spin field lifted to the rotation group.

Pure Python, no dependencies; writes s2conv_logo.svg (for light backgrounds) and
s2conv_logo_dark.svg.  PNG files, transparent, e.g.
    cairosvg s2conv_logo.svg -o s2conv_logo.png --output-width 300
    rsvg-convert -w 300 s2conv_logo.svg -o s2conv_logo.png

Usage: python make_s2conv_logo.py [--out-dir DIR]
"""
import argparse
import colorsys
import math
import os
import random

CX, CY, R = 545.0, 570.0, 405.0       # sphere centre and radius (SVG y points down)
A = 0.16                              # circumradius of a tile in the tangent plane [rad]
SHRINK = 0.84                         # tiles are shrunk about their centres
CONE_AT = (0.74, 0.74)                # tangent-plane position of the fibre
CONE_H, CONE_R = 0.70, 0.40           # fibre: height and rim radius (fractions of R)

# colour stops along the diagonal, upper left to lower right (the sax palette)
STOPS = [(-1.0, (0.70, 0.37, 0.78)), (-0.15, (0.13, 0.58, 0.51)), (1.0, (0.29, 0.48, 0.83))]
ORANGE = (0.97, 0.58, 0.13)                                  # the tile under the fibre
VARIANTS = {'light': ('#0E1B24', 1.00), 'dark': ('#0E1B24', 1.10)}   # tile stroke, brightness


def lerp3(a, b, u):
    return tuple(x + (y - x) * u for x, y in zip(a, b))


def band(g):
    g = max(-1.0, min(1.0, g))
    for (g0, c0), (g1, c1) in zip(STOPS, STOPS[1:]):
        if g <= g1:
            return lerp3(c0, c1, (g - g0) / (g1 - g0))
    return STOPS[-1][1]


def hexcol(c, f=1.0):
    return '#%02X%02X%02X' % tuple(int(round(255 * max(0.0, min(1.0, x * f)))) for x in c)


def to_sphere(u, v):
    """Tangent-plane point -> unit vector (inverse azimuthal equidistant) in the view frame
    (x right, y up, z towards the viewer); points beyond the limb are pinned to it."""
    rho, phi = min(math.hypot(u, v), math.pi / 2), math.atan2(v, u)
    return (math.sin(rho) * math.cos(phi), math.sin(rho) * math.sin(phi), math.cos(rho))


def screen(p, scale=1.0):
    return CX + R * scale * p[0], CY - R * scale * p[1]


def centre(q, r):
    return A * math.sqrt(3) * (q + r / 2), A * 1.5 * r


def hex_outline(cu, cv, a, n_sub=3):
    """Vertices of a pointy-top hexagon in the plane, each edge subdivided."""
    pts = []
    for k in range(6):
        t0, t1 = math.radians(30 + 60 * k), math.radians(90 + 60 * k)
        p0 = (cu + a * math.cos(t0), cv + a * math.sin(t0))
        p1 = (cu + a * math.cos(t1), cv + a * math.sin(t1))
        pts += [(p0[0] + (p1[0] - p0[0]) * j / n_sub, p0[1] + (p1[1] - p0[1]) * j / n_sub) for j in range(n_sub)]
    return pts


BOX = []                                                     # every point drawn, for the view box


def path(points, close=True):
    BOX.extend(points)
    return 'M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in points) + (' Z' if close else '')


def poly(points, fill, stroke, sw=5.0):
    return (f'<path d="{path(points)}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linejoin="round"/>')


def phase_ring(ell, width, nseg=96):
    """A ring of phase colours, one turn of hue, along the ellipse ell(t)."""
    out = []
    for k in range(nseg):
        ta, tb = 2 * math.pi * k / nseg, 2 * math.pi * (k + 1.06) / nseg
        col = colorsys.hsv_to_rgb(((k + 0.5) / nseg) % 1.0, 0.82, 0.96)
        (xa, ya), (xb, yb) = ell(ta), ell(tb)
        BOX.extend([(xa, ya), (xb, yb)])
        out.append(f'<path d="M{xa:.1f},{ya:.1f} L{xb:.1f},{yb:.1f}" stroke="{hexcol(col)}" '
                   f'stroke-width="{width}" fill="none"/>')
    return out


def convex_hull(points):
    """Convex hull (monotone chain) of points, rounded to 0.01."""
    def cross(o, a_, b_):
        return (a_[0] - o[0]) * (b_[1] - o[1]) - (a_[1] - o[1]) * (b_[0] - o[0])

    pts = sorted(set((round(x, 2), round(y, 2)) for x, y in points))
    lower, upper = [], []
    for p_ in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p_) <= 0:
            lower.pop()
        lower.append(p_)
    for p_ in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p_) <= 0:
            upper.pop()
        upper.append(p_)
    return lower[:-1] + upper[:-1]


def make(variant):
    stroke, bright = VARIANTS[variant]
    rnd = random.Random(7)
    lt = (-0.50, 0.62, 0.61)
    nl = math.sqrt(sum(x * x for x in lt))
    lt = tuple(x / nl for x in lt)
    tiles = [(q, r) for q in range(-12, 13) for r in range(-12, 13) if math.hypot(*centre(q, r)) < math.pi / 2 - 0.02]
    q0, r0 = min(tiles, key=lambda t: math.hypot(centre(*t)[0] - CONE_AT[0], centre(*t)[1] - CONE_AT[1]))
    BOX.clear()
    out = []
    # the tiles, shaded by a light from the upper left, the tile under the fibre orange
    for q, r in tiles:
        cu, cv = centre(q, r)
        n = to_sphere(cu, cv)
        g = (n[0] - n[1]) / math.sqrt(2)
        diffuse = max(0.0, sum(a * b for a, b in zip(n, lt)))
        f = bright * (0.80 + 0.36 * diffuse) * (0.86 + 0.14 * n[2]) * (1 + rnd.uniform(-0.05, 0.05))
        pts = [screen(to_sphere(u, v)) for u, v in hex_outline(cu, cv, A * SHRINK)]
        out.append(poly(pts, hexcol(ORANGE) if (q, r) == (q0, r0) else hexcol(band(g), f), stroke))
    # the fibre over that tile: apex at the tile, rim of radius CONE_R at height CONE_H, axis radial
    n0 = to_sphere(*centre(q0, r0))
    ang = math.atan2(n0[1], n0[0])                       # radial direction on the screen (math angle)
    ux, uy = -math.sin(ang), -math.cos(ang)              # unit along the limb, SVG coordinates
    vx, vy = math.cos(ang), -math.sin(ang)               # unit radial, SVG coordinates
    apex = screen(n0)
    ccx, ccy = screen(n0, 1 + CONE_H)
    ra, rb = CONE_R * R, CONE_R * R * max(0.25, n0[2])

    def ell(t):
        return (ccx + ra * math.cos(t) * ux + rb * math.sin(t) * vx,
                ccy + ra * math.cos(t) * uy + rb * math.sin(t) * vy)

    rim = [ell(2 * math.pi * k / 120) for k in range(120)]
    out.append(poly(convex_hull([apex] + rim), '#F6C391', '#B4541A', 6.0))
    out.append(f'<path d="{path(rim)}" fill="#FDE9D5" stroke="none"/>')
    out += phase_ring(ell, 46)
    out.append(f'<circle cx="{rim[0][0]:.1f}" cy="{rim[0][1]:.1f}" r="26" fill="#D11515" stroke="white" stroke-width="6"/>')
    out.append('</svg>')
    pad = 22.0                                           # margin for strokes
    x0, y0 = min(p[0] for p in BOX) - pad, min(p[1] for p in BOX) - pad
    x1, y1 = max(p[0] for p in BOX) + pad, max(p[1] for p in BOX) + pad
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{x1 - x0:.0f}" height="{y1 - y0:.0f}" '
            f'viewBox="{x0:.1f} {y0:.1f} {x1 - x0:.1f} {y1 - y0:.1f}">')
    return '\n'.join([head] + out) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    for variant in VARIANTS:
        name = os.path.join(args.out_dir, 's2conv_logo' + ('' if variant == 'light' else '_dark') + '.svg')
        with open(name, 'w') as fh:
            fh.write(make(variant))
        print(name)


if __name__ == '__main__':
    main()
