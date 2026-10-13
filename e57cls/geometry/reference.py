"""Independent scalar projection: no production camera or pose calls."""

import math


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _rotate(p, rotation, inverse=False):
    w = rotation["w"]
    q = [rotation[k] * (-1 if inverse else 1) for k in "xyz"]
    a = _cross(q, p)
    b = _cross(q, a)
    return [p[i] + 2*w*a[i] + 2*b[i] for i in range(3)]


def project_point(point, scan_pose, image_pose, kind, params):
    world = _rotate(point, scan_pose["rotation"])
    relative = [world[i] + scan_pose["translation"][k] - image_pose["translation"][k]
                for i, k in enumerate("xyz")]
    x, y, z = _rotate(relative, image_pose["rotation"], inverse=True)
    if kind == "pinholeRepresentation":
        return [params["principalPointX"] - (x/z)*(params["focalLength"]/params["pixelWidth"]),
                params["principalPointY"] - (y/z)*(params["focalLength"]/params["pixelHeight"])]
    azimuth = math.atan2(y, x)
    u = params["imageWidth"]/2 - azimuth/params["pixelWidth"]
    if abs(params["imageWidth"]*params["pixelWidth"] - 2*math.pi) < 1e-6:
        u %= params["imageWidth"]
    if kind == "sphericalRepresentation":
        elevation = math.asin(max(-1., min(1., z/math.sqrt(x*x+y*y+z*z))))
        v = params["imageHeight"]/2 - elevation/params["pixelHeight"]
    elif kind == "cylindricalRepresentation":
        v = params["principalPointY"] - z*params["radius"]/(params["pixelHeight"]*math.sqrt(x*x+y*y))
    else:
        raise ValueError("Unsupported reference projection")
    return [u, v]
