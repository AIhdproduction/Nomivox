"""Small reusable E57 fixtures for SLAM input-mode tests."""

import io
from uuid import uuid4

import numpy as np
from PIL import Image
import pye57
from pye57 import libe57 as lib


_COLORS = {
    "wall": (220, 40, 40),
    "floor": (40, 220, 40),
    "ceiling": (40, 40, 220),
    "box": (220, 220, 40),
}


def _label_rays(position, directions):
    """Return the image colours for the first room surface hit by each ray."""
    origin = np.asarray(position, np.float64)
    height, width = directions.shape[:2]
    nearest = np.full((height, width), np.inf)
    labels = np.full((height, width), "", dtype="U7")

    def plane(axis, value, label, bounds=None):
        component = directions[..., axis]
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = (value-origin[axis])/component
        valid = distance > 1e-6
        hit = origin+distance[..., None]*directions
        if bounds:
            for coordinate, low, high in bounds:
                valid &= (hit[..., coordinate] >= low) & (hit[..., coordinate] <= high)
        take = valid & (distance < nearest)
        nearest[take], labels[take] = distance[take], label

    for axis, value in ((0, -6.), (0, 6.), (1, -6.), (1, 6.)):
        plane(axis, value, "wall")
    plane(2, -2., "floor")
    plane(2, 4., "ceiling")
    plane(0, 1.5, "box", ((1, -.9, .9), (2, -.5, 1.5)))
    image = np.zeros((height, width, 3), np.uint8)
    for label, color in _COLORS.items():
        image[labels == label] = color
    return image


def _surface_image(position, width=128, height=64):
    """Render the labelled synthetic room into a panorama for fake SAM tests."""
    u, v = np.meshgrid(np.arange(width)+.5, np.arange(height)+.5)
    azimuth = (width/2-u)*(2*np.pi/width)
    elevation = (height/2-v)*(np.pi/height)
    directions = np.stack((np.cos(elevation)*np.cos(azimuth),
                           np.cos(elevation)*np.sin(azimuth), np.sin(elevation)), axis=-1)
    return _label_rays(position, directions)


def _pinhole_surface_image(position, width=128, height=64, focal=64.):
    """Render a forward-facing labelled pinhole image for fast pipeline tests."""
    u, v = np.meshgrid(np.arange(width)+.5, np.arange(height)+.5)
    local = np.stack(((u-width/2)/focal, (v-height/2)/focal, -np.ones_like(u)), axis=-1)
    local /= np.linalg.norm(local, axis=-1, keepdims=True)
    # The camera pose rotates negative local Z to positive world X.
    directions = np.stack((-local[..., 2], local[..., 1], local[..., 0]), axis=-1)
    return _label_rays(position, directions)


def _png_bytes(position=None, pinhole=False):
    stream = io.BytesIO()
    image = (Image.fromarray(_pinhole_surface_image(position) if pinhole else _surface_image(position)) if position is not None
             else Image.new("RGB", (16, 8), (128, 140, 152)))
    image.save(stream, format="PNG")
    return np.frombuffer(stream.getvalue(), np.uint8)


def _pose(image_file, position, rotation=None):
    pose = lib.StructureNode(image_file)
    rotation = rotation or {"w": 1., "x": 0., "y": 0., "z": 0.}
    for group, values in (("rotation", rotation),
                          ("translation", dict(zip("xyz", map(float, position))))):
        node = lib.StructureNode(image_file)
        pose.set(group, node)
        for key, value in values.items():
            node.set(key, lib.FloatNode(image_file, value))
    return pose


def _image(writer, index, position, associated_guid=None, timestamp=None, labelled=False, pinhole=False):
    image_file = writer.image_file
    node = lib.StructureNode(image_file)
    writer.root["images2D"].append(node)
    node.set("guid", lib.StringNode(image_file, f"slam-image-{index:03d}"))
    if associated_guid is not None:
        node.set("associatedData3DGuid", lib.StringNode(image_file, associated_guid))
    node.set("pose", _pose(image_file, position,
                            {"w": 2**-.5, "x": 0., "y": -2**-.5, "z": 0.} if pinhole else None))
    if timestamp is not None:
        date_time = lib.StructureNode(image_file)
        date_time.set("dateTimeValue", lib.FloatNode(image_file, float(timestamp)))
        date_time.set("isAtomicClockReferenced", lib.IntegerNode(image_file, 1))
        node.set("acquisitionDateTime", date_time)
    representation = lib.StructureNode(image_file)
    node.set("pinholeRepresentation" if pinhole else "sphericalRepresentation", representation)
    width, height = (128, 64) if labelled else (16, 8)
    representation.set("imageWidth", lib.IntegerNode(image_file, width))
    representation.set("imageHeight", lib.IntegerNode(image_file, height))
    representation.set("pixelWidth", lib.FloatNode(image_file, 1. if pinhole else 2*np.pi/width))
    representation.set("pixelHeight", lib.FloatNode(image_file, 1. if pinhole else np.pi/height))
    if pinhole:
        representation.set("focalLength", lib.FloatNode(image_file, 64.))
        representation.set("principalPointX", lib.FloatNode(image_file, width/2))
        representation.set("principalPointY", lib.FloatNode(image_file, height/2))
    payload = _png_bytes(position if labelled else None, pinhole)
    blob = lib.BlobNode(image_file, len(payload))
    representation.set("pngImage", blob)
    blob.write(payload, 0, len(payload))


def _room_points(timestamps=False):
    xy = np.linspace(-6., 6., 18)
    z = np.linspace(-2., 4., 18)
    horizontal_a, horizontal_b = np.meshgrid(xy, xy)
    vertical_a, vertical_b = np.meshgrid(xy, z)
    faces = [np.column_stack((np.full(vertical_a.size, -6.), vertical_a.ravel(), vertical_b.ravel())),
             np.column_stack((np.full(vertical_a.size, 6.), vertical_a.ravel(), vertical_b.ravel())),
             np.column_stack((vertical_a.ravel(), np.full(vertical_a.size, -6.), vertical_b.ravel())),
             np.column_stack((vertical_a.ravel(), np.full(vertical_a.size, 6.), vertical_b.ravel())),
             np.column_stack((horizontal_a.ravel(), horizontal_b.ravel(), np.full(horizontal_a.size, -2.))),
             np.column_stack((horizontal_a.ravel(), horizontal_b.ravel(), np.full(horizontal_a.size, 4.)))]
    box_a, box_b = np.meshgrid(np.linspace(-.9, .9, 32), np.linspace(-.5, 1.5, 32))
    box = np.column_stack((np.full(box_a.size, 1.5), box_a.ravel(), box_b.ravel()))
    points = np.vstack([*faces, box])
    values = {"cartesianX": points[:, 0], "cartesianY": points[:, 1], "cartesianZ": points[:, 2]}
    if timestamps:
        # Ceiling observations are deliberately outside the image time window.
        values["timeStamp"] = np.where(np.isclose(points[:, 2], 4.), 100., 0.)
    return values


def _write_timed_scan(writer, data):
    """Write the small fixture scan with E57's optional per-point time field."""
    image = writer.image_file
    points = np.column_stack((data["cartesianX"], data["cartesianY"], data["cartesianZ"]))
    node = lib.StructureNode(image)
    writer.root["data3D"].append(node)
    node.set("guid", lib.StringNode(image, "{%s}" % uuid4()))
    node.set("name", lib.StringNode(image, "mobile-room"))
    pose = _pose(image, (0., 0., 0.))
    node.set("pose", pose)
    bounds = lib.StructureNode(image)
    for axis, low, high in zip("xyz", points.min(axis=0), points.max(axis=0)):
        bounds.set(axis+"Minimum", lib.FloatNode(image, float(low)))
        bounds.set(axis+"Maximum", lib.FloatNode(image, float(high)))
    node.set("cartesianBounds", bounds)
    prototype = lib.StructureNode(image)
    for name, values in (("cartesianX", data["cartesianX"]), ("cartesianY", data["cartesianY"]),
                         ("cartesianZ", data["cartesianZ"]), ("timeStamp", data["timeStamp"])):
        prototype.set(name, lib.FloatNode(image, float(values.mean()), lib.E57_DOUBLE,
                                          float(values.min()), float(values.max())))
    vector = lib.CompressedVectorNode(image, prototype, lib.VectorNode(image, True))
    node.set("points", vector)
    arrays = {name: np.ascontiguousarray(values, np.float64) for name, values in data.items()}
    buffers = lib.VectorSourceDestBuffer()
    for name, values in arrays.items():
        buffers.append(lib.SourceDestBuffer(image, name, values, len(values), True, True))
    stream = vector.writer(buffers)
    try:
        stream.write(len(points))
    finally:
        stream.close()


def write_slam_e57(path, associated=False, camera_count=50, timestamps=False, labelled_images=False, pinhole=False):
    """Write one identity block and a configurable posed camera trajectory."""
    writer = pye57.E57(str(path), "w")
    points = _room_points(timestamps)
    if timestamps:
        _write_timed_scan(writer, points)
    else:
        writer.write_scan_raw(points, name="mobile-room", translation=np.zeros(3))
    guid = writer.data3d[0]["guid"].value() if associated else None
    for index, x in enumerate(np.linspace(-5., 5., camera_count), 1):
        _image(writer, index, (x, 0., .6), guid, timestamp=0. if timestamps else None,
               labelled=labelled_images, pinhole=pinhole)
    writer.close()
    return path


def room_labels(points):
    """Return expected class IDs for points returned by the synthetic room."""
    points = np.asarray(points, np.float64)
    labels = np.full(len(points), "wall", dtype="U7")
    labels[np.isclose(points[:, 2], -2.)] = "floor"
    labels[np.isclose(points[:, 2], 4.)] = "ceiling"
    labels[(np.isclose(points[:, 0], 1.5)) & (np.abs(points[:, 1]) <= .9)
           & (points[:, 2] >= -.5) & (points[:, 2] <= 1.5)] = "box"
    return labels


def _pseudo_points():
    theta, phi = np.meshgrid(np.linspace(-np.pi, np.pi, 32, endpoint=False), np.linspace(-1.1, 1.1, 16))
    rays = np.column_stack((np.cos(phi).ravel()*np.cos(theta).ravel(),
                            np.cos(phi).ravel()*np.sin(theta).ravel(), np.sin(phi).ravel()))
    points = np.vstack((rays*2., rays*9.))
    return {"cartesianX": points[:, 0], "cartesianY": points[:, 1], "cartesianZ": points[:, 2]}


def write_pseudo_setup_e57(path):
    """Write panorama-like blocks whose radial duplicate layers fail rule d."""
    writer = pye57.E57(str(path), "w")
    positions = ((20., 0., 0.), (-20., 0., 0.))
    for index, position in enumerate(positions):
        writer.write_scan_raw(_pseudo_points(), name=f"pseudo-{index + 1}", translation=np.array(position))
    for index, position in enumerate(positions, 1):
        _image(writer, index, position, writer.data3d[index-1]["guid"].value())
    writer.close()
    return path
