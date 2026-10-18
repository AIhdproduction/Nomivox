"""Scan-local mirror/glass masks and depth planes for the owner's class-68 rule."""
from dataclasses import asdict, dataclass
from importlib.metadata import version
import logging
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes, binary_erosion, label, distance_transform_edt
from scipy.spatial import ConvexHull, QhullError, cKDTree
from e57cls.files import atomic_json, file_hash
from e57cls.geometry.cameras import Pose, plan_views
from e57cls.geometry.images import render_view
from e57cls.geometry.planes import fit_plane
from e57cls.seg.backend import PromptClass
from e57cls.seg.cache import segment_view
from e57cls.mirror.evidence import DirectIndex, Plane, Registry
from e57cls.pipeline.control import Cancelled


LOG = logging.getLogger(__name__)


@dataclass
class SurfaceMask:
    kind: str
    camera: object
    pose: Pose
    labels: np.ndarray
    planes: dict
    regularized_regions: set = None

    def region_ids(self, uv):
        h,w = self.labels.shape
        finite = np.isfinite(uv).all(axis=1)
        pixels = np.floor(np.where(np.isfinite(uv),uv,0)*[w/self.camera.width,h/self.camera.height]).astype(np.int64)
        valid = finite & ((pixels >= 0) & (pixels < [w,h])).all(axis=1)
        result = np.zeros(len(uv),np.int32)
        result[valid] = self.labels[pixels[valid,1],pixels[valid,0]]
        return result

    def states(self, origin, world, tolerance):
        uv,_,valid = self.camera.project(self.pose.from_world(world))
        regions = self.region_ids(uv)
        regions[~valid] = 0
        result = np.zeros(len(world),np.uint8)
        for region in np.unique(regions):
            if region == 0:
                continue
            ids = np.flatnonzero(regions == region)
            plane = self.planes.get(int(region))
            if plane is None:
                result[ids] = 4
                continue
            # The raw mask always keeps its original extent, even when an
            # accepted form is narrower in a few imperfect mask pixels.
            # Per-scan rays and the independently displaced camera both participate.
            raw_outline = getattr(plane,'raw_outline',None)
            gate = plane if raw_outline is None else Plane(plane.center,plane.normal,plane.basis,raw_outline,plane.frames,plane.quality)
            _,states = Registry([gate]).rays(origin,world[ids],tolerance)
            states[states == 4] = 2
            result[ids] = states
        # A regularized outline is an additional gate, never a replacement for
        # pixels which the original camera mask already accepted.  It is only
        # attached to mirror/glass regions after their form passed validation.
        for region in self.regularized_regions or ():
            plane = self.planes.get(int(region))
            if plane is None:
                continue
            _, states = Registry([plane]).rays(origin, world, tolerance)
            states[states == 4] = 2
            rank = np.array([0, 2, 3, 0, 1], np.uint8)
            take = rank[states] > rank[result]
            result[take] = states[take]
        return result


class MirrorAnalysis:
    def __init__(self,directory,config):
        self.directory,self.config = Path(directory),config
        self.directory.mkdir(parents=True,exist_ok=True)
        self.registry,self.images,self.paths = Registry(),{},{}
        self.report = {'policy':'masked_depth','state_labels':{'0':'outside_or_in_front','1':'on_surface',
                       '2':'behind_surface_inside_own_mask','3':'directly_seen_by_another_scan',
                       '4':'mask_without_reliable_plane'},
                       'images':[],'scans':{}}

    def membership(self,image_index,uv):
        result = np.zeros(len(uv),bool)
        for surface in self.images.get(image_index,[]):
            result |= surface.region_ids(uv)>0
        return result

    def states(self,world,scan,origin,image_indices=()):
        result = np.zeros(len(world),np.uint8)
        rank = np.array([0,2,3,0,1],np.uint8)
        # Only images explicitly associated with this scan can mark its points.
        for index in image_indices:
            for surface in self.images.get(index,[]):
                states = surface.states(origin,world,self.config['mirror_plane_tolerance'])
                take = rank[states]>rank[result]
                result[take] = states[take]
        return result

    def output(self,store,start,stop,valid,world,codes):
        data = np.load(self.paths[store.meta['scan_index']],mmap_mode='r',allow_pickle=False)
        try:
            states = np.array(data[start:stop][valid])
        finally:
            data._mmap.close()
        codes[states == 2] = 68
        return codes,states

    def expand_same_scan_reflections(self, world, states, origin, image_indices, tree, reflected):
        """Fill only local holes behind the *same* finite reflecting plane.

        A candidate needs a proven class-68 neighbour from its own scan and both
        rays must pass through one identical fitted plane.  This never transfers
        reflection evidence between scanner positions or across unrelated planes.
        """
        candidates = np.flatnonzero(states == 0)
        if not len(candidates) or tree is None:
            return 0
        distance, nearest = tree.query(world[candidates], distance_upper_bound=self.config['reflection_neighbor_radius'])
        candidates, nearest = candidates[np.isfinite(distance)], nearest[np.isfinite(distance)]
        if not len(candidates):
            return 0
        accepted = np.zeros(len(candidates), bool)
        for image_index in image_indices:
            for surface in self.images.get(image_index, []):
                for plane in surface.planes.values():
                    registry = Registry([plane])
                    _, target = registry.rays(origin, world[candidates], self.config['mirror_plane_tolerance'])
                    possible = (target == 4) & ~accepted
                    if not possible.any():
                        continue
                    _, source = registry.rays(origin, reflected[nearest[possible]], self.config['mirror_plane_tolerance'])
                    ids = np.flatnonzero(possible)
                    accepted[ids[source == 4]] = True
        states[candidates[accepted]] = 2
        return int(accepted.sum())

    def close(self):
        self.images.clear()


def agreed_probability(scores, agreement):
    """Require several distinct text prompts to support the same surface pixel."""
    stacked = np.asarray(scores)
    required = min(agreement,len(stacked))
    return np.partition(stacked,len(stacked)-required,axis=0)[-required]


def reflection_prompt_classes(config, preset):
    """Return the stable per-phrase mirror prompt identities used by the cache."""
    kinds = ['mirror_mask', 'reflective_mask']+(['glass_mask'] if config['glass_analysis'] else [])
    auxiliary = {c['id']: c for c in preset['auxiliary']}
    fallback = {
        'glass_mask': ['window glass','glass window pane','glass door','glass facade'],
        'reflective_mask': ['reflective surface','still water surface','polished metal'],
    }
    prompts = [PromptClass(kind,tuple(auxiliary.get(kind,{}).get('prompts',fallback.get(kind,['mirror']))),
                           config['instance_confidence']) for kind in kinds]
    phrases = [PromptClass(f'{cls.id}_{i}', (phrase,), cls.instance_threshold)
               for cls in prompts for i, phrase in enumerate(dict.fromkeys(cls.prompts))]
    return kinds, prompts, phrases


def make_masks(reader,item,config,image_cache,mask_cache,model_hash,factory,control,prompts,assets,progress_context=None):
    camera,image,image_mask,image_hash,mask_hash = assets(reader,item,image_cache)
    width = min(camera.width,config['mirror_mask_width'])
    height = max(1,round(camera.height*width/camera.width))
    phrases = [PromptClass(f'{cls.id}_{i}',(phrase,),cls.instance_threshold)
               for cls in prompts for i,phrase in enumerate(dict.fromkeys(cls.prompts))]
    probability = np.zeros((len(phrases),height,width),np.float32)
    # In the shared SAM mode these are deliberately the semantic views.  The
    # higher-resolution mirror masks are an accepted result change and make
    # every call below a cache-only read prepared by the ``sam`` phase.
    shared_sam = config.get('_shared_sam', False)
    view_size = config['view_size'] if shared_sam else config['mirror_view_size']
    scale = config['image_scale'] if shared_sam else config['mirror_image_scale']
    views = plan_views(camera,config['profile'],view_size,config['view_fov']) or [None]
    for view_index,view in enumerate(views):
        control.checkpoint()
        rendered,valid = render_view(image,image_mask,camera,view) if view else (image,image_mask>0 if image_mask is not None else np.ones(image.shape[:2],bool))
        info = asdict(view) if view else {'kind':camera.kind,'params':camera.params}
        info.update(mask_sha256=mask_hash,torch_version=version('torch'))
        scores,cached = segment_view(rendered,valid,phrases,factory,mask_cache,image_hash,info,model_hash,
                                     scale,config['tile_size'],config['tile_overlap'],control.checkpoint,
                                     cache_only=shared_sam)
        sh,sw = scores[0].shape
        size = [view.size,view.size] if view else [camera.width,camera.height]
        for row in range(0,height,32):
            x,y = np.meshgrid((np.arange(width)+.5)*camera.width/width,(np.arange(row,min(row+32,height))+.5)*camera.height/height)
            native = np.column_stack((x.ravel(),y.ravel()))
            uv,_,good = view.project(camera.rays(native)) if view else camera.project(camera.rays(native))
            pixels = np.floor(np.where(np.isfinite(uv),uv,0)*[sw,sh]/size).astype(np.int64)
            pixels[:,0] = pixels[:,0].clip(0,sw-1)
            pixels[:,1] = pixels[:,1].clip(0,sh-1)
            if image_mask is not None:
                nxy = np.floor(native).astype(int)
                good &= image_mask[nxy[:,1],nxy[:,0]]>0
            for k,score in enumerate(scores):
                sampled = np.where(good,score[pixels[:,1],pixels[:,0]],0)
                part = probability[k,row:row+32]
                np.maximum(part,sampled.reshape(part.shape),out=part)
        context = progress_context or {'image_number':1,'image_count':1}
        fraction = .9*(context['image_number']-1+(view_index+1)/len(views))/context['image_count']
        control.event('progress',phase='mirror_masks',image=item['index'],view=view_index+1,
                      view_count=len(views),fraction=fraction,cache=cached,**context)
    preview = np.array(Image.fromarray(image).resize((width,height),Image.Resampling.LANCZOS))
    combined,start = [],0
    for cls in prompts:
        stop = start+len(dict.fromkeys(cls.prompts))
        combined.append(agreed_probability(probability[start:stop],config['reflection_prompt_agreement']))
        start = stop
    return camera,Pose.from_dict(item['metadata']['pose']),np.asarray(combined)>=config['mirror_mask_confidence'],preview


def components(mask,min_pixels,wrap=False):
    labels,count = label(mask)
    if wrap:
        # Panorama seam pixels belong to the same component without filling mask holes.
        for a,b in zip(labels[:,0],labels[:,-1]):
            if a and b and a != b:
                labels[labels == b] = a
    sizes = np.bincount(labels.ravel())
    keep = sizes>=min_pixels
    keep[0] = False
    labels[~keep[labels]] = 0
    return labels


def _polygon_area(points):
    return .5*abs(np.dot(points[:, 0], np.roll(points[:, 1], -1))-np.dot(points[:, 1], np.roll(points[:, 0], -1)))


def _shape_measurements(shape):
    """Return the shortest full extent and area of a finite regular shape."""
    try:
        kind = shape['kind']
        center = np.asarray(shape['center'], np.float64)
        rotation = np.asarray(shape['rotation'], np.float64)
        if center.shape != (2,) or rotation.shape != (2, 2):
            return None
        if not np.isfinite(center).all() or not np.isfinite(rotation).all():
            return None
        if kind == 'rectangle':
            radii = np.asarray(shape['half'], np.float64)
            area = 4*np.prod(radii)
        elif kind in ('ellipse', 'circle'):
            radii = np.asarray(shape['axes'], np.float64)
            area = np.pi*np.prod(radii)
        else:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    if radii.shape != (2,) or not np.isfinite(radii).all() or (radii <= 0).any() or not np.isfinite(area):
        return None
    return 2*float(radii.min()), float(area)


def _valid_shape(shape, min_extent=0., min_area=0.):
    measurements = _shape_measurements(shape)
    return measurements is not None and measurements[0] >= min_extent and measurements[1] >= min_area


def _shape_outline(shape, vertices=64):
    if shape['kind'] == 'rectangle':
        local = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]], np.float64)*shape['half']
    else:
        angle = np.linspace(0, 2*np.pi, vertices, endpoint=False)
        local = np.column_stack((np.cos(angle), np.sin(angle)))*shape['axes']
    return shape['center']+local@shape['rotation'].T


def _contains_shape(shape, points):
    local = (points-shape['center'])@shape['rotation']
    if shape['kind'] == 'rectangle':
        return (np.abs(local) <= shape['half']+1e-9).all(axis=1)
    return ((local/shape['axes'])**2).sum(axis=1) <= 1+1e-9


def _minimum_rectangle(points, hull):
    vertices = points[hull.vertices]
    best = None
    for start, stop in zip(vertices, np.roll(vertices, -1, axis=0)):
        edge = stop-start
        length = np.linalg.norm(edge)
        if length < 1e-10:
            continue
        rotation = np.array([[edge[0], -edge[1]], [edge[1], edge[0]]], np.float64)/length
        projected = points@rotation
        low, high = projected.min(axis=0), projected.max(axis=0)
        candidate = {'kind':'rectangle', 'center':(low+high)/2@rotation.T, 'rotation':rotation,
                     'half':(high-low)/2}
        area = 4*np.prod(candidate['half'])
        if best is None or area < best[0]:
            best = area,candidate
    return None if best is None else best[1]


def _shape_iou(shape, occupied, cell=.01, maximum_cells=2_000_000):
    if not _valid_shape(shape):
        return None
    outline = _shape_outline(shape)
    low = np.floor(outline.min(axis=0)/cell).astype(np.int64)
    high = np.ceil(outline.max(axis=0)/cell).astype(np.int64)
    width = high-low+1
    if np.prod(width) > maximum_cells:
        return None
    yy,xx = np.mgrid[low[1]:high[1]+1, low[0]:high[0]+1]
    pixels = np.column_stack(((xx.ravel()+.5)*cell,(yy.ravel()+.5)*cell))
    candidate = _contains_shape(shape,pixels)
    count = int(candidate.sum())
    overlap = int(_contains_shape(shape,occupied).sum())
    return overlap/max(1, len(occupied)+count-overlap),count/max(1,len(occupied))


def _regular_shape(points, control, occupied=None):
    """Fit rectangle, ellipse and circle candidates to a 1 cm plane raster."""
    cell = .01
    if occupied is None:
        cells = np.unique(np.floor(points/cell).astype(np.int64), axis=0)
        occupied = (cells+.5)*cell
    occupied = np.asarray(occupied, np.float64)
    occupied = occupied[np.isfinite(occupied).all(axis=1)]
    if len(occupied) < 8:
        return None, {'shape':'rejected:insufficient_area'}
    try:
        hull = ConvexHull(occupied)
    except (ValueError,QhullError):
        return None, {'shape':'rejected:degenerate_area'}
    hull_area = _polygon_area(occupied[hull.vertices])
    if hull_area <= 1e-8:
        return None, {'shape':'rejected:degenerate_area'}
    candidates = []
    rectangle = _minimum_rectangle(occupied,hull)
    if rectangle is not None and _valid_shape(rectangle):
        candidates.append(rectangle)
    center = occupied.mean(axis=0)
    covariance = np.cov((occupied-center).T, bias=True)
    if np.isfinite(covariance).all():
        values,vectors = np.linalg.eigh(covariance)
    else:
        values,vectors = np.array([0.,0.]),np.eye(2)
    if np.isfinite(values).all() and values[0] > 1e-10:
        rotation = vectors
        axes = 2*np.sqrt(values)
        scale = np.sqrt(hull_area/max(np.pi*np.prod(axes),1e-12))
        ellipse = {'kind':'ellipse','center':center,'rotation':rotation,'axes':axes*scale}
        radius = np.sqrt(hull_area/np.pi)
        circle = {'kind':'circle','center':center,'rotation':np.eye(2),'axes':np.array([radius,radius])}
        candidates.extend(candidate for candidate in (ellipse,circle) if _valid_shape(candidate))
    scored = []
    for candidate in candidates:
        control.checkpoint()
        score = _shape_iou(candidate,occupied,cell)
        if score is not None:
            scored.append((score[0],score[1],candidate))
    if not scored:
        return None, {'shape':'rejected:raster_too_large'}
    ellipse = next((item for item in scored if item[2]['kind'] == 'ellipse'),None)
    ratio = 0. if ellipse is None else min(ellipse[2]['axes'])/max(ellipse[2]['axes'])
    best = max(scored,key=lambda item:(item[0], item[2]['kind'] == 'circle' and ratio > .9))
    return best[2], {'shape':best[2]['kind'],'iou':float(best[0]),'growth':float(best[1]),'reference_area_m2':float(len(occupied)*cell*cell)}


def _mask_occupancy(points, camera, pose, plane, labels, region):
    """Sample the camera mask on a 1 cm plane raster without sparse pixel gaps."""
    cell = .01
    low = np.floor(points.min(axis=0)/cell).astype(np.int64)
    high = np.ceil(points.max(axis=0)/cell).astype(np.int64)
    width = high-low+1
    if np.prod(width) > 2_000_000:
        return None
    yy,xx = np.mgrid[low[1]:high[1]+1, low[0]:high[0]+1]
    local = np.column_stack(((xx.ravel()+.5)*cell,(yy.ravel()+.5)*cell))
    world = plane.center+local@plane.basis
    uv,_,visible = camera.project(pose.from_world(world))
    h,w = labels.shape
    pixels = np.floor(np.where(np.isfinite(uv),uv,0)*[w/camera.width,h/camera.height]).astype(np.int64)
    pixels[:,0] = pixels[:,0].clip(0,w-1)
    pixels[:,1] = pixels[:,1].clip(0,h-1)
    return local[visible & (labels[pixels[:,1],pixels[:,0]] == region)]


def _plane_hits(points, origin, plane):
    directions = points-origin
    denominator = directions@plane.normal
    fraction = np.divide((plane.center-origin)@plane.normal,denominator,
                         out=np.full(len(points),np.inf),where=np.abs(denominator)>1e-10)
    valid = np.isfinite(fraction) & (fraction > 0) & (fraction <= 1.01)
    hits = origin+fraction[:,None]*directions
    return valid,(hits-plane.center)@plane.basis.T,(points-plane.center)@plane.normal


def _refine_rectangle(shape, points, origin, plane, tolerance, search, control):
    if not _valid_shape(shape) or not len(points) or search <= 0:
        return shape, [0.,0.,0.,0.]
    valid,hit,signed = _plane_hits(points,origin,plane)
    local = (hit-shape['center'])@shape['rotation']
    shifts = np.arange(-search,search+.005,.01)
    selected = []
    for axis,side in ((0,-1),(0,1),(1,-1),(1,1)):
        best, best_score = 0., -np.inf
        other = 1-axis
        for shift in shifts:
            boundary = shape['half'][axis]+shift
            along = side*local[:,axis]
            extent = shape['half'][other]+.03
            inner = valid & (np.abs(local[:,other]) <= extent) & (along >= boundary-.03) & (along <= boundary)
            outer = valid & (np.abs(local[:,other]) <= extent) & (along > boundary) & (along <= boundary+.03)
            if inner.sum() < 20 or outer.sum() < 20:
                continue
            score = (signed[inner] < -tolerance).mean()+(np.abs(signed[outer]) <= tolerance).mean()
            if score > best_score:
                best,best_score = float(shift),score
        selected.append(best)
        control.checkpoint()
    original_half = np.asarray(shape['half'],np.float64)
    # An inward shift may consume at most half of the original half-extent on
    # either edge.  This leaves every side at least half its original length
    # and prevents opposite edges from crossing.
    selected = np.asarray(selected,np.float64)
    selected[[0,1]] = np.maximum(selected[[0,1]],-original_half[0]/2)
    selected[[2,3]] = np.maximum(selected[[2,3]],-original_half[1]/2)
    low = -original_half-np.array([selected[0],selected[2]])
    high = original_half+np.array([selected[1],selected[3]])
    updated = {**shape,'center':shape['center']+((low+high)/2)@shape['rotation'].T,'half':(high-low)/2}
    return updated,selected.tolist()


def _refine_round(shape, points, origin, plane, tolerance, search, control):
    if not _valid_shape(shape) or not len(points) or search <= 0:
        return shape, 0.
    valid,hit,signed = _plane_hits(points,origin,plane)
    local = (hit-shape['center'])@shape['rotation']
    best,best_score = 0.,-np.inf
    for shift in np.arange(-search,search+.005,.01):
        axes = shape['axes']+shift
        if (axes <= .03).any():
            continue
        radius = np.sqrt(((local/axes)**2).sum(axis=1))
        inner = valid & (radius >= 1-.03/max(axes)) & (radius <= 1)
        outer = valid & (radius > 1) & (radius <= 1+.03/max(axes))
        if inner.sum() < 20 or outer.sum() < 20:
            continue
        score = (signed[inner] < -tolerance).mean()+(np.abs(signed[outer]) <= tolerance).mean()
        if score > best_score:
            best,best_score = float(shift),score
        control.checkpoint()
    return {**shape,'axes':shape['axes']+best},best


def _surface_fit_input(camera, mask, config):
    labels = components(mask,config['mirror_min_pixels'],camera.wraps)
    regions = [int(i) for i in np.unique(labels) if i]
    h,w = labels.shape
    # Fit from outside the mask, where the frame/wall returns are real geometry.
    # Interior reflected room points are deliberately not plane-fitting samples.
    distance,nearest = distance_transform_edt(labels == 0,return_indices=True)
    frame_labels = labels[nearest[0],nearest[1]]
    # Holes inside the SAM mask are not neighbouring wall samples.
    frame_labels[binary_fill_holes(labels>0)|(distance>config['mirror_frame_band'])] = 0
    return {'labels': labels, 'regions': regions, 'frame_labels': frame_labels,
            'samples': {i:np.empty((0,3),np.float64) for i in regions},
            'priorities': {i:np.empty(0) for i in regions},
            'rng': np.random.default_rng(config['seed']), 'depth_rng': np.random.default_rng(config['seed']+104729),
            'depth_samples': np.empty((0,3),np.float64), 'depth_priorities': np.empty(0)}


def _collect_surface_samples(store, camera, pose, states, control):
    """Collect all mask-type reservoirs during one ordered store traversal."""
    scan_pose = Pose.from_dict(store.meta['metadata']['pose'])
    for _,xyz,valid in store.chunks():
        control.checkpoint()
        world = scan_pose.to_world(xyz[valid])
        for state in states:
            values = np.concatenate((state['depth_samples'],world))
            keys = np.r_[state['depth_priorities'],state['depth_rng'].random(len(world))]
            take = np.argpartition(keys,200000)[:200000] if len(keys)>200000 else np.arange(len(keys))
            state['depth_samples'],state['depth_priorities'] = values[take],keys[take]
            labels,frame_labels = state['labels'],state['frame_labels']
            h,w = labels.shape
            uv,_,good = camera.project(pose.from_world(world))
            pixels = np.floor(np.where(np.isfinite(uv),uv,0)*[w/camera.width,h/camera.height]).astype(np.int64)
            pixels[:,0] = pixels[:,0].clip(0,w-1)
            pixels[:,1] = pixels[:,1].clip(0,h-1)
            groups = frame_labels[pixels[:,1],pixels[:,0]]
            groups[~good] = 0
            for region in state['regions']:
                selected = world[groups == region]
                if not len(selected):
                    continue
                values = np.concatenate((state['samples'][region],selected))
                keys = np.r_[state['priorities'][region],state['rng'].random(len(selected))]
                take = np.argpartition(keys,20000)[:20000] if len(keys)>20000 else np.arange(len(keys))
                state['samples'][region],state['priorities'][region] = values[take],keys[take]


def fit_surfaces(store,camera,pose,mask,config,control,_allow_expand=True,kind=None,_prepared=None):
    prepared = _prepared or _surface_fit_input(camera, mask, config)
    labels, regions = prepared['labels'], prepared['regions']
    if not regions:
        return labels,{},[]
    h, w = labels.shape
    if _prepared is None:
        _collect_surface_samples(store, camera, pose, [prepared], control)
    samples, depth_samples = prepared['samples'], prepared['depth_samples']
    scan_pose = Pose.from_dict(store.meta['metadata']['pose'])
    planes,details = {},[]
    for region in regions:
        selected = samples[region]
        fit = fit_plane(selected,config['mirror_fit_tolerance'],config['seed'],min_fraction=.5)
        detail = {'region':region,'mask_pixels':int((labels == region).sum()),'frame_samples':len(selected),
                  'frame_band_pixels':config['mirror_frame_band'],'fit_source':'outer_mask_edge_and_adjacent_wall',
                  'interior_samples':0,'accepted':False}
        details.append(detail)
        if fit is None or fit['residual_p95']>config['mirror_fit_tolerance'] or len(fit['points'])<30:
            detail['reason']='insufficient_frame_plane'
            continue
        if (scan_pose.translation-fit['center'])@fit['normal']<0:
            fit['normal'] *= -1
        region_mask = labels == region
        by,bx = np.nonzero(region_mask & ~binary_erosion(region_mask))
        boundary_uv = np.column_stack(((bx+.5)*camera.width/w,(by+.5)*camera.height/h))
        rays = camera.rays(boundary_uv)@pose.rotation.T
        denominator = rays@fit['normal']
        distances = np.divide((fit['center']-pose.translation)@fit['normal'],denominator,
                              out=np.full(len(rays),np.inf),where=np.abs(denominator)>1e-8)
        if not np.isfinite(distances).all() or (distances<=0).any() or distances.max()>config['mirror_max_range']:
            detail['reason']='plane_does_not_face_mask'
            continue
        intersections = pose.translation+distances[:,None]*rays
        outline = (intersections-fit['center'])@fit['basis'].T
        support = (fit['points']-fit['center'])@fit['basis'].T
        coverage = np.minimum(np.ptp(support,axis=0)/np.maximum(np.ptp(outline,axis=0),1e-9),1)
        if coverage.min()<.7:
            detail['reason']='frame_support_does_not_span_mask'
            continue
        try:
            raw_outline = outline[ConvexHull(outline).vertices]
            plane = Plane(fit['center'],fit['normal'],fit['basis'],raw_outline,fit['points'],True)
        except (ValueError,QhullError):
            detail['reason']='degenerate_plane_extent'
            continue
        regular = (config.get('mirror_shape_regularize',False) and kind in config.get('mirror_shape_kinds',()))
        if regular:
            # Project every labelled pixel, not merely the boundary, into the
            # fitted plane.  The 1 cm raster below is the form reference area.
            py,px = np.nonzero(region_mask)
            region_uv = np.column_stack(((px+.5)*camera.width/w,(py+.5)*camera.height/h))
            rays = camera.rays(region_uv)@pose.rotation.T
            denominator = rays@fit['normal']
            distances = np.divide((fit['center']-pose.translation)@fit['normal'],denominator,
                                  out=np.full(len(rays),np.inf),where=np.abs(denominator)>1e-8)
            points = pose.translation+distances[:,None]*rays
            plane_points = (points-fit['center'])@fit['basis'].T
            occupied = _mask_occupancy(plane_points,camera,pose,plane,labels,region)
            if occupied is None:
                shape,shape_detail = None,{'shape':'rejected:raster_too_large'}
            else:
                shape,shape_detail = _regular_shape(plane_points,control,occupied)
            if shape is None:
                detail.update(shape_detail)
            elif shape_detail['iou'] < config['mirror_shape_min_iou']:
                detail.update(shape=f"rejected:iou<{config['mirror_shape_min_iou']}",iou=shape_detail['iou'],
                              growth=shape_detail['growth'])
            elif shape_detail['growth'] > config['mirror_shape_max_growth']:
                detail.update(shape=f"rejected:growth>{config['mirror_shape_max_growth']}",iou=shape_detail['iou'],
                              growth=shape_detail['growth'])
            else:
                if not _valid_shape(shape,.02,.005):
                    detail.update(shape='rejected:degenerate')
                else:
                    if shape['kind'] == 'rectangle':
                        shape,edge_shift = _refine_rectangle(shape,depth_samples,scan_pose.translation,plane,
                                                              config['mirror_plane_tolerance'],config['mirror_edge_search'],control)
                    else:
                        shape,edge_shift = _refine_round(shape,depth_samples,scan_pose.translation,plane,
                                                          config['mirror_plane_tolerance'],config['mirror_edge_search'],control)
                    if not _valid_shape(shape,.02,.005):
                        detail.update(shape='rejected:degenerate')
                    else:
                        final = _shape_iou(shape,occupied)
                        if final is None or final[1] > config['mirror_shape_max_growth']:
                            detail.update(shape=f"rejected:growth>{config['mirror_shape_max_growth']}",
                                          iou=shape_detail['iou'],growth=shape_detail['growth'])
                        else:
                            regular_outline = _shape_outline(shape)
                            try:
                                regular_plane = Plane(fit['center'],fit['normal'],fit['basis'],regular_outline,fit['points'],True)
                            except (ValueError,QhullError):
                                try:
                                    plane = Plane(fit['center'],fit['normal'],fit['basis'],raw_outline,fit['points'],True)
                                except (ValueError,QhullError):
                                    detail['reason']='degenerate_plane_extent'
                                    continue
                                detail.update(shape='rejected:degenerate')
                            else:
                                plane = regular_plane
                                plane.regularized = True
                                plane.raw_outline = raw_outline
                                detail.update(shape=shape['kind'],iou=float(final[0]),growth=float(final[1]),
                                              reference_area_m2=shape_detail['reference_area_m2'],edge_shift_m=edge_shift)
        planes[region] = plane
        detail.update(accepted=True,inlier_fraction=fit['inlier_fraction'],residual_p95=fit['residual_p95'],coverage=coverage.tolist())
        detail.update(center=plane.center.tolist(),normal=plane.normal.tolist())
    if _allow_expand and len(planes)<len(regions) and config['mirror_frame_band_max']>config['mirror_frame_band']:
        wider = {**config,'mirror_frame_band':config['mirror_frame_band_max']}
        _,wall_planes,wall_details = fit_surfaces(store,camera,pose,mask,wider,control,False,kind)
        by_region = {item['region']:item for item in wall_details}
        for i,detail in enumerate(details):
            region = detail['region']
            if region not in planes and region in wall_planes:
                planes[region] = wall_planes[region]
                details[i] = {**by_region[region],'initial_frame_rejection':detail.get('reason')}
            elif region not in planes:
                detail['expanded_frame_rejection'] = by_region[region].get('reason')
    return labels,planes,details


def protect_direct_multiscan_points(analysis, stores, inspection, config, control):
    """Keep a candidate if another scan observed it directly.

    Each reflection candidate already has an image mask and a finite fitted
    plane.  A genuine object behind a window, however, can satisfy those two
    tests as well.  Build an exact, disk-backed index of points that were *not*
    marked as reflections in their own scan, then preserve candidates which a
    different scanner position measured at the same coordinates.  The index is
    deliberately exact (not voxelized), so the confirmation radius retains its
    intended geometric meaning.
    """
    if len(stores) < 2:
        return {store.meta['scan_index']: 0 for store in stores}
    # Avoid constructing a large temporary index in projects where the image /
    # plane stage did not produce any reflection candidate.
    if not any(analysis.report['scans'].get(str(store.meta['scan_index']), {}).get('counts', [0, 0, 0])[2]
               for store in stores):
        return {store.meta['scan_index']: 0 for store in stores}
    index_path = analysis.directory/'direct_multiscan.sqlite'
    index_path.unlink(missing_ok=True)
    origin_value = config.get('voxel_origin')
    if origin_value is None:
        origin_value = Pose.from_dict(stores[0].meta['metadata']['pose']).translation
    origin = np.asarray(origin_value, np.float64)
    direct = DirectIndex(index_path, origin)
    retained = {store.meta['scan_index']: 0 for store in stores}
    try:
        # A point already labelled class 68 by its own image/plane evidence is
        # never used as confirmation for a different scan's candidate.
        for store in stores:
            scan = store.meta['scan_index']
            pose = Pose.from_dict(store.meta['metadata']['pose'])
            states = np.load(analysis.paths[scan], mmap_mode='r', allow_pickle=False)
            try:
                for start,xyz,valid in _state_chunks(store, config):
                    control.checkpoint()
                    local = states[start:start+len(xyz)][valid]
                    keep = (local == 0) | (local == 1)
                    if np.any(keep):
                        direct.add(pose.to_world(xyz[valid][keep]), scan, commit=False)
            finally:
                states._mmap.close()
        # Only a scan other than the source scan can protect a candidate.  A
        # state of 3 is deliberately not exported as class 68.
        for store in stores:
            scan = store.meta['scan_index']
            pose = Pose.from_dict(store.meta['metadata']['pose'])
            writable = np.load(analysis.paths[scan], mmap_mode='r+', allow_pickle=False)
            try:
                for start,xyz,valid in _state_chunks(store, config):
                    control.checkpoint()
                    block = np.array(writable[start:start+len(xyz)], copy=True)
                    local = block[valid]
                    candidates = local == 2
                    if np.any(candidates):
                        world = pose.to_world(xyz[valid][candidates])
                        confirmed = direct.nearby(world, config['mirror_confirmation_radius'], exclude_scan=scan)
                        positions = np.flatnonzero(candidates)[confirmed]
                        local[positions] = 3
                        retained[scan] += len(positions)
                    block[valid] = local
                    writable[start:start+len(xyz)] = block
                writable.flush()
            finally:
                writable._mmap.close()
    finally:
        direct.close()
        # It is an exact, temporary disk index.  Keeping it after all states
        # have been checked only wastes considerable disk space on large jobs.
        index_path.unlink(missing_ok=True)
    return retained


def _state_chunks(store, config):
    """Use all valid points when an original-density export needs their state."""
    if config.get('output_density') == 'original' and hasattr(store, 'full_chunks'):
        yield from store.full_chunks()
    else:
        yield from store.chunks()


def fit_surfaces_many(store, camera, pose, masks, config, control):
    """Fit all mask kinds of one image with one primary store pass.

    Each state owns the same seeded generators that ``fit_surfaces`` used on
    its own, so reservoir membership and downstream planes remain unchanged.
    Wider-frame fallback stays deliberately individual in ``fit_surfaces``.
    """
    prepared = {kind: _surface_fit_input(camera, mask, config) for kind, mask in masks.items()}
    active = [state for state in prepared.values() if state['regions']]
    if active:
        _collect_surface_samples(store, camera, pose, active, control)
    return {kind: fit_surfaces(store, camera, pose, masks[kind], config, control, kind=kind, _prepared=state)
            for kind, state in prepared.items()}


def fit_scan_surfaces(reader, store, inspection, config, analysis, image_cache, mask_cache, model_hash,
                      factory, control, preset, assets, prepared_failures=None, image_numbers=None):
    """Consume cached masks and fit the reflection surfaces of one scan.

    The resulting ``analysis.images`` entries are all ``classify_scan`` needs.
    Registry merging and the state-file finalization below are diagnostic/export
    work and deliberately remain outside this per-scan boundary.
    """
    # Keep the narrow mirror and glass prompts separate from broadly reflective
    # materials.  Each candidate still needs its own image mask and a confirmed
    # plane before it can affect a point, which prevents a detected object from
    # turning unrelated points into reflections.
    kinds, prompts, _ = reflection_prompt_classes(config, preset)
    prepared_failures = prepared_failures or {}
    output = Path(config['output'])/'diagnostics'/'reflections'
    output.mkdir(parents=True,exist_ok=True)
    image_numbers = image_numbers or {}
    image_count = len(image_numbers) or sum(1 for index in inspection['scans'][store.meta['scan_index']-1]['image_indices']
                                            if inspection['images'][index-1]['pose_status']=='valid'
                                            and any(r!='visualReferenceRepresentation' for r in inspection['images'][index-1]['representations']))
    scan = store.meta['scan_index']
    for index in inspection['scans'][scan-1]['image_indices']:
            item = inspection['images'][index-1]
            if item['pose_status']!='valid' or not any(r!='visualReferenceRepresentation' for r in item['representations']):
                continue
            context = dict(image_number=image_numbers.get(index, 1), image_count=image_count,
                           scan=scan, scan_count=len(inspection['scans']))
            analysis.images[index] = []

            def image_failed(kind, error):
                detail = str(error)
                LOG.error('Mirror image processing failed (scan=%s image=%s mask_kind=%s): %s',scan,index,kind,detail)
                control.event('warning',code='mirror_image_failed',scan=scan,image=index,mask_kind=kind,detail=detail)
                analysis.report['images'].append({'scan':scan,'image':index,'kind':kind,'regions':[],
                                                  'failed':True,'error':detail})

            if index in prepared_failures:
                for kind in kinds:
                    image_failed(kind, RuntimeError(prepared_failures[index]))
                continue

            try:
                camera,pose,masks,preview = make_masks(reader,item,config,image_cache,mask_cache,model_hash,factory,control,prompts,assets,context)
            except Cancelled:
                raise
            except Exception as error:
                if config['strict']:
                    raise
                for kind in kinds:
                    image_failed(kind,error)
                continue
            try:
                fitted = fit_surfaces_many(store, camera, pose, dict(zip(kinds, masks)), config, control)
            except Cancelled:
                raise
            except Exception:
                # Keep the established per-kind failure isolation if a shared
                # preparation itself cannot be completed.
                fitted = {}
            for kind,mask in zip(kinds,masks):
                surface = None
                try:
                    labels,planes,details = fitted.pop(kind) if kind in fitted else fit_surfaces(store,camera,pose,mask,config,control,kind=kind)
                    regularized = {detail['region'] for detail in details if detail.get('shape') in ('rectangle','ellipse','circle')}
                    active = labels>0
                    Image.fromarray(active.astype(np.uint8)*255).save(output/f'image_{index:04d}_{kind}.png')
                    overlay = preview.copy()
                    color = np.array([255,0,255] if kind=='mirror_mask' else [0,220,255])
                    overlay[active] = np.rint(.5*overlay[active]+.5*color).astype(np.uint8)
                    overlay_image = Image.fromarray(overlay)
                    draw = ImageDraw.Draw(overlay_image)
                    for region in regularized:
                        plane = planes[region]
                        vertices = plane.center+plane.outline@plane.basis
                        uv,_,visible = camera.project(pose.from_world(vertices))
                        if visible.all():
                            scale = np.array([overlay.shape[1]/camera.width,overlay.shape[0]/camera.height])
                            points = [tuple(point) for point in uv*scale]
                            draw.line(points+[points[0]],fill=tuple(np.minimum(color.astype(int)+np.array([0,35,35]),255)),width=2)
                    overlay_image.save(output/f'image_{index:04d}_{kind}_overlay.jpg',quality=92)
                    if any(not d['accepted'] for d in details):
                        control.event('warning',code='reflection_plane_missing',scan=scan,image=index,mask_kind=kind)
                    atomic_json(output/f'image_{index:04d}_{kind}_planes.json',details)
                    for plane in planes.values():
                        # Registry is diagnostic only: it never marks points outside their own mask.
                        analysis.registry.add(plane,config['mirror_merge_angle'],config['mirror_merge_distance'])
                    surface = SurfaceMask(kind,camera,pose,labels,planes,regularized)
                    analysis.images[index].append(surface)
                    analysis.report['images'].append({'scan':scan,'image':index,'kind':kind,'regions':details})
                except Cancelled:
                    raise
                except Exception as error:
                    if config['strict']:
                        raise
                    if surface is not None:
                        analysis.images[index] = [candidate for candidate in analysis.images[index] if candidate is not surface]
                    image_failed(kind,error)


def finalize_states(analysis, stores, inspection, config, control):
    """Write export states after all scan-local surface fits are available."""
    atomic_json(analysis.directory/'registry.json',[p.serialize() for p in analysis.registry.planes])
    for number,store in enumerate(stores,1):
        scan = store.meta['scan_index']
        pose = Pose.from_dict(store.meta['metadata']['pose'])
        path = analysis.directory/f'states_{scan:04d}.npy'
        temp = path.with_suffix('.partial.npy')
        out = np.lib.format.open_memmap(temp,mode='w+',dtype=np.uint8,shape=(store.meta['record_count'],))
        counts = np.zeros(5,np.int64)
        try:
            for start,xyz,valid in _state_chunks(store, config):
                control.checkpoint()
                states = analysis.states(pose.to_world(xyz[valid]),scan,pose.translation,inspection['scans'][scan-1]['image_indices'])
                block = np.zeros(len(xyz),np.uint8)
                block[valid] = states
                out[start:start+len(xyz)] = block
                counts += np.bincount(states,minlength=5)
            out.flush()
        finally:
            out._mmap.close()
        # One bounded same-scan expansion repairs sparse holes beside already
        # proven reflection points.  It is intentionally non-recursive.
        initial = np.load(temp, mmap_mode='r', allow_pickle=False)
        reflected = []
        try:
            for start,xyz,valid in _state_chunks(store, config):
                block = initial[start:start+len(xyz)][valid]
                if np.any(block == 2):
                    reflected.append(pose.to_world(xyz[valid][block == 2]))
            reflected = np.concatenate(reflected) if reflected else np.empty((0,3), np.float64)
        finally:
            initial._mmap.close()
        expanded = 0
        if len(reflected):
            tree = cKDTree(reflected)
            writable = np.load(temp, mmap_mode='r+', allow_pickle=False)
            try:
                for start,xyz,valid in _state_chunks(store, config):
                    block = np.array(writable[start:start+len(xyz)], copy=True)
                    local = block[valid]
                    world = pose.to_world(xyz[valid])
                    expanded += analysis.expand_same_scan_reflections(world, local, pose.translation,
                                                                       inspection['scans'][scan-1]['image_indices'], tree, reflected)
                    block[valid] = local
                    writable[start:start+len(xyz)] = block
                writable.flush()
            finally:
                writable._mmap.close()
            counts = np.zeros(5, np.int64)
            verify = np.load(temp, mmap_mode='r', allow_pickle=False)
            try:
                for start in range(0, len(verify), 100000):
                    counts += np.bincount(verify[start:start+100000], minlength=5)
            finally:
                verify._mmap.close()
        temp.replace(path)
        analysis.paths[scan] = path
        analysis.report['scans'][str(scan)] = {'counts':counts.tolist(),'same_scan_neighbour_reflections':expanded,'sha256':file_hash(path)}
        control.event('progress',phase='mirror_evidence',scan=scan,scan_count=len(stores),fraction=.9+.1*number/len(stores))
    retained = protect_direct_multiscan_points(analysis, stores, inspection, config, control)
    # The direct check changes state 2 to state 3.  Re-count and fingerprint the
    # final state files so the report describes exactly what will be exported.
    for store in stores:
        scan = store.meta['scan_index']
        states = np.load(analysis.paths[scan], mmap_mode='r', allow_pickle=False)
        try:
            counts = np.zeros(5, np.int64)
            for start in range(0, len(states), 100000):
                counts += np.bincount(states[start:start+100000], minlength=5)
        finally:
            states._mmap.close()
        analysis.report['scans'][str(scan)].update(
            counts=counts.tolist(), directly_confirmed_by_other_scans=retained[scan], sha256=file_hash(analysis.paths[scan]))
    atomic_json(Path(config['output'])/'reflection_report.json',analysis.report)


def build(reader,stores,inspection,config,work,image_cache,mask_cache,model_hash,factory,control,preset,assets,
          prepared_failures=None):
    """Run the serial reflection pipeline used by the compatibility mode."""
    analysis = MirrorAnalysis(work/'mirrors',config)
    image_numbers = {item['index']: number for number, item in enumerate(
        (inspection['images'][index-1] for store in stores
         for index in inspection['scans'][store.meta['scan_index']-1]['image_indices']
         if inspection['images'][index-1]['pose_status']=='valid'
         and any(r!='visualReferenceRepresentation' for r in inspection['images'][index-1]['representations'])), 1)}
    for store in stores:
        fit_scan_surfaces(reader, store, inspection, config, analysis, image_cache, mask_cache, model_hash,
                          factory, control, preset, assets, prepared_failures, image_numbers)
    finalize_states(analysis, stores, inspection, config, control)
    return analysis
