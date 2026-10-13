"""Scan-local mirror/glass masks and depth planes for the owner's class-68 rule."""
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes, binary_erosion, label, distance_transform_edt
from scipy.spatial import ConvexHull, QhullError
from e57cls.files import atomic_json, file_hash
from e57cls.geometry.cameras import Pose, plan_views
from e57cls.geometry.images import render_view
from e57cls.geometry.planes import fit_plane
from e57cls.seg.backend import PromptClass
from e57cls.seg.cache import segment_view
from e57cls.mirror.evidence import Plane, Registry


@dataclass
class SurfaceMask:
    kind: str
    camera: object
    pose: Pose
    labels: np.ndarray
    planes: dict

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
            # The exact camera mask is the gate. A convex plane outline never enlarges it.
            # Per-scan rays and the independently displaced camera both participate.
            _,states = Registry([plane]).rays(origin,world[ids],tolerance)
            states[states == 4] = 2
            result[ids] = states
        return result


class MirrorAnalysis:
    def __init__(self,directory,config):
        self.directory,self.config = Path(directory),config
        self.directory.mkdir(parents=True,exist_ok=True)
        self.registry,self.images,self.paths = Registry(),{},{}
        self.report = {'policy':'masked_depth','state_labels':{'0':'outside_or_in_front','1':'on_surface',
                       '2':'behind_surface_inside_own_mask','3':'unused','4':'mask_without_reliable_plane'},
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

    def close(self):
        self.images.clear()


def agreed_probability(scores, agreement):
    """Require several distinct text prompts to support the same surface pixel."""
    stacked = np.asarray(scores)
    required = min(agreement,len(stacked))
    return np.partition(stacked,len(stacked)-required,axis=0)[-required]


def make_masks(reader,item,config,image_cache,mask_cache,model_hash,factory,control,prompts,assets,progress_context=None):
    camera,image,image_mask,image_hash,mask_hash = assets(reader,item,image_cache)
    width = min(camera.width,config['mirror_mask_width'])
    height = max(1,round(camera.height*width/camera.width))
    phrases = [PromptClass(f'{cls.id}_{i}',(phrase,),cls.instance_threshold)
               for cls in prompts for i,phrase in enumerate(dict.fromkeys(cls.prompts))]
    probability = np.zeros((len(phrases),height,width),np.float32)
    views = plan_views(camera,config['profile'],config['view_size'],config['view_fov']) or [None]
    for view_index,view in enumerate(views):
        control.checkpoint()
        rendered,valid = render_view(image,image_mask,camera,view) if view else (image,image_mask>0 if image_mask is not None else np.ones(image.shape[:2],bool))
        info = asdict(view) if view else {'kind':camera.kind,'params':camera.params}
        info.update(mask_sha256=mask_hash,torch_version=version('torch'))
        scores,cached = segment_view(rendered,valid,phrases,factory,mask_cache,image_hash,info,model_hash,
                                     config['image_scale'],config['tile_size'],config['tile_overlap'],control.checkpoint)
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


def fit_surfaces(store,camera,pose,mask,config,control,_allow_expand=True):
    labels = components(mask,config['mirror_min_pixels'],camera.wraps)
    regions = [int(i) for i in np.unique(labels) if i]
    if not regions:
        return labels,{},[]
    h,w = labels.shape
    # Fit from outside the mask, where the frame/wall returns are real geometry.
    # Interior reflected room points are deliberately not plane-fitting samples.
    distance,nearest = distance_transform_edt(labels == 0,return_indices=True)
    frame_labels = labels[nearest[0],nearest[1]]
    # Holes inside the SAM mask are not neighbouring wall samples.
    frame_labels[binary_fill_holes(labels>0)|(distance>config['mirror_frame_band'])] = 0
    samples = {i:np.empty((0,3),np.float64) for i in regions}
    priorities = {i:np.empty(0) for i in regions}
    rng = np.random.default_rng(config['seed'])
    scan_pose = Pose.from_dict(store.meta['metadata']['pose'])
    for _,xyz,valid in store.chunks():
        control.checkpoint()
        world = scan_pose.to_world(xyz[valid])
        uv,_,good = camera.project(pose.from_world(world))
        pixels = np.floor(np.where(np.isfinite(uv),uv,0)*[w/camera.width,h/camera.height]).astype(np.int64)
        pixels[:,0] = pixels[:,0].clip(0,w-1)
        pixels[:,1] = pixels[:,1].clip(0,h-1)
        groups = frame_labels[pixels[:,1],pixels[:,0]]
        groups[~good] = 0
        for region in regions:
            selected = world[groups == region]
            if not len(selected):
                continue
            values = np.concatenate((samples[region],selected))
            keys = np.r_[priorities[region],rng.random(len(selected))]
            take = np.argpartition(keys,20000)[:20000] if len(keys)>20000 else np.arange(len(keys))
            samples[region],priorities[region] = values[take],keys[take]
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
            plane = Plane(fit['center'],fit['normal'],fit['basis'],outline[ConvexHull(outline).vertices],fit['points'],True)
        except (ValueError,QhullError):
            detail['reason']='degenerate_plane_extent'
            continue
        planes[region] = plane
        detail.update(accepted=True,inlier_fraction=fit['inlier_fraction'],residual_p95=fit['residual_p95'],coverage=coverage.tolist())
        detail.update(center=plane.center.tolist(),normal=plane.normal.tolist())
    if _allow_expand and len(planes)<len(regions) and config['mirror_frame_band_max']>config['mirror_frame_band']:
        wider = {**config,'mirror_frame_band':config['mirror_frame_band_max']}
        _,wall_planes,wall_details = fit_surfaces(store,camera,pose,mask,wider,control,False)
        by_region = {item['region']:item for item in wall_details}
        for i,detail in enumerate(details):
            region = detail['region']
            if region not in planes and region in wall_planes:
                planes[region] = wall_planes[region]
                details[i] = {**by_region[region],'initial_frame_rejection':detail.get('reason')}
            elif region not in planes:
                detail['expanded_frame_rejection'] = by_region[region].get('reason')
    return labels,planes,details


def build(reader,stores,inspection,config,work,image_cache,mask_cache,model_hash,factory,control,preset,assets):
    analysis = MirrorAnalysis(work/'mirrors',config)
    kinds = ['mirror_mask']+(['glass_mask'] if config['glass_analysis'] else [])
    auxiliary = {c['id']:c for c in preset['auxiliary']}
    fallback = {'glass_mask':['window glass','glass window pane','windowpane']}
    prompts = [PromptClass(kind,tuple(auxiliary.get(kind,{}).get('prompts',fallback.get(kind,['mirror']))),config['instance_confidence']) for kind in kinds]
    output = Path(config['output'])/'diagnostics'/'reflections'
    output.mkdir(parents=True,exist_ok=True)
    image_count = sum(1 for store in stores for index in inspection['scans'][store.meta['scan_index']-1]['image_indices']
                      if inspection['images'][index-1]['pose_status']=='valid'
                      and any(r!='visualReferenceRepresentation' for r in inspection['images'][index-1]['representations']))
    image_number = 0
    for store in stores:
        scan = store.meta['scan_index']
        for index in inspection['scans'][scan-1]['image_indices']:
            item = inspection['images'][index-1]
            if item['pose_status']!='valid' or not any(r!='visualReferenceRepresentation' for r in item['representations']):
                continue
            image_number += 1
            context = dict(image_number=image_number,image_count=image_count,scan=scan,scan_count=len(stores))
            camera,pose,masks,preview = make_masks(reader,item,config,image_cache,mask_cache,model_hash,factory,control,prompts,assets,context)
            analysis.images[index] = []
            for kind,mask in zip(kinds,masks):
                labels,planes,details = fit_surfaces(store,camera,pose,mask,config,control)
                analysis.images[index].append(SurfaceMask(kind,camera,pose,labels,planes))
                for plane in planes.values():
                    # Registry is diagnostic only: it never marks points outside their own mask.
                    analysis.registry.add(plane,config['mirror_merge_angle'],config['mirror_merge_distance'])
                analysis.report['images'].append({'scan':scan,'image':index,'kind':kind,'regions':details})
                active = labels>0
                Image.fromarray(active.astype(np.uint8)*255).save(output/f'image_{index:04d}_{kind}.png')
                overlay = preview.copy()
                color = np.array([255,0,255] if kind=='mirror_mask' else [0,220,255])
                overlay[active] = np.rint(.5*overlay[active]+.5*color).astype(np.uint8)
                Image.fromarray(overlay).save(output/f'image_{index:04d}_{kind}_overlay.jpg',quality=92)
                if any(not d['accepted'] for d in details):
                    control.event('warning',code='reflection_plane_missing',scan=scan,image=index,mask_kind=kind)
                atomic_json(output/f'image_{index:04d}_{kind}_planes.json',details)
    atomic_json(analysis.directory/'registry.json',[p.serialize() for p in analysis.registry.planes])
    for number,store in enumerate(stores,1):
        scan = store.meta['scan_index']
        pose = Pose.from_dict(store.meta['metadata']['pose'])
        path = analysis.directory/f'states_{scan:04d}.npy'
        temp = path.with_suffix('.partial.npy')
        out = np.lib.format.open_memmap(temp,mode='w+',dtype=np.uint8,shape=(store.meta['record_count'],))
        counts = np.zeros(5,np.int64)
        try:
            for start,xyz,valid in store.chunks():
                control.checkpoint()
                states = analysis.states(pose.to_world(xyz[valid]),scan,pose.translation,inspection['scans'][scan-1]['image_indices'])
                block = np.zeros(len(xyz),np.uint8)
                block[valid] = states
                out[start:start+len(xyz)] = block
                counts += np.bincount(states,minlength=5)
            out.flush()
        finally:
            out._mmap.close()
        temp.replace(path)
        analysis.paths[scan] = path
        analysis.report['scans'][str(scan)] = {'counts':counts.tolist(),'sha256':file_hash(path)}
        control.event('progress',phase='mirror_evidence',scan=scan,scan_count=len(stores),fraction=.9+.1*number/len(stores))
    atomic_json(Path(config['output'])/'reflection_report.json',analysis.report)
    return analysis
