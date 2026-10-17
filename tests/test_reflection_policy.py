from pathlib import Path
import numpy as np
import pytest
from e57cls.geometry.cameras import Camera,Pose
from e57cls.mirror.evidence import Plane
from e57cls.mirror.pipeline import SurfaceMask,MirrorAnalysis,build,fit_surfaces,agreed_probability
from e57cls.pipeline.config import DEFAULTS
from e57cls.pipeline.control import Cancelled


def test_prompt_agreement_rejects_single_phrase_false_positive():
    scores=np.array([[[.9,.9]],[[.8,.1]],[[.1,.1]]])
    assert (agreed_probability(scores,2)>=.45).tolist()==[[True,False]]
    assert (agreed_probability(scores[:1],2)>=.45).tolist()==[[True,True]]


def setup_surface(kind='mirror_mask'):
    camera=Camera('pinholeRepresentation',dict(imageWidth=160,imageHeight=80,pixelWidth=1.,pixelHeight=1.,focalLength=100.,principalPointX=80.,principalPointY=40.))
    pose=Pose(np.eye(3),np.zeros(3))
    labels=np.zeros((80,160),np.int32)
    labels[20:60,60:100]=1
    labels[37:44,89:92]=0
    plane=Plane([0,0,-2],[0,0,1],[[1,0,0],[0,1,0]],[[-1,-1],[-1,1],[1,1],[1,-1]],np.empty((0,3)),True)
    return SurfaceMask(kind,camera,pose,labels,{1:plane})


def test_scan_local_mask_depth_including_glass_and_mask_holes(tmp_path):
    points=np.array([[0,0,-4],[0,0,-2],[0,0,-1],[1,0,-4],[.4,0,-4]])
    for kind in ('mirror_mask','glass_mask'):
        surface=setup_surface(kind)
        assert surface.states(np.zeros(3),points,.03).tolist()==[2,1,0,0,0]
        analysis=MirrorAnalysis(tmp_path/kind,DEFAULTS)
        analysis.images[8]=[surface]
        # Another scan does not inherit class 68 simply by sharing this voxel.
        assert analysis.states(points,8,np.zeros(3),[8]).tolist()==[2,1,0,0,0]
        assert analysis.states(points,9,np.zeros(3),[]).tolist()==[0,0,0,0,0]
        surface.planes={}
        assert surface.states(np.zeros(3),points,.03).tolist()==[4,4,4,0,0]


def test_frame_plane_does_not_fit_reflected_interior():
    surface=setup_surface()
    x,y=np.meshgrid(np.arange(160)+.5,np.arange(80)+.5)
    rays=surface.camera.rays(np.column_stack((x.ravel(),y.ravel())))
    depth=np.where(surface.labels.ravel()>0,6.,2.)
    world=rays*(-depth/rays[:,2])[:,None]
    class Store:
        meta={'record_count':len(world),'metadata':{'pose':{'rotation':dict(w=1.,x=0.,y=0.,z=0.),'translation':dict(x=0.,y=0.,z=0.)}}}
        def chunks(self):
            yield 0,world,np.ones(len(world),bool)
    class Control:
        def checkpoint(self):
            return None
    _,planes,details=fit_surfaces(Store(),surface.camera,surface.pose,surface.labels>0,DEFAULTS,Control())
    assert details[0]['accepted']
    assert abs(planes[1].center[2]+2)<1e-6


def test_adjacent_wall_recovers_plane_when_narrow_edge_is_noisy():
    from scipy.ndimage import distance_transform_edt,binary_fill_holes
    surface=setup_surface()
    mask=binary_fill_holes(surface.labels>0)
    x,y=np.meshgrid(np.arange(160)+.5,np.arange(80)+.5)
    rays=surface.camera.rays(np.column_stack((x.ravel(),y.ravel())))
    depth=np.where(mask.ravel(),6.,2.)
    edge=(distance_transform_edt(~mask)<=10)&~mask
    depth[edge.ravel()]=np.random.default_rng(5).uniform(3,9,edge.sum())
    world=rays*(-depth/rays[:,2])[:,None]
    class Store:
        meta={'metadata':{'pose':{'rotation':dict(w=1.,x=0.,y=0.,z=0.),'translation':dict(x=0.,y=0.,z=0.)}}}
        def chunks(self): yield 0,world,np.ones(len(world),bool)
    class Control:
        def checkpoint(self): pass
    _,planes,details=fit_surfaces(Store(),surface.camera,surface.pose,mask,DEFAULTS,Control())
    assert details[0]['accepted'] and details[0]['frame_band_pixels']==30
    assert details[0]['interior_samples']==0
    assert abs(planes[1].center[2]+2)<1e-6


def shape_store(camera, mask, interior_depth=6., depth_mask=None):
    x,y=np.meshgrid(np.arange(camera.width)+.5,np.arange(camera.height)+.5)
    rays=camera.rays(np.column_stack((x.ravel(),y.ravel())))
    depth=np.where((mask if depth_mask is None else depth_mask).ravel(),interior_depth,2.)
    world=rays*(-depth/rays[:,2])[:,None]
    class Store:
        meta={'metadata':{'pose':{'rotation':dict(w=1.,x=0.,y=0.,z=0.),'translation':dict(x=0.,y=0.,z=0.)}}}
        def chunks(self):
            yield 0,world,np.ones(len(world),bool)
    return Store()


class Control:
    def checkpoint(self):
        return None


def test_regularized_rectangle_fills_a_missing_mask_corner():
    surface=setup_surface()
    mask=surface.labels>0
    mask[20:28,60:68]=False
    config={**DEFAULTS,'mirror_edge_search':0.}
    labels,planes,details=fit_surfaces(shape_store(surface.camera,mask),surface.camera,surface.pose,mask,config,Control(),kind='mirror_mask')
    assert details[0]['shape']=='rectangle'
    assert details[0]['iou'] >= .75
    regular=SurfaceMask('mirror_mask',surface.camera,surface.pose,labels,planes,{1})
    # This point projects into the deliberately missing top-left mask corner.
    point=np.array([[-.65,-.65,-4.]])
    assert regular.region_ids(surface.camera.project(point)[0]).tolist()==[0]
    assert regular.states(np.zeros(3),point,.03).tolist()==[2]
    assert regular.states(np.zeros(3),np.array([[0.,0.,-1.]]),.03).tolist()==[0]


def test_oval_and_circle_choose_their_own_regular_shape():
    surface=setup_surface()
    yy,xx=np.mgrid[:80,:160]
    oval=((xx-80)/28)**2+((yy-40)/18)**2 <= 1
    circle=(xx-80)**2+(yy-40)**2 <= 18**2
    config={**DEFAULTS,'mirror_edge_search':0.}
    _,_,oval_detail=fit_surfaces(shape_store(surface.camera,oval),surface.camera,surface.pose,oval,config,Control(),kind='mirror_mask')
    _,_,circle_detail=fit_surfaces(shape_store(surface.camera,circle),surface.camera,surface.pose,circle,config,Control(),kind='mirror_mask')
    assert oval_detail[0]['shape']=='ellipse'
    assert circle_detail[0]['shape']=='circle'


def test_regularization_is_per_region_and_can_be_disabled():
    surface=setup_surface()
    yy,xx=np.mgrid[:80,:160]
    mask=np.zeros((80,160),bool)
    mask[20:45,30:55]=True
    mask |= (xx-115)**2+(yy-40)**2 <= 13**2
    config={**DEFAULTS,'mirror_min_pixels':20,'mirror_edge_search':0.}
    labels,planes,details=fit_surfaces(shape_store(surface.camera,mask),surface.camera,surface.pose,mask,config,Control(),kind='mirror_mask')
    assert {detail['shape'] for detail in details} == {'rectangle','circle'}
    regular=SurfaceMask('mirror_mask',surface.camera,surface.pose,labels,planes,{detail['region'] for detail in details})
    raw=SurfaceMask('mirror_mask',surface.camera,surface.pose,labels,planes)
    points=np.array([[.55,0.,-4.],[0.,0.,-1.]])
    assert raw.states(np.zeros(3),points,.03).tolist() == SurfaceMask('mirror_mask',surface.camera,surface.pose,labels,planes,set()).states(np.zeros(3),points,.03).tolist()
    assert regular.states(np.zeros(3),points,.03)[1] == 0
    disabled={**config,'mirror_shape_regularize':False}
    old_labels,old_planes,_=fit_surfaces(shape_store(surface.camera,mask),surface.camera,surface.pose,mask,disabled,Control(),kind='mirror_mask')
    assert SurfaceMask('mirror_mask',surface.camera,surface.pose,old_labels,old_planes).states(np.zeros(3),points,.03).tolist() == SurfaceMask('mirror_mask',surface.camera,surface.pose,old_labels,old_planes,set()).states(np.zeros(3),points,.03).tolist()


def test_depth_edges_restore_a_mask_six_centimetres_too_small():
    surface=setup_surface()
    true_mask=surface.labels>0
    mask=true_mask.copy()
    mask[20:23]=False
    mask[57:60]=False
    config={**DEFAULTS,'mirror_edge_search':.10}
    _,_,details=fit_surfaces(shape_store(surface.camera,mask,depth_mask=true_mask),surface.camera,surface.pose,mask,config,Control(),kind='mirror_mask')
    # Pinhole pixels at this distance are 2 cm apart, so three missing rows
    # correspond to a 6 cm edge correction.
    assert np.allclose(details[0]['edge_shift_m'][2:],.06,atol=.01)


def test_wall_spill_rejects_regularization_and_keeps_the_raw_gate():
    surface=setup_surface()
    mask=np.zeros((80,160),bool)
    mask[25:55,50:90]=True
    mask[35:45,90:150]=True
    config={**DEFAULTS,'mirror_edge_search':0.}
    labels,planes,details=fit_surfaces(shape_store(surface.camera,mask),surface.camera,surface.pose,mask,config,Control(),kind='mirror_mask')
    assert details[0]['shape'].startswith('rejected:')
    raw=SurfaceMask('mirror_mask',surface.camera,surface.pose,labels,planes)
    # The point lies in the form's bounding box, but in the empty L-shape corner.
    assert raw.states(np.zeros(3),np.array([[.8,-.35,-4.]]),.03).tolist()==[0]


def test_narrow_regularized_mask_keeps_the_raw_plane():
    surface=setup_surface()
    mask=np.zeros((80,160),bool)
    # This 17 cm narrow strip remains representable by the raw mask hull, but
    # its 2 cm side leaves too little area for a finite regular outline.
    mask[36:45,80:82]=True
    config={**DEFAULTS,'mirror_min_pixels':1,'mirror_edge_search':.10}
    _,planes,details=fit_surfaces(shape_store(surface.camera,mask),surface.camera,surface.pose,mask,config,Control(),kind='mirror_mask')
    assert details[0]['accepted']
    assert details[0]['shape']=='rejected:degenerate'
    assert not getattr(planes[1],'regularized',False)


def test_collapsed_rectangle_refinement_keeps_the_raw_plane(monkeypatch):
    import e57cls.mirror.pipeline as pipeline
    surface=setup_surface()
    mask=np.zeros((80,160),bool)
    mask[25:55,60:100]=True

    def collapse(shape,*args):
        return {**shape,'half':np.array([0.,shape['half'][1]])},[0.,0.,0.,0.]

    monkeypatch.setattr(pipeline,'_refine_rectangle',collapse)
    _,planes,details=fit_surfaces(shape_store(surface.camera,mask),surface.camera,surface.pose,mask,
                                  {**DEFAULTS,'mirror_edge_search':.10},Control(),kind='mirror_mask')
    assert details[0]['accepted']
    assert details[0]['shape']=='rejected:degenerate'
    assert not getattr(planes[1],'regularized',False)


def test_rectangle_refinement_cannot_cross_opposite_edges(monkeypatch):
    import e57cls.mirror.pipeline as pipeline
    local=[]
    signed=[]
    for axis,side in ((0,-1),(0,1),(1,-1),(1,1)):
        for along,value in ((-.115,-1.),(-.085,0.)):
            point=np.zeros(2)
            point[axis]=side*along
            local.extend([point]*25)
            signed.extend([value]*25)
    local=np.asarray(local)
    signed=np.asarray(signed)
    monkeypatch.setattr(pipeline,'_plane_hits',lambda *args:(np.ones(len(local),bool),local,signed))
    shape={'kind':'rectangle','center':np.zeros(2),'rotation':np.eye(2),'half':np.array([.1,.1])}
    refined,shifts=pipeline._refine_rectangle(shape,np.empty((len(local),3)),np.zeros(3),None,.03,.2,Control())
    assert np.allclose(refined['half'],[.05,.05])
    assert np.allclose(shifts,[-.05,-.05,-.05,-.05])


def test_regular_outline_hull_failure_keeps_the_raw_plane(monkeypatch):
    import e57cls.mirror.pipeline as pipeline
    from scipy.spatial import QhullError
    surface=setup_surface()
    mask=np.zeros((80,160),bool)
    mask[25:55,60:100]=True
    plane_factory=pipeline.Plane
    calls=0

    def fail_regular_outline(*args):
        nonlocal calls
        calls+=1
        if calls==2:
            raise QhullError('regular outline is flat')
        return plane_factory(*args)

    monkeypatch.setattr(pipeline,'Plane',fail_regular_outline)
    _,planes,details=fit_surfaces(shape_store(surface.camera,mask),surface.camera,surface.pose,mask,
                                  {**DEFAULTS,'mirror_edge_search':0.},Control(),kind='mirror_mask')
    assert details[0]['accepted']
    assert details[0]['shape']=='rejected:degenerate'
    assert not getattr(planes[1],'regularized',False)


def test_nan_round_axes_keep_the_raw_plane(monkeypatch):
    import e57cls.mirror.pipeline as pipeline
    surface=setup_surface()
    yy,xx=np.mgrid[:80,:160]
    mask=(xx-80)**2+(yy-40)**2 <= 18**2

    def nan_axes(shape,*args):
        return {**shape,'axes':np.array([np.nan,shape['axes'][1]])},0.

    monkeypatch.setattr(pipeline,'_refine_round',nan_axes)
    _,planes,details=fit_surfaces(shape_store(surface.camera,mask),surface.camera,surface.pose,mask,
                                  {**DEFAULTS,'mirror_edge_search':.10},Control(),kind='mirror_mask')
    assert details[0]['accepted']
    assert details[0]['shape']=='rejected:degenerate'
    assert not getattr(planes[1],'regularized',False)


def build_with_fit_failure(tmp_path, monkeypatch, error, strict=False, fail_masks=False):
    import e57cls.mirror.pipeline as pipeline
    surface=setup_surface()

    class Store:
        meta={'scan_index':1,'record_count':1,'metadata':{'pose':{'rotation':dict(w=1.,x=0.,y=0.,z=0.),'translation':dict(x=0.,y=0.,z=0.)}}}

        def chunks(self):
            yield 0,np.array([[0.,0.,-1.]]),np.ones(1,bool)

    class EventControl(Control):
        def __init__(self):
            self.events=[]

        def event(self, kind, **payload):
            self.events.append((kind,payload))

    def fake_masks(reader,item,*args):
        if fail_masks and item['index']==1:
            raise error
        failed=np.zeros((80,160),bool)
        failed[0,0]=item['index']==1
        return surface.camera,surface.pose,[failed,np.zeros_like(failed),np.zeros_like(failed)],np.zeros((80,160,3),np.uint8)

    def fake_fit(store,camera,pose,mask,config,control,kind=None):
        if kind=='mirror_mask' and mask[0,0]:
            raise error
        labels=np.zeros(mask.shape,np.int32)
        labels[20:25,20:25]=1
        return labels,{},[{'region':1,'accepted':True}]

    monkeypatch.setattr(pipeline,'make_masks',fake_masks)
    monkeypatch.setattr(pipeline,'fit_surfaces',fake_fit)
    inspection={'scans':[{'image_indices':[1,2]}],
                'images':[{'index':1,'pose_status':'valid','representations':['pinholeRepresentation']},
                          {'index':2,'pose_status':'valid','representations':['pinholeRepresentation']}]}
    control=EventControl()
    analysis=build(None,[Store()],inspection,{**DEFAULTS,'output':str(tmp_path/'output'),'strict':strict},
                   tmp_path/'work',None,None,None,None,control,{'auxiliary':[]},None)
    return analysis,control


def test_mirror_fit_failure_isolated_to_one_image_and_kind(tmp_path,monkeypatch):
    analysis,control=build_with_fit_failure(tmp_path,monkeypatch,RuntimeError('plane fit failed'))
    failed=[item for item in analysis.report['images'] if item.get('failed')]
    assert failed == [{'scan':1,'image':1,'kind':'mirror_mask','regions':[],
                       'failed':True,'error':'plane fit failed'}]
    assert all(surface.kind!='mirror_mask' for surface in analysis.images[1])
    assert len(analysis.images[2]) == 3
    assert ('warning',{'code':'mirror_image_failed','scan':1,'image':1,
                       'mask_kind':'mirror_mask','detail':'plane fit failed'}) in control.events


def test_mirror_mask_failure_marks_all_image_kinds_failed(tmp_path,monkeypatch):
    analysis,control=build_with_fit_failure(tmp_path,monkeypatch,RuntimeError('mask creation failed'),fail_masks=True)
    failed=[item for item in analysis.report['images'] if item.get('failed')]
    assert [item['kind'] for item in failed] == ['mirror_mask','reflective_mask','glass_mask']
    assert all(item['error']=='mask creation failed' for item in failed)
    assert analysis.images[1] == []
    assert len(analysis.images[2]) == 3
    assert len([event for event in control.events if event[1].get('code')=='mirror_image_failed']) == 3


def test_mirror_fit_failure_is_strict_when_requested(tmp_path,monkeypatch):
    with pytest.raises(RuntimeError,match='plane fit failed'):
        build_with_fit_failure(tmp_path,monkeypatch,RuntimeError('plane fit failed'),strict=True)


def test_mirror_fit_cancellation_is_always_propagated(tmp_path,monkeypatch):
    with pytest.raises(Cancelled):
        build_with_fit_failure(tmp_path,monkeypatch,Cancelled())
