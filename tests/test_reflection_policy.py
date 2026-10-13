from pathlib import Path
import numpy as np
from e57cls.geometry.cameras import Camera,Pose
from e57cls.mirror.evidence import Plane
from e57cls.mirror.pipeline import SurfaceMask,MirrorAnalysis,fit_surfaces,agreed_probability
from e57cls.pipeline.config import DEFAULTS


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
