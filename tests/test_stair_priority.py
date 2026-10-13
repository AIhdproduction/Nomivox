import numpy as np
from e57cls.vote.engine import VoteStore,voxel_keys
from e57cls.pipeline.run import evidence_map
from e57cls.pipeline.config import DEFAULTS


def test_stairs_survive_other_scan_floor_votes_only_with_strong_evidence(tmp_path):
    votes=VoteStore(tmp_path/'v.sqlite')
    keys=voxel_keys([[0,0,0],[1,0,0],[2,0,0]],np.zeros(3),.02)
    votes.add_scan(1,'v2',[(keys,np.array([[.8,0,0],[.55,0,0],[.9,0,0]]))],[105,102,100])
    for scan in (2,3,4):
        votes.add_scan(scan,'v2',[(keys,np.array([[0,.95,0],[0,.95,0],[0,0,.95]]))],[105,102,100])
    votes.decide(1,.4,[105,102,100],priority={'code':105,'over':[102,101,2],'confidence':.65})
    assert votes.lookup(keys)[0].tolist()==[105,102,100]
    assert votes.priority_overrides==1
    votes.close()


def test_pixel_stair_priority_keeps_confidence_and_boundary_rules():
    classes=[{'id':'floor'},{'id':'stairs'}]
    floor=np.full((8,8),.95)
    stairs=np.full((8,8),.8)
    labels,confidence=evidence_map([floor,stairs],classes,DEFAULTS)
    assert labels[4,4]==1
    assert confidence[4,4]==np.float32(.8)
    assert labels[0,0]==-1
