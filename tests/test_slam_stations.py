import numpy as np

from e57cls.geometry.cameras import Camera, Pose
from e57cls.pipeline.run import slam_observation_weight
from e57cls.slam.cameras import SlamCamera
from e57cls.slam.stations import cluster_stations
from e57cls.vote.engine import VoteStore, voxel_keys


def _camera(index, x, timestamp=None):
    model = Camera('sphericalRepresentation', {'imageWidth': 16, 'imageHeight': 8,
                   'pixelWidth': np.pi/8, 'pixelHeight': np.pi/8})
    return SlamCamera(index, Pose(np.eye(3), np.array((x, 0., 0.))), model, timestamp)


def test_virtual_stations_cluster_known_trajectory_with_and_without_timestamps():
    for timestamps in (False, True):
        cameras = [_camera(index + 1, index*.2, float(index) if timestamps else None) for index in range(60)]
        stations, membership, ordering = cluster_stations(cameras, 1.)
        assert len(stations) == 12
        assert sorted(membership.values()) == [station for station in range(1, 13) for _ in range(5)]
        assert ordering == ('timestamp' if timestamps else 'nearest_neighbour')


def test_station_votes_count_multiple_views_once_and_track_final_station():
    votes = VoteStore(':memory:', station_tracking=True)
    key = voxel_keys(np.array([[0., 0., 0.]]), np.zeros(3), .02)
    # The input vector already contains the maximum over a station's views.
    votes.add_scan(1, 'slam', iter([(key, np.array([[.9]], np.float32))]), [100], station_id=1)
    votes.add_scan(2, 'slam', iter([(key, np.array([[.8]], np.float32))]), [100], station_id=2)
    assert votes.db.execute('SELECT count FROM observations').fetchone()[0] == 2
    votes.decide(1, .4, [100])
    votes.finalize_station_ids()
    assert votes.lookup_station_ids(key).tolist() == [1]
    votes.close()


def test_slam_weight_prefers_near_central_perpendicular_observation():
    config = {'slam_weight_distance_scale': 5., 'slam_weight_incidence_power': 1., 'slam_weight_radial_power': 1.}
    weights = slam_observation_weight(np.array((1., 8.)), np.array((0., 1.2)), np.array((0., 1.)), config)
    assert weights[0] > weights[1]
