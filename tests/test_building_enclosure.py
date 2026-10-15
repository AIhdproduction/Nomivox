import numpy as np

from e57cls.vote.engine import VoteStore


def key(x, y, z=0):
    return np.asarray([x, y, z], dtype='<i8').tobytes()


def add(store, code, x, y, z=0):
    store.db.execute('INSERT INTO decisions VALUES (?,?,?)', (key(x, y, z), code, 255))


def test_vertical_facade_ring_splits_ground_and_floor_without_nearest_fill(tmp_path):
    store = VoteStore(tmp_path/'votes.sqlite')
    try:
        # A measured, three-voxel-high facade ring encloses the centre.  The
        # ground label inside must become floor; an exterior floor becomes
        # ground.  No class is invented for cells without a decision.
        for x in range(5):
            for y in (0, 4):
                for z in range(3):
                    add(store, 6, x, y, z)
        for y in range(1, 4):
            for x in (0, 4):
                for z in range(3):
                    add(store, 6, x, y, z)
        add(store, 2, 2, 2)
        add(store, 102, 6, 2)
        store.db.commit()

        report = store.split_ground_floor_by_building(2, 102, [6], 1., 1., 2., 10_000)

        assert report['applied']
        result = dict(store.db.execute('SELECT key,code FROM decisions'))
        assert result[key(2, 2)] == 102
        assert result[key(6, 2)] == 2
    finally:
        store.close()


def test_unclosed_facade_is_not_interpolated_into_a_building(tmp_path):
    store = VoteStore(tmp_path/'votes.sqlite')
    try:
        # A single observed wall is not closed into an imaginary building.
        for y in range(5):
            for z in range(3):
                add(store, 6, 0, y, z)
        add(store, 2, 2, 2)
        store.db.commit()

        report = store.split_ground_floor_by_building(2, 102, [6], 1., 1., 2., 10_000)

        assert not report['applied']
        assert store.db.execute('SELECT code FROM decisions WHERE key=?', (key(2, 2),)).fetchone()[0] == 2
    finally:
        store.close()
