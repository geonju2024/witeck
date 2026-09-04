import numpy as np

from witeck_auth.data import WiteckArrays


def test_dataset_branch_flattened_schema(tmp_path):
    path = tmp_path / "dataset.npz"
    np.savez(
        path,
        X=np.zeros((3, 32 * 169), dtype=np.float32),
        performer=np.array(["P01", "P02", "P03"]),
        gesture=np.array(["G1", "G1", "G2"]),
        T=np.array(32),
        D=np.array(169),
    )

    arrays = WiteckArrays.from_npz(path)

    assert arrays.x.shape == (3, 32, 169)
    assert arrays.user_ids.tolist() == ["P01", "P02", "P03"]
    assert arrays.gesture_ids.tolist() == ["G1", "G1", "G2"]
