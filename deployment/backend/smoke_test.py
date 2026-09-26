import numpy as np

try:
    from .encoder import EMBEDDING_DIM, embed_preprocessed, load_model
except ImportError:
    from encoder import EMBEDDING_DIM, embed_preprocessed, load_model


if __name__ == "__main__":
    print(load_model("cpu"))
    gesture, user = embed_preprocessed(
        np.zeros((32, 127), dtype=np.float32),
        2.0,
    )
    assert gesture.shape == (EMBEDDING_DIM,)
    assert user.shape == (EMBEDDING_DIM,)
    assert np.isclose(np.linalg.norm(gesture), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(user), 1.0, atol=1e-5)
    print("smoke test passed")
