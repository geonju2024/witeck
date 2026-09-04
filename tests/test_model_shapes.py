import torch

from witeck_auth.models import InceptionTimeSiamese, TCNSiamese, Prototypical1DCNN


def test_siamese_shapes():
    left = torch.randn(4, 32, 169)
    right = torch.randn(4, 32, 169)
    for model in (InceptionTimeSiamese(), TCNSiamese()):
        output = model(left, right)
        assert output["left_embedding"].shape == (4, 128)
        assert output["similarity"].shape == (4,)
        assert torch.allclose(output["left_embedding"].norm(dim=1), torch.ones(4), atol=1e-5)


def test_prototypical_shapes():
    model = Prototypical1DCNN()
    support_x = torch.randn(6, 32, 169)
    support_y = torch.tensor([0, 0, 0, 1, 1, 1])
    query_x = torch.randn(4, 32, 169)
    output = model(support_x, support_y, query_x)
    assert output["logits"].shape == (4, 2)
    assert output["prototypes"].shape == (2, 128)
