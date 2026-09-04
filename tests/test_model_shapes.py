import torch

from witeck_auth.models import (
    InceptionTimeSiamese,
    MultiStreamDilatedSiamese,
    Prototypical1DCNN,
    TCNSiamese,
)


def test_siamese_shapes():
    left = torch.randn(4, 32, 169)
    right = torch.randn(4, 32, 169)
    for model in (
        InceptionTimeSiamese(),
        TCNSiamese(),
        MultiStreamDilatedSiamese(),
    ):
        output = model(left, right)
        assert output["left_embedding"].shape == (4, 128)
        assert output["similarity"].shape == (4,)
        assert torch.allclose(output["left_embedding"].norm(dim=1), torch.ones(4), atol=1e-5)



def test_multistream_rejects_non_witeck_schema():
    try:
        MultiStreamDilatedSiamese(input_dim=168)
    except ValueError as error:
        assert "169-feature schema" in str(error)
    else:
        raise AssertionError("expected invalid input_dim to be rejected")


def test_prototypical_shapes():
    model = Prototypical1DCNN()
    support_x = torch.randn(6, 32, 169)
    support_y = torch.tensor([0, 0, 0, 1, 1, 1])
    query_x = torch.randn(4, 32, 169)
    output = model(support_x, support_y, query_x)
    assert output["logits"].shape == (4, 2)
    assert output["prototypes"].shape == (2, 128)
