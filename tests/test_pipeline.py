import pytest
import torch
from torch.nn import Module

from torchtransformers.pipeline import Pipeline
from torchtransformers.base import TransformerMixin


class MockTransformer(TransformerMixin, Module):
    def __init__(self):
        super().__init__()

    def fit(self, x, y=None):
        return self

    def forward(self, x):
        return x * 2

    def inverse_transform(self, x):
        return x / 2


@pytest.fixture
def setup_pipeline():
    transformers = [
        ("mock_transformer1", MockTransformer()),
        ("mock_transformer2", MockTransformer())
    ]
    return Pipeline(transformers)


def test_fit_transform(setup_pipeline):
    pipeline = setup_pipeline
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    # Separate fit and transform
    pipeline.fit(x)
    transformed = pipeline.forward(x)
    assert transformed.shape == x.shape
    assert torch.allclose(transformed, x * 4)

    # Combined fit_transform
    transformed = pipeline.fit_transform(x)
    assert transformed.shape == x.shape
    assert torch.allclose(transformed, x * 4)


def test_inverse_transform(setup_pipeline):
    # This particular pipeline should be able to reverse the transformation. Not possible with all
    # pipelines, particularly if there's a regression model at the end.
    pipeline = setup_pipeline
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    transformed = pipeline.forward(x)

    inverse_transformed = pipeline.inverse_transform(transformed)

    assert torch.allclose(inverse_transformed, x)


def test_forward(setup_pipeline):
    # forward and transform give the same result
    pipeline = setup_pipeline
    x = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0]])

    pipeline.fit(x)
    output_forward = pipeline.forward(x)
    output_transform = pipeline.fit_transform(x)

    assert output_forward.shape == torch.Size([2, 2])
    assert torch.allclose(output_forward, x * 4)
    assert torch.allclose(output_transform, x * 4)


if __name__ == "__main__":
    pytest.main()
