import pytest
import torch

from torchtransformers.compose import ColumnTransformer, TransformedTargetRegressor
from torchtransformers.pipeline import Pipeline
from torchtransformers.preprocessing import StandardScaler


@pytest.fixture
def setup_column_transformer():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    transformers = [
        ("scaler1", StandardScaler(), [0]),
        ("scaler2", StandardScaler(), [1])
    ]
    return x, ColumnTransformer(transformers)


class TestColumnTransformer:
    def test_inverse_transform(self, setup_column_transformer):
        x, column_transformer = setup_column_transformer
        column_transformer.fit(x)
        transformed = column_transformer.forward(x)
        inverse_transformed = column_transformer.inverse_transform(transformed)
        assert torch.allclose(inverse_transformed, x)

    def test_device(self, setup_column_transformer):
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            pytest.skip("No CUDA or MPS devices available")

        x, column_transformer = setup_column_transformer
        column_transformer.fit(x)

        # Try to evaluate the transformer on the device (was fitted on CPU). If this fails, the
        # the transformer is not being moved to the device correctly.
        try:
            column_transformer.to(device)(x.to(device))
        except RuntimeError as e:
            pytest.fail(f"ColumnTransformer not moved to device correctly. Error Message: {str(e)}")
