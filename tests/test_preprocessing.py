import pytest
import os
import copy
import pickle
import torch
from torch.nn import Module

from torchtransformers.preprocessing import *

#===============
# Test fixtures
#===============

# Data fixtures

@pytest.fixture
def setup_2d_data():
    return torch.tensor([[1.0, 2.0],
                         [3.0, 4.0]])

@pytest.fixture
def setup_2d_data_long():
    return torch.tensor([[1.0,  2.0],
                         [3.0,  4.0],
                         [5.0,  6.0],
                         [7.0,  8.0],
                         [9.0, 10.0]])

@pytest.fixture
def setup_3d_data():
    return torch.tensor([[[1.0, 2.0],
                          [3.0, 4.0]]])

@pytest.fixture
def setup_3d_data_long():
    return torch.tensor([[[1.0,  2.0],
                          [3.0,  4.0],
                          [5.0,  6.0],
                          [7.0,  8.0],
                          [9.0, 10.0]]])

# Preprocessor fixtures

@pytest.fixture
def setup_standard_scaler():
    return StandardScaler()

@pytest.fixture
def setup_min_max_scaler():
    return MinMaxScaler()

@pytest.fixture
def setup_normalizer():
    return Normalizer("l1")

@pytest.fixture
def setup_clamp1():
    return Clamp(min_val=0.0)

@pytest.fixture
def setup_clamp2():
    return Clamp(max_val=1.0)

@pytest.fixture
def setup_clamp3():
    return Clamp(min_val=0.0, max_val=1.0)

@pytest.fixture
def setup_robust_scaler():
    return RobustScaler()


#================
# Test functions
#================

class TestStandardScaler:
    def test_standard_scaler_fit_2d(self, setup_standard_scaler, setup_2d_data):
        scaler = setup_standard_scaler
        x = setup_2d_data

        scaler.fit(x)

        assert torch.allclose(scaler.mean, torch.tensor([2.0, 3.0]))
        assert torch.allclose(scaler.std, torch.tensor([2.0, 2.0]) ** 0.5)

    def test_standard_scaler_fit_3d(self, setup_standard_scaler, setup_3d_data):
        scaler = setup_standard_scaler
        x = setup_3d_data

        scaler.fit(x)

        assert torch.allclose(scaler.mean, torch.tensor([2.0, 3.0]))
        assert torch.allclose(scaler.std, torch.tensor([2.0, 2.0]) ** 0.5)

    def test_standard_scaler_forward(self, setup_standard_scaler, setup_2d_data):
        scaler = setup_standard_scaler
        x = setup_2d_data

        scaler.fit(x)
        transformed = scaler.forward(x)
        xt_true = torch.tensor([[-1.0, -1.0], [1.0, 1.0]]) / (2.0 ** 0.5)

        assert torch.allclose(transformed, xt_true)

    def test_standard_scaler_inverse_transform(self, setup_standard_scaler, setup_2d_data):
        scaler = setup_standard_scaler
        x = setup_2d_data

        scaler.fit(x)
        transformed = scaler.forward(x)
        inverse_transformed = scaler.inverse_transform(transformed)

        assert torch.allclose(inverse_transformed, x)

    def test_save_model(self, setup_standard_scaler, setup_2d_data, tmp_path):
        scaler = setup_standard_scaler
        x = setup_2d_data

        scaler.fit(x)
        torch.save(scaler, tmp_path / "std.pt")
        scaler2 = torch.load(tmp_path / "std.pt")

        assert torch.allclose(scaler.transform(x), scaler2.transform(x))
        for k in scaler.state_dict().keys():
            assert torch.allclose(scaler.state_dict()[k], scaler2.state_dict()[k])


class TestMinMaxScaler:
    def test_min_max_scaler_fit_2d(self, setup_min_max_scaler, setup_2d_data):
        scaler = setup_min_max_scaler
        x = setup_2d_data

        scaler.fit(x)

        assert torch.allclose(scaler.min, torch.tensor([1.0, 2.0]))
        assert torch.allclose(scaler.max, torch.tensor([3.0, 4.0]))

    def test_min_max_scaler_fit_3d(self, setup_min_max_scaler, setup_3d_data):
        scaler = setup_min_max_scaler
        x = setup_3d_data

        scaler.fit(x)

        assert torch.allclose(scaler.min, torch.tensor([1.0, 2.0]))
        assert torch.allclose(scaler.max, torch.tensor([3.0, 4.0]))

    def test_min_max_scaler_forward(self, setup_min_max_scaler, setup_2d_data):
        scaler = setup_min_max_scaler
        x = setup_2d_data

        scaler.fit(x)
        transformed = scaler.forward(x)
        xt_true = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

        assert torch.allclose(transformed, xt_true)

    def test_min_max_scaler_inverse_transform(self, setup_min_max_scaler, setup_2d_data):
        scaler = setup_min_max_scaler
        x = setup_2d_data

        scaler.fit(x)
        transformed = scaler.forward(x)
        inverse_transformed = scaler.inverse_transform(transformed)

        assert torch.allclose(inverse_transformed, x)

    def test_save_model(self, setup_min_max_scaler, setup_2d_data, tmp_path):
        scaler = setup_min_max_scaler
        x = setup_2d_data

        scaler.fit(x)
        torch.save(scaler, tmp_path / "minmax.pt")
        scaler2 = torch.load(tmp_path / "minmax.pt")

        assert torch.allclose(scaler.transform(x), scaler2.transform(x))
        for k in scaler.state_dict().keys():
            assert torch.allclose(scaler.state_dict()[k], scaler2.state_dict()[k])


class TestRobustScaler:
    def test_robust_scaler_fit_2d(self, setup_robust_scaler, setup_2d_data_long):
        scaler = setup_robust_scaler
        x = setup_2d_data_long

        scaler.fit(x)

        assert torch.allclose(scaler.loc, torch.tensor([5.0, 6.0]))
        assert torch.allclose(scaler.scale, torch.tensor([4.0, 4.0]))

    def test_robust_scaler_fit_3d(self, setup_robust_scaler, setup_3d_data_long):
        scaler = setup_robust_scaler
        x = setup_3d_data_long

        scaler.fit(x)

        assert torch.allclose(scaler.loc, torch.tensor([5.0, 6.0]))
        assert torch.allclose(scaler.scale, torch.tensor([4.0, 4.0]))

    def test_robust_scaler_forward(self, setup_robust_scaler, setup_2d_data_long):
        scaler = setup_robust_scaler
        x = setup_2d_data_long

        scaler.fit(x)
        transformed = scaler.forward(x)
        xt_true = torch.tensor([[-1.0, -1.0],
                                [-0.5, -0.5],
                                [ 0.0,  0.0],
                                [ 0.5,  0.5],
                                [ 1.0,  1.0]])

        assert torch.allclose(transformed, xt_true)

    def test_robust_scaler_inverse_transform(self, setup_robust_scaler, setup_2d_data_long):
        scaler = setup_robust_scaler
        x = setup_2d_data_long

        scaler.fit(x)
        transformed = scaler.forward(x)
        inverse_transformed = scaler.inverse_transform(transformed)

        assert torch.allclose(inverse_transformed, x)

    def test_save_model(self, setup_robust_scaler, setup_2d_data, tmp_path):
        scaler = setup_robust_scaler
        x = setup_2d_data

        scaler.fit(x)
        torch.save(scaler, tmp_path / "robust.pt")
        scaler2 = torch.load(tmp_path / "robust.pt")

        assert torch.allclose(scaler.transform(x), scaler2.transform(x))
        for k in scaler.state_dict().keys():
            assert torch.allclose(scaler.state_dict()[k], scaler2.state_dict()[k])


class TestNormalizer:
    def test_normalizer_fit(self, setup_normalizer, setup_2d_data):
        normalizer = setup_normalizer
        x = setup_2d_data

        # Nothing to fit, but need to check the p parameter value being set correctly
        normalizer.fit(x)

        assert normalizer.p == 1.0

    def test_normalizer_fit_3d(self, setup_normalizer, setup_3d_data):
        normalizer = setup_normalizer
        x = setup_3d_data

        # Nothing to fit, but need to check the p parameter value being set correctly
        normalizer.fit(x)

        assert normalizer.p == 1.0

    def test_normalizer_forward(self, setup_normalizer, setup_2d_data):
        normalizer = setup_normalizer
        x = setup_2d_data

        normalizer.fit(x)
        transformed = normalizer.forward(x)
        xt_true = torch.tensor([[1.0 / 3.0, 2.0 / 3.0], [3.0 / 7.0, 4.0 / 7.0]])

        assert torch.allclose(transformed, xt_true)

    def test_save_model(self, setup_normalizer, setup_2d_data, tmp_path):
        normalizer = setup_normalizer
        x = setup_2d_data

        normalizer.fit(x)
        torch.save(normalizer, tmp_path / "robust.pt")
        scaler2 = torch.load(tmp_path / "robust.pt")

        assert torch.allclose(normalizer.transform(x), scaler2.transform(x))
        for k in normalizer.state_dict().keys():
            assert torch.allclose(normalizer.state_dict()[k], scaler2.state_dict()[k])


class TestClamp:
    def test_clamp1_forward(self, setup_clamp1):
        clamp = setup_clamp1
        x1 = torch.tensor([[-1.0, 2.0], [3.0, 4.0]])

        transformed = clamp.forward(x1)
        x1t_true = torch.tensor([[0.0, 2.0], [3.0, 4.0]])

        assert torch.allclose(transformed, x1t_true)

    def test_clamp2_forward(self, setup_clamp2):
        clamp = setup_clamp2
        x2 = torch.tensor([[0.5, 2.0], [3.0, 4.0]])

        transformed = clamp.fit(x2).transform(x2)
        x2t_true = torch.tensor([[0.5, 1.0], [1.0, 1.0]])

        assert torch.allclose(transformed, x2t_true)

    def test_clamp3_forward(self, setup_clamp3):
        clamp = setup_clamp3
        x3 = torch.tensor([[-1.0, 2.0], [0.5, 0.7]])

        transformed = clamp.fit_transform(x3)
        x3t_true = torch.tensor([[0.0, 1.0], [0.5, 0.7]])

        assert torch.allclose(transformed, x3t_true)

    def test_save_model(self, setup_clamp1, setup_2d_data, tmp_path):
        clamp = setup_clamp1
        x = setup_2d_data

        xt = clamp.fit_transform(x)
        torch.save(clamp, tmp_path / "clamp.pt")
        clamp2 = torch.load(tmp_path / "clamp.pt")
        xt2 = clamp2.transform(x)

        assert torch.allclose(xt, xt2)
