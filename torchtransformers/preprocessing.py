import torch

from torchtransformers.base import TransformerMixin, require_fitted


__all__ = ["StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer", "FunctionTransformer", "Clamp"]


class StandardScaler(TransformerMixin):
    def __init__(self):
        super().__init__()
        # Default values result in no scaling
        # self.register_buffer("loc", 0.0)
        # self.register_buffer("scale", 1.0)

    def __repr__(self):
        return f"StandardScaler(loc={self.loc}, scale={self.scale})"

    def fit(self, x: torch.Tensor, y: torch.Tensor | None = None) -> "StandardScaler":
        """
        Calculates the mean and standard deviation of the values in the last dimension of the input tensor.
        """
        dims = tuple(range(x.dim() - 1))
        self.register_buffer("loc", x.mean(dim=dims))
        self.register_buffer("scale", x.std(dim=dims))
        # self.loc = x.mean(dim=dims)
        # self.scale = x.std(dim=dims)
        self._is_fitted = True
        return self

    @require_fitted
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.loc) / self.scale

    @require_fitted
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.loc


class MinMaxScaler(TransformerMixin):
    def __init__(self):
        super().__init__()
        # Default values result in no scaling
        # self.register_buffer("loc", 0.0)
        # self.register_buffer("scale", 1.0)

    def __repr__(self):
        return f"MinMaxScaler(loc={self.loc}, scale={self.scale})"

    def fit(self, x: torch.Tensor, y: torch.Tensor | None = None) -> "MinMaxScaler":
        """
        Calculates the minimum and maximum of the values in the last dimension of the input tensor.
        """
        self.register_buffer("loc", x.min(dim=-2).values)
        self.register_buffer("scale", x.max(dim=-2).values - self.loc)
        # self.loc = x.min(dim=-2).values
        # self.scale = x.max(dim=-2).values - self.loc
        self._is_fitted = True
        return self

    @require_fitted
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.loc) / self.scale

    @require_fitted
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.loc


class RobustScaler(TransformerMixin):
    def __init__(self):
        super().__init__()
        # Default values result in no scaling
        # self.register_buffer("loc", 0.0)
        # self.register_buffer("scale", 1.0)

    def __repr__(self):
        return f"RobustScaler(loc={self.loc}, scale={self.scale})"

    def fit(self, x: torch.Tensor, y: torch.Tensor | None = None) -> "RobustScaler":
        """
        Calculates the median and the interquartile range of the values in the last dimension of the input tensor.
        """
        q1 = torch.quantile(x, 0.25, dim=-2)
        q2 = torch.quantile(x, 0.50, dim=-2)
        q3 = torch.quantile(x, 0.75, dim=-2)
        self.register_buffer("loc", q2)
        self.register_buffer("scale", q3 - q1)
        # self.loc = q2
        # self.scale = q3 - q1
        self._is_fitted = True
        return self

    @require_fitted
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.loc) / self.scale

    @require_fitted
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.loc


class Normalizer(TransformerMixin):
    def __init__(self, norm="l1"):
        super().__init__()

        p = None
        if isinstance(norm, str) and norm.lower() == "l1":
            p = 1
        elif isinstance(norm, str) and norm.lower() == "l2":
                p = 2
        elif isinstance(norm, int):
            p = norm
        else:
            raise ValueError(f"Invalid norm: {norm}")
        self.p = p
        self._is_fitted = True  # No fitting required

    def __repr__(self):
        return f"Normalizer(norm={self.p})"

    def fit(self, x: torch.Tensor, y: torch.Tensor | None = None) -> "Normalizer":
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.clone()
        xt = xt / xt.norm(p=self.p, dim=-1, keepdim=True).clamp_min(1e-12).expand_as(xt)
        return xt

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        # I don't think we'll actually need this
        raise NotImplementedError("Inverse transform not well-defined for normalization")


class FunctionTransformer(TransformerMixin):
    def __init__(self, func, inverse_func, validate=False):
        super().__init__()
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate
        self._is_fitted = True  # No fitting required

    def __repr__(self):
        return f"FunctionTransformer(func={self.func}, inverse_func={self.inverse_func})"

    def fit(self, x: torch.Tensor, y: torch.Tensor | None = None) -> "FunctionTransformer":
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.func(x)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.inverse_func(x)


class Clamp(TransformerMixin):
    def __init__(self, min_val: float | None = None, max_val: float | None = None):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self._is_fitted = True  # No fitting required

    def __repr__(self):
        return f"Clamp(min_val={self.min_val}, max_val={self.max_val})"

    def fit(self, x: torch.Tensor, y: torch.Tensor | None = None) -> "Clamp":
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=self.min_val, max=self.max_val)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=self.min_val, max=self.max_val)
