import torch

from torchtransformers.base import TransformerMixin, BaseEstimator


__all__ = ["ColumnTransformer", "TransformedTargetRegressor"]


class ColumnTransformer(TransformerMixin):
    def __init__(self, transformers: list[tuple[str, TransformerMixin, list[int]]], remainder="passthrough"):
        super().__init__()
        self.transformers = transformers
        self.remainder = remainder  # Not used

    def __repr__(self):
        s = "ColumnTransformer(transformers=[\n"
        for name, transformer, cols in self.transformers:
            s += f"\t{name}: {transformer}, cols={cols}\n"
        s += "])\n"
        return s

    def fit(self, x: torch.Tensor, y: torch.Tensor | None = None) -> "ColumnTransformer":
        # for name, transformer, cols in self.transformers:
        #     print("ColumnTransformer.fit", name, cols, x[..., cols].shape)
        #     x[..., cols] = transformer.fit_transform(x[..., cols])
        self._is_fitted = True
        return self

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        xt = x.clone()
        yt = y.clone() if y is not None else None
        for name, transformer, cols in self.transformers:
            if not transformer._is_fitted:
                xt[..., cols] = transformer.fit_transform(xt[..., cols], yt)
            else:
                xt[..., cols] = transformer(xt[..., cols])
        return xt

    def inverse_transform(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        xt = x.clone()
        yt = y.clone() if y is not None else None
        for name, transformer, cols in self.transformers:
            xt[..., cols] = transformer.inverse_transform(xt[..., cols], yt)
        return xt


class TransformedTargetRegressor(TransformerMixin, BaseEstimator):
    def __init__(self, regressor, transformer):
        super().__init__()
        self.regressor = regressor
        self.transformer = transformer

    def __repr__(self):
        return f"TransformedTargetRegressor(regressor={self.regressor}, transformer={self.transformer})"

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "TransformedTargetRegressor":
        yt = self.transformer.fit_transform(y)
        self.regressor.fit(x, yt)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer.inverse_transform(self.regressor(x))

    def parameters(self):
        return [*self.regressor.parameters(), *self.transformer.parameters()]
