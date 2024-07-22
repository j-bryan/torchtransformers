import torch

from torchtransformers.base import TransformerMixin, BaseEstimator, require_fitted


__all__ = ["ColumnTransformer", "TransformedTargetRegressor"]


class ColumnTransformer(TransformerMixin):
    def __init__(self, transformers: list[tuple[str, TransformerMixin, list[int]]], remainder="passthrough"):
        super().__init__()
        self.transformers = torch.nn.ModuleDict({name: transformer for name, transformer, _ in transformers})
        self.cols = {name: cols for name, _, cols in transformers}
        self.remainder = remainder  # Not used

    def __repr__(self):
        s = "ColumnTransformer(transformers=[\n"
        for name, transformer in self.transformers.items():
            s += f"\t{name}: {transformer}, cols={self.cols[name]}\n"
        s += "])\n"
        return s

    def fit(self, x: torch.Tensor, y: torch.Tensor | None = None) -> "ColumnTransformer":
        xt = x.clone()
        yt = y.clone() if y is not None else None
        for name, transformer in self.transformers.items():
            cols = self.cols[name]
            xt[..., cols] = transformer.fit_transform(xt[..., cols], yt)
        self._is_fitted = True
        return self

    @require_fitted
    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        xt = x.clone()
        for name, transformer in self.transformers.items():
            cols = self.cols[name]
            xt[..., cols] = transformer(xt[..., cols])
        return xt

    @require_fitted
    def inverse_transform(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        xt = x.clone()
        for name, transformer in self.transformers.items():
            cols = self.cols[name]
            xt[..., cols] = transformer.inverse_transform(xt[..., cols])
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
        self._is_fitted = True
        return self

    @require_fitted
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer.inverse_transform(self.regressor(x))

    def parameters(self):
        return [*self.regressor.parameters(), *self.transformer.parameters()]
