import torch

from torchtransformers.base import BaseEstimator, TransformerMixin


__all__ = ["Pipeline"]


class Pipeline(BaseEstimator, TransformerMixin):
    def __init__(self, steps: list[tuple[str, TransformerMixin]]):
        """
        input_transformer and output_transformer should be fitted already
        """
        super().__init__()
        self.model = torch.nn.Sequential(*[step for _, step in steps])
        self.names = [name for name, _ in steps]

    def __repr__(self):
        s = "Pipeline(steps=[\n"
        for name, step in zip(self.names, self.model):
            s += f"\t{name}: {step}\n"
        s += "])\n"
        return s

    def fit(self, x: torch.Tensor, y: torch.Tensor | None = None) -> "Pipeline":
        xt = x.clone()
        yt = y.clone() if y is not None else None
        for step in self.model[:-1]:
            xt = step.fit_transform(xt, yt)
        self.model[-1].fit(xt, yt)
        return self

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.clone()
        for step in reversed(self.model):
            if not isinstance(step, TransformerMixin):
                break
            xt = step.inverse_transform(xt)
        return xt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def parameters(self) -> list:
        return self.model.parameters()

    def get(self, name):
        idx = self.names.index(name)
        return self.model[idx]
