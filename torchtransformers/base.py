import torch


__all__ = ["TransformerMixin"]


def require_fitted(func):
    def wrapper(self, *args, **kwargs):
        if not self._is_fitted:
            raise ValueError(f"Transformer {self.__class__.__name__} not fitted yet")
        return func(self, *args, **kwargs)

    return wrapper


class TransformerMixin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._is_fitted = False

    def fit(self, x: torch.Tensor, y: torch.Tensor | None = None) -> "TransformerMixin":
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def fit_transform(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        return self.fit(x).transform(x)

    def parameters(self) -> list:
        return []


class BaseEstimator(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "BaseEstimator":
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def fit_predict(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.fit(x, y).predict(x)
