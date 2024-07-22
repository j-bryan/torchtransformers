import torch
from tqdm import tqdm
from torchtransformers.base import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: torch.nn.Module,
                 batch_size: int = 32,
                 epochs: int = 100,
                 device: str = "cpu",
                 **kwargs):
        super().__init__()
        self.model = model.to(device)
        opt_kwargs = self._extract_kwargs(kwargs, "optimizer")
        self.optimizer = optimizer(self.model.parameters(), **opt_kwargs)
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.losses = None

    def __repr__(self):
        return f"Regressor(model={self.model}, optimizer={self.optimizer}, loss_fn={self.loss_fn}, batch_size={self.batch_size}, epochs={self.epochs}, device={self.device})"

    def _extract_kwargs(self, kwargs, prefix):
        return {k.split("__")[1]: v for k, v in kwargs.items() if k.startswith(prefix + "__")}

    def fit(self, x: torch.Tensor, y: torch.Tensor, **kwargs):
        dataloader_kwargs = self._extract_kwargs(kwargs, "dataloader")
        self.batch_size = dataloader_kwargs.pop("batch_size", self.batch_size)
        dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x.to(self.device), y.to(self.device)),
            batch_size=self.batch_size,
            **dataloader_kwargs
        )
        len_dataloader = len(dataloader)

        bar = tqdm(range(self.epochs))
        losses = torch.zeros(self.epochs)

        for i in bar:
            total_loss = 0
            for xi, yi in dataloader:
                self.optimizer.zero_grad()
                y_pred = self.model(xi)
                loss = self.loss_fn(y_pred, yi)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            losses[i] = total_loss / len_dataloader
            bar.set_postfix({"Loss": f"{total_loss / len_dataloader:.4f}"})

        self.losses = losses
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
