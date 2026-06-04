from typing import Tuple, List, Dict
from torch import Tensor
import torch


class Tokenizer:
    def __init__(self, vocab: List[str]) -> None:
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(vocab)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(vocab)}

    def encode(self, text: str) -> Tensor:
        indices = [self.stoi[c] for c in text]
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices: Tensor) -> str:
        indices_list = indices.detach().cpu().tolist()
        return "".join([self.itos[i] for i in indices_list])


class Data:
    def __init__(
        self,
        file_path: str,
        train_ratio: float,
        block_size: int,
        batch_size: int,
        random_seed: int,
    ) -> None:
        torch.manual_seed(random_seed)
        self.data = self.read_file(file_path)
        self.vocab = sorted(list(set(self.data)))
        self.vocab_size = len(self.vocab)
        self.tokenizer = Tokenizer(self.vocab)
        self.train_ratio = train_ratio
        self.block_size = block_size
        self.batch_size = batch_size
        self.train_data, self.valid_data = self.split_data()
        self.train_data, self.valid_data = (
            self.tokenizer.encode(self.train_data),
            self.tokenizer.encode(self.valid_data),
        )

    @staticmethod
    def read_file(file_path: str) -> str:
        with open(file_path, "r") as f:
            return f.read()

    def split_data(self) -> Tuple[str, str]:
        split_index = int(len(self.data) * self.train_ratio)
        train_data = self.data[:split_index]
        validation_data = self.data[split_index:]
        return train_data, validation_data

    def get_batch(self, sub_data: Tensor) -> Tuple[Tensor, Tensor]:
        ix = torch.randint(len(sub_data) - self.block_size, (self.batch_size,))
        x = torch.stack([sub_data[i : i + self.block_size] for i in ix])
        y = torch.stack([sub_data[i + 1 : i + self.block_size + 1] for i in ix])
        return x, y
