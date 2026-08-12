from typing import Iterator, Sized
from itertools import islice

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[int]:
        epoch_seed = self.seed + self.epoch + self.epoch_offset
        custom_iterator = getattr(self.dataset, "iter_epoch_indices", None)
        if callable(custom_iterator):
            indices = custom_iterator(
                epoch_seed,
                batch_size=self.batch_size,
                num_processes=self.num_processes,
            )
        else:
            g = torch.Generator(device="cpu")
            g.manual_seed(epoch_seed)
            indices = iter(torch.randperm(len(self.dataset), generator=g).tolist())
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = islice(indices, sample_offset, None)
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)
