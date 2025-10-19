import torch
import math

from torch.utils.data import Dataset, IterableDataset, DataLoader, get_worker_info
from src.data_preparation.special_tokens import SpecialTokens


class GPTDataset(Dataset):
    # Stride is the step size for the sliding window
    # max_length is the length of each chunk
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={SpecialTokens.EOT.value})

        # Use a sliding window to chunk the text into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):  # stride = step
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        # os.system("taskset -p 0x3f %d" % os.getpid())
        return self.input_ids[idx], self.target_ids[idx]


class GPTIterableDataset(IterableDataset):

    def __init__(self, txt, tokenizer, max_length, stride):
        super(GPTIterableDataset).__init__()

        self.input_ids = []
        self.target_ids = []
        self.items = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={SpecialTokens.EOT.value})

        # Use a sliding window to chunk the text into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):  # stride = step
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            self.items.append((torch.tensor(input_chunk), torch.tensor(target_chunk)))

        self.start = 0
        self.end = len(self.items)
        self.length = self.end
        assert self.end > self.start, "this example only works with end >= start"

    def __iter__(self):
        worker_info = get_worker_info()
        
        if worker_info is None:  # single-process data loading, return the full iterator
            start_idx = self.start
            end_idx = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(
                math.ceil((self.end - self.start) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            start_idx = self.start + worker_id * per_worker
            end_idx = min(start_idx + per_worker, self.end)

        return iter(self.items[start_idx:end_idx])

    def __len__(self):
        # Return the known length
        return self.length


def create_dataloader(tokenizer, txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Create dataset
    dataset = GPTIterableDataset(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader
