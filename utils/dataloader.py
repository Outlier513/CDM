from torch.utils.data import Dataset, DataLoader
from functools import partial
import json
import torch


class CognitiveDiagnosisDataset(Dataset):
    def __init__(self, record_file, is_val) -> None:
        super().__init__()
        self.is_val = is_val
        with open(record_file, "r") as rf:
            self.data = json.load(rf)
            self.len = len(self.data)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index):
        if self.is_val:
            item = self.data[index]
            user_id = item["user_id"]
            exer_id, knowledge_code, score = [], [], []
            for log in item["logs"]:
                exer_id.append(log["exer_id"])
                score.append(log["score"])
                knowledge_code.append(log["knowledge_code"])
        else:
            item = self.data[index]
            user_id = item["user_id"]
            exer_id = item["exer_id"]
            score = item["score"]
            knowledge_code = item["knowledge_code"]
        return user_id, exer_id, knowledge_code, score


def get_dataset(record_file, is_val=False):
    res = CognitiveDiagnosisDataset(record_file, is_val)
    return res


def collate_fn(batch, include_knowledge, knowledge_num):
    user_batch, exercise_batch, knowledge_batch, score_batch = [], [], [], []
    for user_id, exercise_id, knowledge_code, score in batch:
        user_batch.append(user_id)
        exercise_batch.append(exercise_id)
        score_batch.append(score)
        if include_knowledge:
            knowledge_emb = [0.0] * knowledge_num
            for c in knowledge_code:
                knowledge_emb[c - 1] = 1.0
            knowledge_batch.append(knowledge_emb)
    return (
        torch.tensor(user_batch, dtype=torch.int64)-1,
        torch.tensor(exercise_batch, dtype=torch.int64)-1,
        torch.tensor(knowledge_batch, dtype=torch.float32),
        torch.tensor(score_batch, dtype=torch.float32),
    )


def generate_dataloader(filename, args, include_knowledge=True, is_val=False):
    dataset = "data/" + args.dataset + "/" + filename
    dataset = get_dataset(dataset, is_val)
    if is_val:
        dataloader = []
        for user_id, exercise_id, knowledge_codes, score in dataset:
            user_batch = torch.tensor(
                [(user_id - 1)] * len(exercise_id), dtype=torch.int64)
            exercise_batch = torch.tensor(
                exercise_id, dtype=torch.int64) - 1
            score_batch = torch.tensor(score, dtype=torch.float32)
            knowledge_batch = []
            if include_knowledge:
                for knowledge_code in knowledge_codes:
                    knowledge_emb = [0.0] * args.knowledge_num
                    for c in knowledge_code:
                        knowledge_emb[c - 1] = 1.0
                    knowledge_batch.append(knowledge_emb)
            knowledge_batch = torch.tensor(
                knowledge_batch, dtype=torch.float32)
            dataloader.append((user_batch, exercise_batch,
                              knowledge_batch, score_batch))
    else:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=partial(
                collate_fn, include_knowledge=include_knowledge, knowledge_num=args.knowledge_num),
            num_workers=args.num_workers,
        )
    return dataloader
