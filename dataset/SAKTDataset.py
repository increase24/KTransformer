from torch.utils.data import Dataset, DataLoader
import numpy as np

ACCEPTED_USER_CONTENT_SIZE = 4

class SAKTDataset(Dataset):
    def __init__(self, group, n_skill, max_seq=100):
        super(SAKTDataset, self).__init__()
        self.samples, self.n_skill, self.max_seq = {}, n_skill, max_seq

        self.user_ids = []
        for i, user_id in enumerate(group.index):
            if (i % 10000 == 0):
                print(f'Processed {i} users')
            content_id, answered_correctly = group[user_id] # np.array, np.array
            if len(content_id) >= ACCEPTED_USER_CONTENT_SIZE: # threshold
                if len(content_id) > self.max_seq:
                    total_questions = len(content_id)
                    last_pos = total_questions // self.max_seq
                    for seq in range(last_pos):
                        index = f"{user_id}_{seq}"
                        self.user_ids.append(index)
                        start = seq * self.max_seq
                        end = (seq + 1) * self.max_seq
                        self.samples[index] = (content_id[start:end], answered_correctly[start:end])
                    if len(content_id[end:]) >= ACCEPTED_USER_CONTENT_SIZE:
                        index = f"{user_id}_{last_pos + 1}"
                        self.user_ids.append(index)
                        self.samples[index] = (content_id[end:], answered_correctly[end:])
                else:
                    index = f'{user_id}'
                    self.user_ids.append(index)
                    self.samples[index] = (content_id, answered_correctly)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_seq_id = self.user_ids[index]
        content_id, answered_correctly = self.samples[user_seq_id]
        seq_len = len(content_id)

        content_id_seq = np.zeros(self.max_seq, dtype=int)
        answered_correctly_seq = np.zeros(self.max_seq, dtype=int)
        if seq_len >= self.max_seq:
            content_id_seq[:] = content_id[-self.max_seq:]
            answered_correctly_seq[:] = answered_correctly[-self.max_seq:]
        else:
            content_id_seq[-seq_len:] = content_id
            answered_correctly_seq[-seq_len:] = answered_correctly

        exercise_id = content_id_seq[1:]
        label = answered_correctly_seq[1:]

        recall = content_id_seq[:-1].copy()
        recall += (answered_correctly_seq[:-1] == 1) * self.n_skill

        return recall, exercise_id, label