import numpy as np
import torch
from torch.utils.data import Dataset

ANSWER_WORDS = 500
QUESTION_WORDS = 15552
QUESTION_MAX_LENGTH = 26

class VQA(Dataset):
    def __init__(self, questions, answers_dict, images, words_index, answer_options):
        self.questions = questions
        self.images = images
        self.answers_dict = answers_dict
        self.words_index = words_index
        self.answer_options = answer_options

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()
        q = self.questions[idx]
        qid = q["question_id"]
        iid = q["image_id"]
        words = [w.translate({ord(i): None for i in "?!.:;,"}) for w in q["question"].split(' ')]
        idxs_tbl = np.zeros((QUESTION_MAX_LENGTH, QUESTION_WORDS))
        for i in range(QUESTION_MAX_LENGTH):
            if i < len(words) and words[i] in self.words_index:
                idxs_tbl[i][self.words_index[words[i]]] = 1
            else:
                idxs_tbl[i] += np.ones(QUESTION_WORDS) / QUESTION_WORDS
        image = self.images[iid]
        answers = self.answers_dict[str(qid)]
        ans_len = len(answers)
        answers_np = np.zeros(ANSWER_WORDS)
        for answer in answers:
            answer_key = answer["answer"]
            if answer_key in self.answer_options:
                answers_np[self.answer_options[answer_key]] += 1 / ans_len
            else:
                answers_np += np.ones(ANSWER_WORDS) / ANSWER_WORDS / ans_len
        return qid, torch.FloatTensor(image), torch.FloatTensor(idxs_tbl), torch.FloatTensor(answers_np)

