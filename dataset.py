import numpy as np
import torch
from torch.utils.data import Dataset
from statistics import mode

ANSWER_WORDS = 500
QUESTION_MAX_LENGTH = 26

class VQA(Dataset):
    def __init__(self, questions, answers_dict, images, gloves, answer_options):
        self.questions = questions
        self.images = images
        self.answers_dict = answers_dict
        self.gloves = gloves
        self.answer_options = answer_options

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()
        q = self.questions[idx]
        qid = q["question_id"]
        iid = q["image_id"]
        words = [w.translate({ord(i): None for i in "?!.:;,\"()#$/'`"}).lower() for w in q["question"].split(' ')]
        word_vectors = []
        for i in range(QUESTION_MAX_LENGTH):
            if i >= len(words) or words[i] not in self.gloves:
                word_vectors.append(np.zeros(300))
            else:
                word_vectors.append(self.gloves[words[i]])
        word_vectors = np.asarray(word_vectors)
        image = self.images[iid]
        answers = [a["answer"] for a in self.answers_dict[str(qid)]]
        answer = mode(answers)
        if answer in self.answer_options:
            answer_num = self.answer_options[answer]
        else:
            answer_num = ANSWER_WORDS

        return qid, torch.FloatTensor(image), torch.FloatTensor(word_vectors), answer_num

