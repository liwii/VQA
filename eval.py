from dataset import VQA
from network import VQANN
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import json
import time
import copy
import sys

IMAGE_FEATURES = 2048

def main(input_file, batch_size, output_file):
    image_pickle_f = "val.pickle"
    word_embeddings_f = "glove_in_question.pickle"
    answer_options_f = "answer_options500.json"
    answers_dict_f = "answers_from_question_val.json"
    questions_f = "v2_OpenEnded_mscoco_val2014_questions.json"
    state_dict_f = "vqann.pth"

    with open(image_pickle_f, "rb") as f:
        images = pickle.load(f)
        print("Loaded ", image_pickle_f)

    with open(word_embeddings_f, "rb") as f:
        gloves = pickle.load(f)
        print("Loaded ", word_embeddings_f)

    with open(answer_options_f, "r") as f:
        answer_options = json.load(f)
        print("Loaded ", answer_options_f)

    with open(answers_dict_f, "r") as f:
        answers_dict = json.load(f)
        print("Loaded ", answers_dict_f)

    with open(questions_f, "r") as f:
        questions = json.load(f)["questions"]
        print("Loaded ", questions_f)

    
    answer_words = len(answer_options)
    vqann = VQANN(IMAGE_FEATURES, answer_words + 1)
    answer_words_list = [None] * (answer_words + 1)
    for k in answer_options:
        v = answer_options[k]
        answer_words_list[v] = k
    answer_words_list[answer_words] = "yes"
    dataset = VQA(questions, answers_dict, images, gloves, answer_options)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    vqann = vqann.to(device)
    vqann.load_state_dict(torch.load(input_file))

    vqann.eval()

    dataset_len = len(dataset)
    done = 0
    answers = []

    for qids, image, words, _, word_lengths in dataloader:
        with torch.no_grad():
            this_batch_size = len(image)
            image = image.to(device)
            words = words.to(device)
            word_lengths = word_lengths.to(device)

            out = vqann(words, image, word_lengths)

            _, indices = out.max(1)
            for qid, index in zip(qids, indices):
                answers.append(
                    { "question_id": int(qid), "answer": answer_words_list[index.item()] }
                )
        # reg_loss = torch.norm(model.fc.weight)


        done += this_batch_size
        sys.stdout.write("\r{}/{}".format(done, dataset_len))

    with open(output_file, "w") as f:
        json.dump(answers, f)

if __name__ == '__main__':
    input_file = sys.argv[1]
    batch_size = int(sys.argv[2])
    output_file = sys.argv[3]
    main(input_file, batch_size, output_file)