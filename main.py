from dataset import VQA, ANSWER_WORDS, QUESTION_WORDS
from network import VQANN, MultiClassCrossEntropyLoss
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

def answer_acc(prediction, answers):
    _, pred_max_idx = prediction.max(0)
    pred_max_idx = pred_max_idx.item()
    n_of_answers = answers[pred_max_idx].item() * 10
    return min(1, n_of_answers / 3)

def answer_acc_batch(predictions, answerss):
    acc = 0
    for prediction, answers in zip(predictions, answerss):
        acc += answer_acc(prediction, answers)
    return acc

def main(epochs, batch_size, output_file):
    with open("train.pickle", "rb") as f:
        images = pickle.load(f)
        print("Loaded train.pickle")

    with open("words_index.json", "r") as f:
        words_index = json.load(f)
        print("Loaded words_index.json")

    with open("answer_options500.json", "r") as f:
        answer_options = json.load(f)
        print("Loaded answer_options500.json")

    with open("answers_from_question.json", "r") as f:
        answers_dict = json.load(f)
        print("Loaded answers_from_question.json")

    with open("v2_OpenEnded_mscoco_train2014_questions.json", "r") as f:
        questions = json.load(f)["questions"]
        print("Loaded questions")

    vqann = VQANN(QUESTION_WORDS, IMAGE_FEATURES, ANSWER_WORDS)
    criterion = MultiClassCrossEntropyLoss()
    dataset = VQA(questions, answers_dict, images, words_index, answer_options)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vqann = vqann.to(device)
    optimizer = optim.SGD(vqann.parameters(), lr=1, momentum=0.9)

    since = time.time()

    acc_history = []
    loss_history = []

    best_model_wts = copy.deepcopy(vqann.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        vqann.train()
        running_loss = 0.0
        running_corrects = 0

        dataset_len = len(dataset)
        done = 0
    
        for _, image, words, answers in dataloader:
            this_batch_size = len(image)
            image = image.to(device)
            words = words.to(device)
            answers = answers.to(device)

            out = vqann(words, image)
            loss = criterion(out, answers)
            # reg_loss = torch.norm(model.fc.weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            batch_loss = loss.item()
            running_loss += batch_loss * this_batch_size
            running_corrects += answer_acc_batch(out.cpu(), answers.cpu())
            done += this_batch_size
            sys.stdout.write("\r{}/{}, Loss: {}".format(done, dataset_len, batch_loss))

        epoch_loss = running_loss / dataset_len
        epoch_acc = running_corrects / dataset_len

        print('\n Loss: {:.6f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(vqann.state_dict())

        acc_history.append(epoch_acc)
        loss_history.append(epoch_loss)

        print() 

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Acc: {:4f}'.format(best_acc))

    vqann.load_state_dict(best_model_wts)
    torch.save(vqann.state_dict(), output_file)

    plt.title("Training Accuracy")
    plt.xlabel("Training Epochs")
    plt.ylabel("Training Accuracy")

    plt.plot(range(1, epochs + 1), acc_history)
    plt.xticks(np.arange(1, epochs+1, 1.0))
    plt.savefig('vqann_loss.png')

    plt.title("Training Loss")
    plt.xlabel("Training Epochs")
    plt.ylabel("Training Loss")

    plt.plot(range(1, epochs + 1), loss_history)
    plt.xticks(np.arange(1, epochs+1, 1.0))
    plt.savefig('vqann_loss.png')

if __name__ == '__main__':
    epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    output_file = sys.argv[3]
    main(epochs, batch_size, output_file)