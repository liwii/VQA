from dataset import VQA, ANSWER_WORDS
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

def answer_acc(predictions, answers):
    _, pred_max_idx = predictions.max(1)
    return (answers == pred_max_idx).sum().item()

def main(epochs, batch_size, output_file):
    with open("train.pickle", "rb") as f:
        images = pickle.load(f)
        print("Loaded train.pickle")

    with open("glove_in_question.pickle", "rb") as f:
        gloves = pickle.load(f)
        print("Loaded gloves")

    with open("answer_options1000.json", "r") as f:
        answer_options = json.load(f)
        print("Loaded answer_options")

    with open("answers_from_question.json", "r") as f:
        answers_dict = json.load(f)
        print("Loaded answers_from_question.json")

    with open("v2_OpenEnded_mscoco_train2014_questions.json", "r") as f:
        questions = json.load(f)["questions"]
        print("Loaded questions")

    vqann = VQANN(IMAGE_FEATURES, ANSWER_WORDS + 1)
    criterion = nn.CrossEntropyLoss()
    dataset = VQA(questions, answers_dict, images, gloves, answer_options)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    vqann = vqann.to(device)
    optimizer = optim.Adam(vqann.parameters())

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
    
        for _, image, words, answers, word_lengths in dataloader:
            this_batch_size = len(image)
            image = image.to(device)
            words = words.to(device)
            answers = answers.to(device)
            word_lengths = word_lengths.to(device)

            out = vqann(words, image, word_lengths)
            loss = criterion(out, answers)
            # reg_loss = torch.norm(model.fc.weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            batch_loss = loss.item()
            running_loss += batch_loss * this_batch_size
            running_corrects += answer_acc(out, answers)
            done += this_batch_size
            sys.stdout.write("\r{}/{}, Loss: {:3f}, Acc: {:3f}".format(done, dataset_len, batch_loss, running_corrects / done))

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