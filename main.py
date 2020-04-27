from dataset import VQA, ANSWER_WORDS, QUESTION_WORDS
from network import VQANN
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

def main():
    num_epochs = 3
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
    criterion = nn.MSELoss()
    dataset = VQA(questions, answers_dict, images, words_index, answer_options)
    optimizer = optim.SGD(vqann.parameters(), lr=0.001, momentum=0.9)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vqann = vqann.to(device)

    since = time.time()

    acc_history = []
    loss_history = []

    best_model_wts = copy.deepcopy(vqann.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        vqann.train()
        running_loss = 0.0
        running_corrects = 0

        dataset_len = len(dataset)
        done = 0
    
        for _, image, words, answers in dataset:
            sys.stdout.write("\r{}/{}".format(done, dataset_len))
            image = image.to(device)
            words = words.to(device)
            answers = answers.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                out = vqann(words, image)
                loss = criterion(out, answers)
                # reg_loss = torch.norm(model.fc.weight)
                loss.backward()
                optimizer.step()


            running_loss += loss.item()
            running_corrects += answer_acc(out.cpu(), answers.cpu())
            done += 1

        epoch_loss = running_loss / dataset_len
        epoch_acc = running_corrects.double() / dataset_len

        print('\n Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        
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
    torch.save(vqann.state_dict(), 'vqann.pth')

    plt.title("Training Accuracy")
    plt.xlabel("Training Epochs")
    plt.ylabel("Training Accuracy")

    plt.plot(range(1, num_epochs + 1), acc_history)
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.savefig('vqann_loss.png')

    plt.title("Training Loss")
    plt.xlabel("Training Epochs")
    plt.ylabel("Training Loss")

    plt.plot(range(1, num_epochs + 1), loss_history)
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.savefig('vqann_loss.png')

#qid, image, words, answers = dataset[1]
#out = vqann(words, image)
#breakpoint()

if __name__ == '__main__':
    main()