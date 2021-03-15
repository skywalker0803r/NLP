import torch
from utils import create_mini_batch,FakeNewsDataset,get_predictions
from transformers import BertForSequenceClassification,BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import joblib

device = "cuda:0"

# 1. model and tokenizer 
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=3).to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# 2. dataset and dataloader
trainset = FakeNewsDataset("train", tokenizer=tokenizer)
testset = FakeNewsDataset("test", tokenizer=tokenizer)
trainloader = DataLoader(trainset, batch_size=256,collate_fn=create_mini_batch)
testloader = DataLoader(testset, batch_size=256, collate_fn=create_mini_batch)

# def how to train the model
def train(model,trainloader,device=device,EPOCHS=100,lr=1e-5):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc = -np.inf
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for data in tqdm(trainloader):
            tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]
            optimizer.zero_grad()
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors, 
                            labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        _,_, acc = get_predictions(model, trainloader, compute_acc=True)
        print('[epoch %d] loss: %.3f, acc: %.3f' %(epoch + 1, running_loss, acc))
        if acc >= best_acc:
            print('model is improve so dump model')
            joblib.dump(model,'./checkpoint/bert_model.pkl')

if __name__ == "__main__":
    train(model,trainloader)
