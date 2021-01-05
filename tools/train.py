import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_from_item(item):
    x = item[0].to(device).long()
    target_id = item[1].to(device).long()
    label = item[2].to(device).float()
    target_mask = (target_id != 0)
    return x, target_id, label, target_mask

def update_stats(tbar, train_loss, loss, output, label, num_corrects, num_total, labels, outs):
    train_loss.append(loss.item())
    pred = (torch.sigmoid(output) >= 0.5).long()
    num_corrects += (pred == label).sum().item()
    num_total += len(label)
    labels.extend(label.view(-1).data.cpu().numpy())
    outs.extend(output.view(-1).data.cpu().numpy())
    tbar.set_description('loss - {:.4f}'.format(loss))
    return num_corrects, num_total

def train_epoch(model, dataloader, optim, criterion, scheduler, device="cpu"):
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    tbar = tqdm(dataloader)
    for item in tbar:
        # recall, exercise_id, label : (batch_size, seq_len)
        recall, exercise_id, label, target_mask = load_from_item(item)

        optim.zero_grad()
        output, _ = model(recall, exercise_id)

        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)

        loss = criterion(output, label)
        loss.backward()
        optim.step()
        scheduler.step()

        tbar.set_description('loss - {:.4f}'.format(loss))


def val_epoch(model, val_iterator, criterion, device="cpu"):
    model.eval()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    tbar = tqdm(val_iterator)
    for item in tbar:
        x, target_id, label, target_mask = load_from_item(item)

        with torch.no_grad():
            output, atten_weight = model(x, target_id)

        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)

        loss = criterion(output, label)

        num_corrects, num_total = update_stats(tbar, train_loss, loss, output, label, num_corrects, num_total, labels,
                                               outs)

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.average(train_loss)

    return loss, acc, auc

def do_train(model,train_dataloader,val_dataloader,EPOCHS, LR, MODEL_PATH):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR,
                                                    steps_per_epoch=len(train_dataloader), epochs=EPOCHS)
    model.to(device)
    criterion.to(device)
    best_auc = 0.0
    for epoch in range(EPOCHS):
        train_epoch(model, train_dataloader, optimizer, criterion, scheduler, device)
        val_loss, avl_acc, val_auc = val_epoch(model, val_dataloader, criterion, device)
        print(f"epoch - {epoch + 1} val_loss - {val_loss:.3f} acc - {avl_acc:.3f} auc - {val_auc:.3f}")
        if best_auc < val_auc:
            print(f'epoch - {epoch + 1} best model with val auc: {val_auc}')
            best_auc = val_auc
            torch.save(model.state_dict(), MODEL_PATH)