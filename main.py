import numpy as np
import pandas as pd
import torch
from torch import utils
from torch.utils import data
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os,shutil
from tqdm import tqdm
import matplotlib.pyplot as plt


#Create validation set
# os.chdir('platesv2/plates/train/')
# classes = ['cleaned','dirty']
# for plate_class in classes:
#     i = 0
#     for file in os.listdir(plate_class):
#         if file == ".DS_Store":
#             continue
#         if i % 6 == 0:
#             shutil.move('/Users/daniilavtusko/Desktop/Coding/CleanDirtyPlate/platesv2/plates/train/' + plate_class + '/' + file,'/Users/daniilavtusko/Desktop/Coding/CleanDirtyPlate/platesv2/plates/val/' + plate_class + '/')
#         i += 1


myTransforms = transforms.Compose([transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train_dataset = torchvision.datasets.ImageFolder('platesv2/plates/train/',myTransforms)
val_dataset = torchvision.datasets.ImageFolder('/Users/daniilavtusko/Desktop/Coding/CleanDirtyPlate/platesv2/plates/val/',myTransforms)
batch_size = 4
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)

print(len(train_dataloader),len(train_dataset))



model = models.resnet18(pretrained=True)

# Disable grad for all conv layers
for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(model.fc.in_features,2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3, weight_decay=0.0001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)




test_accuracy_history = []
test_loss_history = []
val_accuracy_history = []
val_loss_history = []
def train_model(model, loss, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                dataloader = train_dataloader
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                dataloader = val_dataloader
                model.eval()  # Set model to evaluate mode

            running_loss = 0.
            running_acc = 0.

            # Iterate over data.
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # forward and backward
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    loss_value = loss(preds, labels)
                    preds_class = preds.argmax(dim=1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                # statistics
                running_loss += loss_value.item()
                running_acc += (preds_class == labels.data).float().mean()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)
            ##
            if (phase == 'train'):
                test_accuracy_history.append(epoch_acc)
                test_loss_history.append(epoch_loss)
            elif (phase == 'val'):
                val_accuracy_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc), flush=True)

    return model

train_model(model,loss,optimizer,scheduler,20)

plt.plot(test_accuracy_history, label='test_acc')
plt.plot(val_accuracy_history, label='val_acc')
plt.legend()
plt.title('Accuracy')
plt.show()

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


test_dataset = ImageFolderWithPaths('platesv2/plates/test/',myTransforms)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)

model.eval()

test_predictions = []
test_img_paths = []
for inputs, labels, paths in tqdm(test_dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    with torch.set_grad_enabled(False):
        preds = model(inputs)
    test_predictions.append(
        torch.nn.functional.softmax(preds, dim=1)[:, 1].data.cpu().numpy())
    test_img_paths.extend(paths)

test_predictions = np.concatenate(test_predictions)

submission_df = pd.DataFrame.from_dict({'id': test_img_paths, 'label': test_predictions})
submission_df['label'] = submission_df['label'].map(lambda pred: 'dirty' if pred > 0.5 else 'cleaned')
submission_df['id'] = submission_df['id'].str.replace('test/unknown/', '')
submission_df['id'] = submission_df['id'].str.replace('.jpg', '')
submission_df.set_index('id', inplace=True)
submission_df.head(n=6)
submission_df.to_csv('submission.csv')