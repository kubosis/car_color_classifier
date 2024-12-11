import copy

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(self, dataset_dict: dict, batch_size: int = 32, patience: int = 3, verbose: bool = False):
        self._trn_loss_epochs = []
        self._trn_acc_epochs = []
        self._val_loss_epochs = []
        self._val_acc_epochs = []
        self._test_acc = []
        self._test_labels = []
        self._test_correct_labels = []
        self._patience = patience
        self._verbose = verbose
        self._dataset_dict = dataset_dict
        self.data_loaders = {
            'train': DataLoader(dataset_dict['train'], batch_size=batch_size, shuffle=True),
            'test': DataLoader(dataset_dict['test'], batch_size=batch_size, shuffle=False),
            'val': DataLoader(dataset_dict['val'], batch_size=batch_size, shuffle=False),
        }

    def _print(self, message, **kwargs):
        """Helper function to print messages if verbose is True."""
        if self._verbose:
            print(message, **kwargs)

    def train(self, model, criterion, optimizer, device, num_epochs=10):
        self._print(f"[INFO] training on {device} for {num_epochs} epochs")

        # Early stopping variables
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = float('inf')
        early_stop_counter = 0
        best_epoch = 0

        for epoch in range(num_epochs):
            self._print(f'[INFO] Epoch {epoch + 1}/{num_epochs}')

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                dataloader = self.data_loaders[phase]
                # just for the progress bar (on my pc it is terribly slow)
                if self._verbose:
                    iterator = tqdm(dataloader, desc=f'[{phase.upper()}] Epoch {epoch + 1}', leave=True)
                else:
                    iterator = dataloader

                for inputs, labels in iterator:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(self._dataset_dict[phase])
                epoch_acc = running_corrects.double() / len(self._dataset_dict[phase])

                if phase == 'train':
                    self._trn_loss_epochs.append(epoch_loss)
                    self._trn_acc_epochs.append(epoch_acc)
                    self._print(f'[{phase.upper()}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                else: # phase == val
                    self._val_loss_epochs.append(epoch_loss)
                    self._val_acc_epochs.append(epoch_acc)

                    # Check for improvement in the validation loss
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(model.state_dict())
                        early_stop_counter = 0  # Reset counter
                        best_epoch = epoch
                    else:
                        early_stop_counter += 1
                    self._print(f'[{phase.upper()}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}, best loss: {best_loss:.4f} in epoch: {best_epoch+1}')



            # Early stopping
            if early_stop_counter >= self._patience:
                self._print(f'[INFO] Early stopping triggered at epoch {epoch + 1}, loading model from epoch {best_epoch + 1}')
                break

        self._print("[SUCCESS] model trained successfully")
        # Load the best model weights
        model.load_state_dict(best_model_wts)
        return model

    def test(self, model, device):
        self._test_acc = 0
        self._test_labels = []
        self._test_correct_labels = []

        self._print(f"[INFO] testing on {device}")
        iterator = tqdm(self.data_loaders['test'], desc=f'[TEST] ', leave=True)

        running_corrects = 0
        for inputs, labels in iterator:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)

            # data for final visualization and discussion
            self._test_labels.extend(preds.cpu().numpy().tolist())
            self._test_correct_labels.extend((preds == labels.data).cpu().numpy().tolist())

        self._test_acc = running_corrects.double() / len(self._dataset_dict['test'])
        self._print(f"[SUCCESS] Testing ended with accuracy {(self._test_acc*100):.2f}%")


    @property
    def trn_loss_epochs(self):
        return self._trn_loss_epochs

    @property
    def trn_acc_epochs(self):
        return self._trn_acc_epochs

    @property
    def val_loss_epochs(self):
        return self._val_loss_epochs

    @property
    def val_acc_epochs(self):
        return self._val_acc_epochs

    @property
    def test_acc(self):
        return self._test_acc

    @property
    def test_labels(self):
        return self._test_labels

    @property
    def test_correct_labels(self):
        return self._test_correct_labels
