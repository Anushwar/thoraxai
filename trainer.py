# trainer.py
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from metrics import compute_metrics
from config import DEVICE

class ModelTrainer:
    def __init__(self, model, criterion, learning_rate, weight_decay):
        self.model = model
        self.criterion = criterion
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.1, patience=5, verbose=True
        )

    def train_and_evaluate(self, train_dataloader, test_dataloader, num_epochs=30):
        best_metrics = {'mean_auc': 0}
        best_epoch = 0
        patience = 15
        no_improve = 0

        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            running_loss = self._train_epoch(train_dataloader, epoch, num_epochs)
            
            # Evaluation phase
            metrics = self._evaluate(test_dataloader)
            
            self._update_training_state(metrics, epoch, running_loss, train_dataloader)
            
            if self._should_stop(metrics, best_metrics, no_improve, patience):
                break
            
            best_metrics, best_epoch, no_improve = self._update_best_metrics(
                metrics, best_metrics, epoch, no_improve
            )

        return best_metrics

    def _train_epoch(self, dataloader, epoch, num_epochs):
        running_loss = 0.0
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        return running_loss

    def _evaluate(self, dataloader):
        self.model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Evaluating"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(inputs)
                y_pred.append(torch.sigmoid(outputs).cpu().numpy())
                y_true.append(labels.cpu().numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        return compute_metrics(y_true, y_pred)

    def _update_training_state(self, metrics, epoch, running_loss, dataloader):
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | AUC: {metrics['mean_auc']:.4f}")
        self.scheduler.step(metrics['mean_auc'])

    def _should_stop(self, metrics, best_metrics, no_improve, patience):
        if metrics['mean_auc'] <= best_metrics['mean_auc']:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered. No improvement for {patience} epochs.")
                return True
        return False

    def _update_best_metrics(self, metrics, best_metrics, epoch, no_improve):
        if metrics['mean_auc'] > best_metrics['mean_auc']:
            best_metrics = metrics
            best_epoch = epoch
            no_improve = 0
            torch.save(self.model.state_dict(), 'best_model.pth')
            print(f"New best AUC: {best_metrics['mean_auc']:.4f} (Epoch {best_epoch+1})")
        return best_metrics, best_epoch, no_improve