import flwr as fl
import torch
import numpy as np
from collections import OrderedDict
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score

class HealthcareClient(fl.client.NumPyClient):
    def __init__(
        self,
        model,
        trainloader,
        valloader,
        device: torch.device | None = None,
        optimizer=None,
        privacy_engine=None,
        lr: float = 5e-5,
    ):
        """
        Initializes the hospital client with its local model and private data.
        """
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = model.to(self.device)
        self.trainloader = trainloader
        self.valloader = valloader
        self.privacy_engine = privacy_engine
        
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr
            )
        else:
            self.optimizer = optimizer

    def get_parameters(self, config):
        """
        Extracts only the trainable LoRA matrices to send to the central server.
        The 65 million frozen base parameters stay safely on the local machine.
        """
        return [val.detach().cpu().numpy() for _, val in self.model.named_parameters() if val.requires_grad]

    def set_parameters(self, parameters):
        """
        Receives the newly averaged global LoRA matrices from the server 
        and updates the local model.
        """
        params_dict = zip([name for name, val in self.model.named_parameters() if val.requires_grad], parameters)
        state_dict = OrderedDict({k: torch.tensor(v, device=self.device) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        """
        The core loop: Receives global weights, trains locally for 1 epoch, 
        and sends the new weights back.
        """
        # 1. Update local model with the server's global weights
        self.set_parameters(parameters)
        
        # 2. Train on the hospital's private data
        self.model.train()
        
        for batch in self.trainloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

        # 3. Return the updated LoRA weights, the number of local examples, and any extra metrics
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """
        Tests how well the newly updated global model performs on this hospital's local validation data.
        """
        self.set_parameters(parameters)
        self.model.eval()
        
        loss = 0.0
        all_preds = []
        all_labels = []
        all_logits = []

        with torch.no_grad():
            for batch in self.valloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss += outputs.loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(predictions.detach().cpu().numpy())
                all_labels.extend(batch["labels"].detach().cpu().numpy())
                all_logits.append(outputs.logits.detach().cpu())

        total = len(all_labels)
        if total == 0:
            return 0.0, 0, {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "mcc": 0.0,
                "auc_roc": 0.0,
            }

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_logits = torch.cat(all_logits, dim=0)

        accuracy = float((all_preds == all_labels).sum() / total)
        precision = float(precision_score(all_labels, all_preds, zero_division=0))
        recall = float(recall_score(all_labels, all_preds, zero_division=0))
        f1 = float(f1_score(all_labels, all_preds, zero_division=0))
        mcc = float(matthews_corrcoef(all_labels, all_preds))

        probs = torch.softmax(all_logits, dim=-1)[:, 1].numpy()
        try:
            auc_roc = float(roc_auc_score(all_labels, probs))
        except ValueError:
            auc_roc = 0.0

        avg_loss = loss / len(self.valloader) if len(self.valloader) > 0 else 0.0
        # NOTE: Only aggregate statistics (single floats) are returned. Never add per-sample metrics here — that would leak private data.
        return avg_loss, total, {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mcc": mcc,
            "auc_roc": auc_roc,
        }