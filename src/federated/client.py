import flwr as fl
import torch
from collections import OrderedDict

class HealthcareClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader, optimizer=None, privacy_engine=None, lr=5e-5):
        """
        Initializes the hospital client with its local model and private data.
        """
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        # Auto-detect your Mac's GPU (MPS) or fallback to CPU
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
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
        
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in self.valloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss += outputs.loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == batch["labels"]).sum().item()
                total += len(batch["labels"])
                
        accuracy = correct / total if total > 0 else 0
        # NOTE: Only aggregate statistics (single floats) are returned. Never add per-sample metrics here — that would leak private data.
        return loss / len(self.valloader), total, {"accuracy": accuracy}