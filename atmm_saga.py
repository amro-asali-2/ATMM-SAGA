import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# ========== Configuration ==========
NUM_EPOCHS = 100
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-9
SCORE_ACTIVATION = 'sigmoid'
MODEL_DIR = 'MODELS'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LengthNorm(nn.Module):
    """
    L2-normalization layer along the feature dimension.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=1)


class SReLU(nn.Module):
    """
    Structural ReLU activation with a learnable transformation matrix Wa.
    Applies Wa @ x and element-wise ReLU, with Wa initialized as identity.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.Wa = nn.Parameter(torch.eye(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transform inputs and apply ReLU
        transformed = x @ self.Wa.T
        return F.relu(transformed)


class SASVScoreAttention(nn.Module):
    """
    Spoofing-Aware Speaker Verification with Score-Aware Gated Attention.

    Integrates a countermeasure (CM) score into ASV embeddings via
    a gating mechanism to suppress spoofed samples.
    """
    def __init__(
        self,
        asv_embed_dim: int,
        cm_embed_dim: int,
        score_activation: str = 'sigmoid'
    ):
        """
        Args:
            asv_embed_dim: Dimension of ASV embeddings.
            cm_embed_dim: Dimension of CM embeddings.
            score_activation: Activation for final output ('sigmoid' supported).
        """
        super().__init__()
        self.name = "SAGA S1"
        self.score_activation = score_activation

        # Countermeasure (CM) feature extraction path
        self.cm_fc1 = nn.Linear(cm_embed_dim, cm_embed_dim)
        self.cm_srelu = SReLU(cm_embed_dim)
        self.cm_fc2 = nn.Linear(cm_embed_dim, cm_embed_dim)
        self.cm_fc3 = nn.Linear(cm_embed_dim, cm_embed_dim)
        self.length_norm = LengthNorm()
        self.cm_score_layer = nn.Linear(cm_embed_dim, 1)

        # ASV classification path
        hidden_dim = 2 * asv_embed_dim
        self.asv_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.asv_fc2 = nn.Linear(hidden_dim, hidden_dim)
        if self.score_activation == 'sigmoid':
            self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        asv_features: torch.Tensor,
        cm_embeddings: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """
        Forward pass for training.

        Args:
            asv_features: Tensor of shape (batch, 2 * _EMBED_DIM)
            cm_embeddings: Tensor of shape (batch, CM_EMBED_DIM)

        Returns:
            x_pred: Raw SASV output logits
            cm_score: Predicted CM score (logit)
        """
        # CM score computation
        x = self.cm_srelu(self.cm_fc1(cm_embeddings))
        x = self.cm_srelu(self.cm_fc2(x))
        x = self.length_norm(self.cm_fc3(x))
        cm_score = self.cm_score_layer(x)

        # ASV gating and classification
        r = F.relu(self.asv_fc1(asv_features))
        r = self.length_norm(r)
        gate = torch.sigmoid(cm_score)
        r = gate * r
        r = F.relu(self.asv_fc2(r))

        x_pred = self.output_layer(r)
        return x_pred, cm_score

    def inference(
        self,
        asv_features: torch.Tensor,
        cm_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Inference pass with gradients disabled.

        Returns final SASV probability if sigmoid activation is used.
        """
        self.eval()
        with torch.no_grad():
            x_pred, _ = self.forward(asv_features, cm_embeddings)
            if self.score_activation == 'sigmoid':
                return torch.sigmoid(x_pred)
            return x_pred

    def num_params(self) -> int:
        """
        Returns the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters())

    def get_name(self) -> str:
        """
        Retrieves the model's descriptive name.
        """
        return self.name



# ========== Parameter Groups ==========
def get_optimizers(model):
    cm_params = [
        {'params': model.cm_fc1.parameters()},
        {'params': model.cm_fc2.parameters()},
        {'params': model.cm_fc3.parameters()},
        {'params': model.cm_score_layer.parameters()},
        {'params': model.cm_srelu.parameters()},
        {'params': model.asv_fc2.parameters()},
        {'params': model.output_layer.parameters()}
    ]
    asv_params = [
        {'params': model.asv_fc1.parameters()},
        {'params': model.asv_fc2.parameters()},
        {'params': model.output_layer.parameters()}
    ]
    cm_optimizer = optim.Adam(cm_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    asv_optimizer = optim.Adam(asv_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    return [cm_optimizer, asv_optimizer]

# ========== Training Step ==========
def train_one_epoch(model, train_loader, optimizer, criterion, lambda_val, device):
    model.train()
    epoch_loss, loss_asv, loss_cm = 0.0, 0.0, 0.0
    num_batches = len(train_loader) // 100

    with tqdm(total=num_batches, desc="Training") as pbar:
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx > num_batches:
                break

            optimizer.zero_grad()
            enroll_embeds, t_asv, target_classification, t_cm, *_ = batch
            enroll_embeds = enroll_embeds.squeeze(1).squeeze(1).to(device).float()
            t_asv = t_asv.squeeze(1).squeeze(1).to(device).float()
            t_cm = t_cm.squeeze(1).squeeze(1).to(device).float()
            target_score_asv = target_classification[:, 0].unsqueeze(1).float().to(device)
            target_score_cm = 1 - target_classification[:, 2].unsqueeze(1).float().to(device)

            inputs = torch.cat((enroll_embeds, t_asv), dim=-1)
            pred_asv, pred_cm = model(inputs, t_cm)

            loss1 = criterion(pred_asv, target_score_asv)
            loss2 = criterion(pred_cm, target_score_cm)
            loss = lambda_val * loss1 + (1 - lambda_val) * loss2

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() / num_batches
            loss_asv += loss1.item() / num_batches
            loss_cm += loss2.item() / num_batches
            pbar.update(1)

    return epoch_loss, loss_asv, loss_cm

# ========== Main Training Loop ==========
def train(model, train_loaders, num_epochs, device):
    optimizers = get_optimizers(model)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(num_epochs):
        task_index = random.randint(0, 1)
        train_loader = train_loaders[task_index]
        optimizer = optimizers[task_index]
        lambda_val = [0.1, 0.9][task_index]

        epoch_loss, loss_asv, loss_cm = train_one_epoch(
            model, train_loader, optimizer, criterion, lambda_val, device
        )

        print(f"Epoch {epoch + 1}/{num_epochs}: Train [T: {epoch_loss:.4f}, ASV: {loss_asv:.4f}, CM: {loss_cm:.4f}]\n")

    torch.save(model.state_dict(), model_filename)
    print(f"Model saved to {model_filename}")

if __name__ == "__main__":
    # ========== Loaders & Model ==========
    train_loaders = [spf_trainloader, bnf_trainloader]
    model = SASVScoreAttention(asv_embed_dim=192, cm_embed_dim=160).to(device)
    model_filename = os.path.join(MODEL_DIR, f"{model.get_name()}.pth")
    train(model, train_loaders, NUM_EPOCHS, device)
