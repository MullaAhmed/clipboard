import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter  # Import the SummaryWriter

class CustomTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

start=0
end=1000

df=pd.read_csv("final.csv")
texts=df["Sentences"].values[start:end]
labels=df["Labels"].values[start:end]


# Convert labels to numerical values
label_mapping = {label: idx for idx, label in enumerate(set(labels))}
numeric_labels = [label_mapping[label] for label in labels]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(texts, numeric_labels, test_size=0.2, random_state=42, stratify=numeric_labels)

# Tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

# Parameters
max_len = 256
batch_size = 16

# Dataset and DataLoader
train_dataset = CustomTextDataset(X_train, y_train, tokenizer, max_len)
test_dataset = CustomTextDataset(X_test, y_test, tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_mapping))
model.to(device)

# Optimizer and scheduler
num_epochs = 3
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Initialize the SummaryWriter
writer = SummaryWriter('runs/bert_experiment')

def evaluate_model(model, dataloader, device):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()

            predictions.extend(np.argmax(logits, axis=1))
            true_labels.extend(label_ids)

    return accuracy_score(true_labels, predictions), classification_report(true_labels, predictions, target_names=list(label_mapping.keys()))
from tqdm import tqdm
for epoch in tqdm(range(num_epochs)):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        model.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backpropagation
        loss.backward()

        # Log gradients of parameters and parameter norms
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and parameter.grad is not None:
                writer.add_histogram(f"Gradients/{name}", parameter.grad, epoch * len(train_loader) + batch_idx)
                grad_norm = parameter.grad.norm()
                writer.add_scalar(f"Gradient_norm/{name}", grad_norm, epoch * len(train_loader) + batch_idx)

        # Clip gradients to prevent exploding gradient problem in deep networks
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()

        # Log training loss
        writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + batch_idx)
        # Log learning rate
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Learning Rate', current_lr, epoch * len(train_loader) + batch_idx)

    avg_train_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss:.4f}')
    
    # Optionally log histograms of model parameters
    for name, parameter in model.named_parameters():
        writer.add_histogram(f"Parameters/{name}", parameter, epoch)

# Evaluation and logging accuracy
accuracy, report = evaluate_model(model, test_loader, device)
writer.add_scalar('Accuracy', accuracy, num_epochs)

print(f"Accuracy: {accuracy}\n")
print("Classification Report:\n", report)
