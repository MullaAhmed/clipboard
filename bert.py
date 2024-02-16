# Import necessary libraries
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

start=0
end=10000


df=pd.read_csv("final.csv")
texts=df["Sentences"].values[start:end]
labels=df["Labels"].values[start:end]


# Convert labels to numerical values (0, 1, 2, 3)
label_mapping = {label: idx for idx, label in enumerate(set(labels))}
numeric_labels = [label_mapping[label] for label in labels]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, numeric_labels, test_size=0.2, random_state=42,stratify=numeric_labels)

# Create a CountVectorizer to convert text data into a bag-of-words representation
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_mapping))

# Tokenize and encode the training and testing data
X_train_tokens = tokenizer(list(X_train), padding=True, truncation=True, return_tensors="pt")
X_test_tokens = tokenizer(list(X_test), padding=True, truncation=True, return_tensors="pt")

# Convert labels to PyTorch tensors
y_train_tensor = torch.tensor(y_train)
y_test_tensor = torch.tensor(y_test)

# Create DataLoader for training and testing data
train_dataset = TensorDataset(X_train_tokens.input_ids, X_train_tokens.attention_mask, y_train_tensor)
test_dataset = TensorDataset(X_test_tokens.input_ids, X_test_tokens.attention_mask, y_test_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Set up GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Set up optimizer and training parameters
optimizer = AdamW(model.parameters(), lr=2e-5)

# Train the model
num_epochs = 10
from tqdm import tqdm 
for epoch in tqdm(range(num_epochs)):
    model.train()
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'bert_model.pth')


# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()  # Set model to evaluation mode
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Move logits and labels to CPU if necessary
            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()

            predictions.extend(np.argmax(logits, axis=1).flatten())
            true_labels.extend(label_ids.flatten())

    return predictions, true_labels


# Perform prediction on the test dataset
test_predictions, test_true_labels = evaluate_model(model, test_dataloader)

# Calculate accuracy
accuracy = accuracy_score(test_true_labels, test_predictions)
print("Accuracy:", accuracy)


print("Classification Report:")
print(classification_report(test_true_labels, test_predictions, target_names=label_mapping.keys()))

