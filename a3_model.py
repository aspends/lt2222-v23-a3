import argparse
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    parser.add_argument("authorfile", type=str, help="The file containing the table of authors and their corresponding IDs.")
    parser.add_argument("--hidden_size", type=int, default=256, help="The size of the hidden layer.")
    parser.add_argument("--nonlinearity", type=str, default="relu", choices=["relu", "tanh", "sigmoid"],
                        help="The nonlinearity function to use.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="The learning rate for training.")
    parser.add_argument("--epochs", type=int, default=10, help="The number of epochs to train for.")
    
    args = parser.parse_args()
    
    print("Reading {}...".format(args.featurefile))
    print("Done!")
    print("Reading {}...".format(args.authorfile))
    print("Done!")
    # Load the data into pandas DataFrames
    output_df = pd.read_csv(args.featurefile)
    authors_df = pd.read_csv(args.authorfile)

    # Merge the two DataFrames
    merged_df = output_df.merge(authors_df, left_index=True, right_index=True)

    # Split the merged DataFrame into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(merged_df.drop('label', axis=1), 
                                                        merged_df['label'], 
                                                        test_size=0.2, 
                                                        random_state=42)

    # Convert the training data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values).float()
    y_train_tensor = torch.tensor(pd.Categorical(y_train).codes).long()
    X_test_tensor = torch.tensor(X_test.values).float()
    y_test_tensor = torch.tensor(pd.Categorical(y_test).codes).long()
    print(X_train_tensor)
    print(y_train_tensor)
    # Define the model
    class Model(nn.Module):
        def __init__(self, input_dim, hidden_size, output_dim, nonlinearity):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_dim)
            if nonlinearity == 'relu':
                self.activation = nn.ReLU()
            elif nonlinearity == 'tanh':
                self.activation = nn.Tanh()
            elif nonlinearity == 'sigmoid':
                self.activation = nn.Sigmoid()

        def forward(self, x):
            x = self.fc1(x)
            x = self.activation(x)
            x = self.fc2(x)
            return nn.functional.log_softmax(x, dim=1)

    model = Model(X_train.shape[1], args.hidden_size, len(authors_df), args.nonlinearity) 
    print(authors_df)
    # Train the model
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        running_loss = 0.0
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print("Epoch {}: Loss = {:.4f}".format(epoch+1, running_loss/len(X_train)))


    # Test the model
    print("Testing the model")
    model.eval()
    with torch.no_grad():
        output = model(X_test_tensor)
        _, predicted = torch.max(output, 1)
        cm = confusion_matrix(y_test.astype(str), predicted.numpy().astype(str))
        print("Confision matrix", cm)
        print("Accuracy: {:.2f}%".format(100*numpy.trace(cm)/numpy.sum(cm)))


    sns.heatmap(cm, annot=True, cmap='Reds')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

