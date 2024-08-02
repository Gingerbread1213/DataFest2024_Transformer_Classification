from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

import torch
import torch.nn as nn
import math
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt


X_class, y_class = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)
X_reg, y_reg = make_regression(n_samples=1000, n_features=20, n_informative=10, noise=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)



# Random Forest

def tryRandomForest(reg = True, n_estimators=100, X_train = None, y_train = None, X_test = None, y_test = None):

    if(reg):
        # Initialize the RandomForestRegressor
        rf_regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

        # Train the model
        rf_regressor.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = rf_regressor.predict(X_test)

        # Calculate the mean squared error of the model
        mse = mean_squared_error(y_test, predictions)

        print(f'Mean Squared Error of the RandomForest regression model: {mse:.2f}')
    else:
        # Initialize the RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)

        # Train the model
        rf.fit(X_train, y_train)

        # Make predictions on the test set
        predictions = rf.predict(X_test)

        # Calculate the accuracy of the model
        accuracy = accuracy_score(y_test, predictions)

        print(f'Accuracy of the RandomForest model: {accuracy:.2f}')

# tryRandomForest(X_train=X_train_reg, y_train=y_train_reg, X_test=X_test_reg, y_test=y_test_reg)



# Xgboost

def tryXgboost(objective = 'reg:squarederror', n_estimators=100, X_train = None, y_train = None, X_test = None, y_test = None, seed = 123):
        

        if(objective == 'reg:squarederror'):
                xgb_reg = xgb.XGBRegressor(objective=objective, n_estimators=100, seed=42)
                # Train the model
                xgb_reg.fit(X_train, y_train)
                # Predict labels for the test set
                predictions = xgb_reg.predict(X_test)
                # Evaluate the accuracy
                mse = mean_squared_error(y_test, predictions)
                print(f"XGboost Regression MSE: {mse:.2f}")
        else:
                xgb_clf = xgb.XGBClassifier(objective=objective, n_estimators=n_estimators, seed=seed)
                # Train the model
                xgb_clf.fit(X_train, y_train)
                # Predict labels for the test set
                predictions = xgb_clf.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                print(f"XGboost Multi-Class Classification Accuracy: {accuracy:.2f}")

# tryXgboost(X_train=X_train_reg, y_train=y_train_reg, X_test=X_test_reg, y_test=y_test_reg)



# SVM
# classification only

def trySVM(kernel = 'linear', random_state=42, X_train = None, y_train = None, X_test = None, y_test = None):

    svm_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    for kernel in svm_kernels:
    
        svm_classifier = SVC(kernel=kernel, random_state=42)

        # Train the SVM classifier
        svm_classifier.fit(X_train, y_train)

        # Predict labels for the test set
        predictions = svm_classifier.predict(X_test)

        # Evaluate the accuracy
        accuracy = accuracy_score(y_test, predictions)
        print(f"Classification Accuracy of {kernel}: {accuracy:.2f}")


# trySVM(X_train=X_train_class, y_train=y_train_class, X_test=X_test_class, y_test=y_test_class)


def tryDecisionTree(reg=True, random_state=42, X_train=None, y_train=None, X_test=None, y_test=None):
    if reg:
        # Initialize and train a Decision Tree Regressor
        dt_regressor = DecisionTreeRegressor(random_state=random_state)
        dt_regressor.fit(X_train, y_train)
        predictions = dt_regressor.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Decision Tree Regression MSE: {mse:.2f}")
    else:
        # Initialize and train a Decision Tree Classifier
        dt_classifier = DecisionTreeClassifier(random_state=random_state)
        dt_classifier.fit(X_train, y_train)
        predictions = dt_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Decision Tree Classification Accuracy: {accuracy:.2f}")


def tryLogisticRegression(random_state=42, X_train=None, y_train=None, X_test=None, y_test=None):
    # Initialize and train Logistic Regression
    log_reg = LogisticRegression(random_state=random_state)
    log_reg.fit(X_train, y_train)
    predictions = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Logistic Regression Accuracy: {accuracy:.2f}")

def tryKNN(k=5, reg=True, X_train=None, y_train=None, X_test=None, y_test=None):
    if reg:
        # Initialize and train a K-Nearest Neighbors Regressor
        knn_regressor = KNeighborsRegressor(n_neighbors=k)
        knn_regressor.fit(X_train, y_train)
        predictions = knn_regressor.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"KNN Regression MSE: {mse:.2f}")
    else:
        # Initialize and train a K-Nearest Neighbors Classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(X_train, y_train)
        predictions = knn_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"KNN Classification Accuracy: {accuracy:.2f}")


def tryGaussianNB(X_train=None, y_train=None, X_test=None, y_test=None):
    # Initialize and train Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    predictions = gnb.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Gaussian Naive Bayes Accuracy: {accuracy:.2f}")


def tryAdaBoost(reg=True, n_estimators=50, random_state=42, X_train=None, y_train=None, X_test=None, y_test=None):
    if reg:
        # Initialize and train an AdaBoost Regressor
        ada_regressor = AdaBoostRegressor(n_estimators=n_estimators, random_state=random_state)
        ada_regressor.fit(X_train, y_train)
        predictions = ada_regressor.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"AdaBoost Regression MSE: {mse:.2f}")
    else:
        # Initialize and train an AdaBoost Classifier
        ada_classifier = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state)
        ada_classifier.fit(X_train, y_train)
        predictions = ada_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"AdaBoost Classification Accuracy: {accuracy:.2f}")

# Simple Transformer

class SimpleTransformerModel(nn.Module):
    def __init__(self, input_size, num_classes, mode='classification', apply_softmax=False):
        super(SimpleTransformerModel, self).__init__()
        self.mode = mode
        self.apply_softmax = apply_softmax
        self.encoder = nn.Linear(input_size, 128)
        self.transformer_block = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.classifier = nn.Linear(128, num_classes)
        self.regressor = nn.Linear(128, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = x.unsqueeze(1)  # Add a sequence dimension
        x = self.transformer_block(x)
        x = x.squeeze(1)
        if self.mode == 'classification':
            x = self.classifier(x)
            if self.apply_softmax:
                x = F.softmax(x, dim=1)
            return x
        elif self.mode == 'regression':
            return self.regressor(x)

def prepare_data(mode, X_train, y_train, X_test, y_test):
    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    if mode == 'classification':
        # Convert one-hot encoded labels back to class indices
        y_train = torch.tensor(y_train.idxmax(axis=1).apply(lambda x: ['status_On Progress', 'status_Risky', 'status_Fall Behind'].index(x)).values, dtype=torch.long)
    elif mode == 'regression':
        y_train = torch.tensor(y_train.values, dtype=torch.float)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return loader

# Model Initialization
model_class = SimpleTransformerModel(input_size=20, num_classes=2, mode='classification')
model_reg = SimpleTransformerModel(input_size=20, num_classes=1, mode='regression')

# Optimizers
optimizer_class = optim.Adam(model_class.parameters(), lr=0.001)
optimizer_reg = optim.Adam(model_reg.parameters(), lr=0.001)

# Loss Functions
criterion_class = nn.CrossEntropyLoss()
criterion_reg = nn.MSELoss()

def train_model(model, data_loader, optimizer, criterion, epochs, mode):
    epoch_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        sum_squared_error = 0.0

        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            if mode == 'classification':
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)
            elif mode == 'regression':
                # Calculate sum of squared errors for MSE calculation
                sum_squared_error += ((outputs.squeeze() - targets) ** 2).sum().item()
                total_samples += targets.size(0)

        epoch_loss = total_loss / total_samples
        epoch_losses.append(epoch_loss)
        if mode == 'classification':
            accuracy = total_correct / total_samples * 100
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
        elif mode == 'regression':
            mse = sum_squared_error / total_samples
            print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, MSE: {mse:.2f}")
    if mode == 'classification':
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
    elif mode == 'regression':
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, MSE: {mse:.2f}")
    return(epoch_losses)


# Initialize and train classification model
# print("Training Classifier:")
# class_loader = prepare_data('classification', X_train_class, y_train_class, X_test_class, y_test_class)
# train_model(model_class, class_loader, optimizer_class, criterion_class, 30, 'classification')

# print("Training Regressor:")
# reg_loader = prepare_data('regression', X_train_reg, y_train_reg, X_test_reg, y_test_reg)
# train_model(model_reg, reg_loader, optimizer_reg, criterion_reg, 30, 'regression')

