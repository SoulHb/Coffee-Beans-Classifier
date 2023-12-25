import torch.nn as nn
import torchvision.transforms as transforms
import argparse
import mlflow
import mlflow.pytorch
import sys
sys.path.append("src")
from config import *
from dataset import CoffeeBeans
from model import Resnet
torch.manual_seed(0)


#Create train and test loops
def train_loop(train_loader, loss, optimizer, model):
    """
            Training loop for a neural network.

            Args:
                model (torch.nn.Module): The neural network model.
                optimizer (torch.optim.Optimizer): The optimizer.
                loss (torch.nn.Module): The loss function.
                train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.

            Returns:
                tuple: Tuple containing the total loss, number of correct predictions, and total samples processed.
            """
    size_dataset = len(train_loader.dataset)
    model.train()
    correct, train_loss = 0, 0
    for X, y in train_loader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        y = torch.nn.functional.one_hot(y)
        y = y.argmax(1)
        pred = model(X)
        L = loss(pred, y)
        optimizer.zero_grad()
        L.backward()
        optimizer.step()
        train_loss += L.item() * X.shape[0]
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    train_loss = (train_loss / size_dataset)
    accuracy = (correct / size_dataset) * 100
    print(f"train_loss/epoch: {train_loss}, accuracy/epoch: {accuracy}")


def val_loop(val_loader, loss, model):
    """
            Validation loop for a neural network.

            Args:
                model (torch.nn.Module): The neural network model.
                loss (torch.nn.Module): The loss function.
                val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.

            Returns:
                tuple: Tuple containing the total loss, number of correct predictions, and total samples processed.
            """
    model.eval()
    size_dataset = len(val_loader.dataset)
    correct, test_loss = 0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            y = torch.nn.functional.one_hot(y)
            y = y.argmax(1)
            pred = model(X)
            test_loss += loss(pred, y).item() * X.shape[0]
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    val_loss = (test_loss / size_dataset)
    accuracy = (correct / size_dataset) * 100
    print(f"val_loss/epoch: {val_loss}, accuracy/epoch: {accuracy}")


def main(args):
    """
        Main function for training a neural network on a custom dataset.

        Args:
            args (dict): Dictionary containing command-line arguments or default values.
                Possible keys: 'data_path', 'saved_model_path', 'image_height', 'image_width',
                               'epoch', 'lr', 'batch_size'.

        Returns:
            None
        """
    dataset_path = args["data_path"] if args["data_path"] else DATA_PATH
    csv_file = os.path.join(dataset_path, 'Coffee Bean.csv')
    save_model_path = args["saved_model_path"] if args["saved_model_path"] else SAVED_MODEL_FOLDER

    image_height = args["image_height"] if args["image_height"] else IMAGE_SIZE[0]
    image_width = args["image_width"] if args["image_width"] else IMAGE_SIZE[1]
    epochs = args['epoch'] if args['epoch'] else EPOCHS
    lr = args['lr'] if args['lr'] else LR
    batch_size = args['batch_size'] if args['batch_size'] else BATCH_SIZE
    # Load and transform data
    transform = transforms.Compose([transforms.Resize((image_height, image_width)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
    coffee_beans_dataset = CoffeeBeans(dataset_path, csv_file, transform=transform)
    train_dataset, test_dataset = coffee_beans_dataset.train_test_loader()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    model = Resnet()
    model.to(DEVICE)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Set experiment name and ports
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment("Metrics")
    # Training loop with MLflow tracking
    with mlflow.start_run() as run:
        for i in range(epochs):
            train_loop(train_loader, loss, optimizer, model)
            val_loop(test_loader, loss, model)
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'{save_model_path}/ResNet_{i}_epoch.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        help='Specify path to your dataset')
    parser.add_argument("--saved_model_path", type=str,
                        help='Specify path for save models, where models folder will be created')
    parser.add_argument("--epoch", type=int,
                        help='Specify epoch for model training')
    parser.add_argument("--batch_size", type=int,
                        help='Specify batch size for model training')
    parser.add_argument("--lr", type=float,
                        help='Specify learning rate')
    parser.add_argument("--image_height", type=float,
                        help='Specify image height')
    parser.add_argument("--image_width", type=float,
                        help='Specify image width')
    args = parser.parse_args()
    args = vars(args)
    main(args)
