## Coffee Beans Classifier

### Overview
This documentation provides information about the Coffee Beans Classifier project, including the data used, the methods and ideas employed, and the accuracy achieved. It also includes usage instructions and author information.


### Data
The dataset used for training and scoring is loaded with pytorch and consists images with coffee beans.

[Link to the dataset on Kaggle](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224)
## Model Architecture
The Coffee Beans Classifier neural network model is built using the [ResNet-34](https://arxiv.org/abs/1512.03385) architecture.
## Accuracy
After training, the model achieved an accuracy of 95% on the validation set.
## Usage
### Requirements

- Python 3.10

### Getting Started
Clone repository
```bash
git clone git@github.com:SoulHb/Coffee-Beans-Classifier.git
```
Move to project folder
```bash
cd Coffee-Beans-Classifier
```
Install dependencies
```bash
pip install -r requirements.txt
```
### Training
The model is trained on the provided dataset using the following configuration:
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 8
- Number of epochs: 10

Move to src folder
```bash
cd src
```
Run Mlflow for checking metrics:
```bash
mlflow server --host 127.0.0.1 --port 8080
```
Run train.py
```bash
python train.py --saved_model_path your_model_path --epochs 10 --lr 0.001 --batch_size 32 /path/to/Examples /path/to/masks
```

## Inference
To use the trained model for Coffee Beans Classifier classification, follow the instructions below:

### Without docker:
Move to src folder
```bash
cd src
```
Run Flask api
```bash
python inference.py --saved_model_path /path/to/your/saved/model
```
Run streamlit ui
```bash
python ui.py
```

Open streamlit ui in browser
```bash
streamlit run /your_path/Coffee-Beans-Classifier/src/ui.py
```
### With docker:

Run docker-compose
 ```bash
docker-compose -f docker_compose.yml up
```

## Author
This Coffee Beans Classifier project was developed by Namchuk Maksym. If you have any questions, please contact me: namchuk.maksym@gmail.com
