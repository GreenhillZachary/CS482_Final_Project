# Cisco Dacanay
# Zach Greenhill

import torch
import numpy as np
import math
import joblib
import sys
import warnings

def main():
    # Silence warnings
    warnings.filterwarnings("ignore")

    # Variables
    price = -1
    show_price = True

    # Get price from CLI
    price = parse_args()

    # Get price from user if none was provided
    if price <= 0:
        price = getInput()
        show_price = False

    # If valid price was provided then run algorithm
    if price > 0:
        prediction = run_algorithm(price)
        if show_price:
            print(f"Current price ${price:.2f}")
        print(f"Predicted price in one month: ${prediction:.2f}")
    # Invalid price, exit
    else:
        print("Invalid value, please enter a value greater than 0.\nExiting script")

    

def run_algorithm(price):
    # Normalization values
    SCALAR = 89.819
    CONST = 80.245

    # Get dataset
    data = get_data()

    # Find associated entry
    value = (price - CONST) / SCALAR
    dif = math.inf
    index = 0
    for i in range(len(data)):
        temp = abs(data[i][49] - value)
        if temp < dif:
            dif = temp
            index = i
    pred_sample = data[index]
    pred_sample[49] = value

    # Run model and get prediction
    randforest = joblib.load("./Models/random_forest.joblib")
    test_prediction = randforest.predict(pred_sample.reshape(1, -1))

    return test_prediction[0]

def get_data():
    test_set = torch.load("./Datasets/test_set.pt", weights_only=False)
    train_set = torch.load("./Datasets/train_set.pt", weights_only=False)
    val_set = torch.load("./Datasets/val_set.pt", weights_only=False)

    test, _ = separate_dataset(test_set)
    train, _ = separate_dataset(train_set)
    val, _ = separate_dataset(val_set)

    return np.vstack((test, train, val))

def parse_args():
    price = -1
    n = len(sys.argv)

    if n > 1:
        price = float(sys.argv[1])
        
    return price

def getInput():
    i = input(f"Enter current price: $")
    return float(i)

# Function to seperate
def separate_dataset(dataset):
    # Initialize lists to hold data and labels
    data_list = []
    label_list = []

    # Iterate through each tuple in the dataset
    for data_tensor, label_tensor in dataset:
        # Convert tensors to NumPy arrays and append to the respective lists
        data_list.append(data_tensor.cpu().numpy())  # Move to CPU if needed
        label_list.append(label_tensor.cpu().numpy())

    # Convert lists to NumPy arrays
    data_array = np.array(data_list)
    label_array = np.array(label_list)

    return data_array, label_array

if __name__ == "__main__":
    main()