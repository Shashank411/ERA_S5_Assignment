# ERA - Session 5 Assignment

We are building a simple classifier model for the MNIST dataset. Our project comprises mainly of three files, which are -

## 1. S5.ipynb

This is the main notebook file of our project in which we train our classifier model and test its accuracy

## 2. model.py

This file contains the model definition - the layers involved along with the forward function

### usage

paste the below code in the S5 notebook to use the model


    from model import Net
    import torch
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    model = Net().to(device)


## 3. utils.py

This file contains the functions we use for gettting the data loaders. We can also define the train and test functions (for 1 epoch) used for generic model training and evaluation

### usage

paste the below code in the S5 notebook to import and use the functions required for training, testing and getting data loaders


    from utils import loaders, train, test
    batch_size = 512
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
    train_loader, test_loader = loaders(batch_size, kwargs)


after initialising model, device, optimizer and criterion in the S5 notebook, run the following -


    train_accuracy, train_loss = train(model, device, train_loader, optimizer, criterion)
    test_accuracy, test_loss = test(model, device, test_loader, criterion)

