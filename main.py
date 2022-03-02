"""
author: Diaaeldin Shalaby
"""
import numpy as np
import torch
import tqdm
import dill
from torch.utils.tensorboard.writer import SummaryWriter
from CNN import SimpleCNN
from data_proc import *

np.random.seed(0)  # Set a known random seed for reproducibility
torch.random.manual_seed(0)  # Set a known random seed for reproducibility


def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device):
    """
    Function for evaluation of a model `model` on the data in `dataloader` on device `device`

    @param model: best_model.pt
    @param dataloader: from [train, test, validation]
    @param device: from ['cpu', 'cuda']
    """
    # Define a loss (mse loss)
    mse = torch.nn.MSELoss()
    # We will accumulate the mean loss in variable `loss`
    loss = torch.tensor(0., device=device)
    with torch.no_grad():  # We do not need gradients for evaluation
        # Loop over all samples in `dataloader`
        for data in tqdm.tqdm(dataloader, desc="scoring", position=0):
            # Get a sample and move inputs and targets to device
            try:
                inputs, targets, labels = data
            except:
                pass
            inputs = inputs.to(device)
            targets = targets.to(device)

            knowns = inputs[:, 1, :, :]

            # Get outputs for network
            outputs = model(inputs)

            preds = outputs[:, 0, :, :]
            n_samples = inputs.shape[0]
            stacked_preds = torch.zeros(size=(n_samples, 2475))
            k = 0
            for pred, kn in zip(preds, knowns):
                # print(pred[kn == 0])
                masked = torch.clone(pred)[kn == 0]
                stacked_preds[k, :masked.shape[0]] = masked
                k += 1

            # Here we could clamp the outputs to the minimum and maximum values of inputs for better performance

            # Calculate mean mse loss over all samples in dataloader (accumulate mean losses in `loss`)
            loss += (torch.stack([mse(output, target) for output, target in zip(stacked_preds, targets)]).sum()
                     / len(dataloader.dataset))
    return loss


def main(learningrate: int = 1e-3, weight_decay: float = 1e-5, n_updates: int = 50):
    """Main training function that takes hyperparameters as input and performs training and evaluation of model"""
    # Initiate the whole dataset

    our_dataset = Image_dataset(4, 5)
    # only take 20 folders for debugging
    n_samples = len(our_dataset)

    # Shuffle integers from 0 n_samples to get shuffled sample indices
    shuffled_indices = np.random.permutation(len(our_dataset))

    testset_inds = shuffled_indices[int(n_samples / 10) * 9:int(n_samples)]
    validationset_inds = shuffled_indices[int(n_samples / 10) * 7:int(n_samples / 10) * 9]
    trainingset_inds = shuffled_indices[:int(n_samples / 10) * 7]

    # Create PyTorch subsets from our subset-indices
    testset = Subset(our_dataset, indices=testset_inds)
    validationset = Subset(our_dataset, indices=validationset_inds)
    trainingset = Subset(our_dataset, indices=trainingset_inds)

    # Create dataloaders from each subset
    testloader = DataLoader(testset,  # we want to load our dataset
                            shuffle=False,  # shuffle for training
                            batch_size=1,  # 1 sample at a time
                            num_workers=0,  # no background workers
                            collate_fn=stack_images_arrays
                            )
    valloader = DataLoader(validationset,  # we want to load our dataset
                           shuffle=False,  # shuffle for training
                           batch_size=1,  # stack 4 samples to a minibatch
                           num_workers=0,
                           collate_fn=stack_images_arrays
                           )
    trainloader = DataLoader(trainingset,  # we want to load our dataset
                             shuffle=True,  # shuffle for training
                             batch_size=1,  # stack 4 samples to a minibatch
                             num_workers=0,
                             collate_fn=stack_images_arrays
                             )

    # Define a tensorboard summary writer that writes to directory "results_path/tensorboard"
    writer = SummaryWriter(log_dir=os.path.join(os.getcwd(), 'results', 'tensorboard'))

    # Create network
    # cnn = SimpleCNN(2, 3, 32, 7)

    # load trained model
    cnn = torch.load(os.path.join('results', 'best_model.pt'))

    # Get mse loss function
    mse = torch.nn.MSELoss()

    # Get adam optimizer
    optimizer = torch.optim.AdamW(cnn.parameters(), lr=learningrate, weight_decay=weight_decay)

    # print status to tensorboard every x updates
    print_stats_at = 100

    validate_at = n_updates/5  # evaluate model on validation set and check for new best model every x updates
    update = 0  # current update counter
    best_validation_loss = np.inf  # best validation loss so far
    update_progess_bar = tqdm.tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)  # progressbar

    # Save initial model as "best" model (will be overwritten later)
    # torch.save(cnn, os.path.join('results', 'best_model.pt'))

    while update < n_updates:
        for i, data in enumerate(trainloader):
            inputs, targets, labels = data

            knowns = inputs[:, 1, :, :]

            # Reset gradients
            optimizer.zero_grad()

            # Get outputs for network
            outputs = cnn(inputs)

            preds = outputs[:, 0, :, :]

            n_samples = inputs.shape[0]

            stacked_preds = torch.zeros(size=(n_samples, 2475))
            for k, data in enumerate(zip(preds, knowns)):
                pred, kn = data
                masked = pred[kn == 0]
                stacked_preds[k, :masked.shape[0]] = masked

            # Calculate loss, do backward pass, and update weights
            loss = mse(stacked_preds, targets)

            loss.backward()
            optimizer.step()

            # Print current status and score
            if update % print_stats_at == 0 and update > 0:
                writer.add_scalar(tag="training/loss",
                                  scalar_value=loss.cpu(),
                                  global_step=update)

            # Evaluate model on validation set
            if update % validate_at == 0 and update > 0:
                val_loss = evaluate_model(cnn, dataloader=valloader, device='cpu')
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss.cpu(), global_step=update)
                # Add weights as arrays to tensorboard
                for i, param in enumerate(cnn.parameters()):
                    writer.add_histogram(tag=f'validation/param_{i}', values=param.cpu(),
                                         global_step=update)
                # Add gradients as arrays to tensorboard
                for i, param in enumerate(cnn.parameters()):
                    writer.add_histogram(tag=f'validation/gradients_{i}',
                                         values=param.grad.cpu(),
                                         global_step=update)
                # Save best model for early stopping
                if best_validation_loss > val_loss:
                    best_validation_loss = val_loss
                    torch.save(cnn, os.path.join('results', 'best_model.pt'))

                update_progess_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
                update_progess_bar.update()

        torch.save(cnn, os.path.join('results', 'best_model.pt'))

        # Increment update counter, exit if maximum number of updates is reached
        update += 1
        if update >= n_updates:
            break
    update_progess_bar.close()

    print('Finished Training!')

    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    net = torch.load(os.path.join('results', 'best_model.pt'))
    device = 'cpu'
    test_loss = evaluate_model(net, dataloader=testloader, device=device)
    val_loss = evaluate_model(net, dataloader=valloader, device=device)
    train_loss = evaluate_model(net, dataloader=trainloader, device=device)

    print(f"Scores:")
    print(f"test loss: {test_loss}")
    print(f"validation loss: {val_loss}")
    print(f"training loss: {train_loss}")

    # Write result to file
    with open(os.path.join('results', 'results.txt'), 'w') as fh:
        print(f"Scores:", file=fh)
        print(f"test loss: {test_loss}", file=fh)
        print(f"validation loss: {val_loss}", file=fh)
        print(f"training loss: {train_loss}", file=fh)


def make_predictions(learningrate: int = 1e-3, weight_decay: float = 1e-5, ):
    # Get mse loss function
    mse = torch.nn.MSELoss()

    # load trained model
    cnn = torch.load(os.path.join('results', 'best_model.pt'))

    optimizer = torch.optim.Adam(cnn.parameters(), lr=learningrate, weight_decay=weight_decay)

    with open(r"S:\Programming in Python 2\Assigment 2\Excercise_5\testset_dataset\testset.pkl", 'rb') as test:
        testset = dill.load(test)

    # Create a torch testset from the pickle file
    processed_testset = Test_set(testset)


    testloader = DataLoader(processed_testset,  # we want to load our dataset
                            shuffle=False,  # shuffle for training
                            batch_size=1,  # 1 sample at a time
                            num_workers=0,  # no background workers
                            # pin_memory=True,
                            collate_fn=stack_images_arrays
                            )

    preds_list = []
    for data in testloader:
        inputs, targets, labels = data
        # Get outputs for network
        outputs = cnn(inputs)

        knowns = inputs[:, 1, :, :]

        # Reset gradients
        optimizer.zero_grad()

        preds = outputs[:, 0, :, :]

        n_samples = inputs.shape[0]
        stacked_preds = torch.zeros(size=(n_samples, 2475))
        k = 0
        for pred, kn in zip(preds, knowns):
            # print(pred[kn == 0])
            masked = torch.clone(pred)[kn == 0]
            x = masked.type(torch.uint8)
            preds_list.append(x.detach().numpy())
            stacked_preds[k, :masked.shape[0]] = masked
            k += 1

        loss = mse(stacked_preds, targets)

        loss.backward()
        optimizer.step()

    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    #     net = torch.load(os.path.join('results', 'best_model.pt'))

    test_loss = evaluate_model(cnn, dataloader=testloader, device='cpu')

    print(f"test loss: {test_loss}")

    with open('predictions.pkl', 'wb') as p:
        dill.dump(preds_list, p)

    print('DONE!!')


if __name__ == '__main__':
    main()
    make_predictions()

