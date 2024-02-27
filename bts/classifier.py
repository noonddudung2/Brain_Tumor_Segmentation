import torch
import bts.loss as loss
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

import numpy as np

from datetime import datetime
from time import time

class BrainTumorClassifier():
    """ Returns a BrainTumorClassifier class object which represents our 
    optimizer for our network.
    """

    def __init__(self, model,device):
        """ Constructor for our BrainTumorClassifier class.
        Parameters:
            model(DynamicUNet): UNet model to be trained.
            device(torch.device): Device currently used for all computations.

        Returns: 
            None 
        """
        self.model = model
        self.device = device
        #Loss
        # self.criterion = loss.BCEDiceLoss(self.device).to(device)        
        self.criterion = loss.BCELoss().to(device)
        # self.criterion = loss.DiceLoss_2().to(device)


        self.log_path = datetime.now().strftime("%I-%M-%S_%p_on_%B_%d,_%Y")
    
    def dice_coefficient(self, predicted, target):
        """Calculates the Sørensen–Dice Coefficient for a
        single sample.
        Parameters:
            predicted(torch.Tensor): Predicted single output of the network.
                                    Shape - (Channel,Height,Width)
            target(torch.Tensor): Actual required single output for the network
                                    Shape - (Channel,Height,Width)

        Returns:
            coefficient(torch.Tensor): Dice coefficient for the input sample.
                                        1 represents high similarity and
                                        0 represents low similarity.
        """
        predicted = torch.tensor(predicted)
        target = torch.tensor(target)
        smooth = 1
        product = torch.mul(predicted, target)
        intersection = product.sum()
        coefficient = (2*intersection + smooth) / \
            (predicted.sum() + target.sum() + smooth)
        return coefficient
    

    def _calculate_iou(self,predicted, target):
        """
        Calculate Intersection over Union (IOU) for predicted and true masks.
        
        Parameters:
            predicted_mask (numpy.ndarray): Predicted mask, binary mask with values 0 or 1.
            true_mask (numpy.ndarray): True mask, binary mask with values 0 or 1.
        
        Returns:
            iou (float): Intersection over Union (IOU) score.
        """
        # Flatten masks to 1D arrays
        predicted = predicted.flatten()
        target = target.flatten()
        
        # Calculate intersection and union
        intersection = np.sum(predicted * target)
        union = np.sum(predicted) + np.sum(target) - intersection
        
        # Calculate IOU
        iou = intersection / union if union > 0 else 0.0
        
        return iou

    def train(self, epochs, trainloader, valloader, mini_batch=None, learning_rate=0.001, save_best=None, plot_image=None):
        """ Train the model using Adam Optimizer.
        Parameters:
            epochs(int): Number of epochs for the training session.
            trainloader(torch.utils.data.Dataloader): Training data
                        loader for the optimizer.
            valloader(torch.utils.data.Dataloader): Validation data
                        loader for evaluating the model performance.
            mini_batch(int): Used to print logs for epoch batches.
                            If None then batch logs won't be printed.
                            Default: None
            learning_rate(float): Learning rate for optimizer.
                                  Default: 0.001
            save_best(str): Path to save the best model. At the end of 
                            the training the epoch with lowest loss will
                            be saved. If None then model won't be saved.
                            Default: None
            plot_image(list): Plot some samples in Tensorboard while training.
                          Visualization of model training progress.If None
                          then nothing will be done.
                          Default: None

        Returns:
            history(dict): Contains information about training session.
                            'train_loss': List of loss at every epoch
                            'val_loss': List of validation loss at every epoch
        """
        # Tensorboard Writer
        self.tb_writer = SummaryWriter(log_dir=f'logs/{self.log_path}')
        # Training session history data.
        history = {'train_loss': list(), 'val_loss': list()}
        # For save best feature. Initial loss taken a very high value.
        last_loss = 1000
        # Optimizer used for training process. Adam Optimizer.
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # Reducing LR on plateau feature to improve training.
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.85, patience=2, verbose=True)
        print('Starting Training Process')
        # Epoch Loop
        for epoch in range(epochs):
            start_time = time()
            # Training a single epoch
            epoch_train_loss = self._train_epoch(trainloader, mini_batch)
            # Validating the model
            epoch_val_loss = self._validate_epoch(valloader)
            # Collecting all epoch loss values for future visualization.
            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)
            # # Logging to Tensorboard
            # self.tb_writer.add_scalar('Train Loss', epoch_train_loss, epoch)
            # self.tb_writer.add_scalar('Validation Loss', epoch_val_loss, epoch)
            # self.tb_writer.add_scalar(
            #     'Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)
            # Reduce LR On Plateau
            self.scheduler.step(epoch_val_loss)

            # Plotting some sample output on TensorBoard for visualization purpose.
            if plot_image:
                self.model.eval()
                self._plot_image(epoch, plot_image)
                self.model.train()

            time_taken = time()-start_time
            # Training Logs printed.
            print(f'Epoch: {epoch+1:03d},  ', end='')
            print(f'Train Loss:{epoch_train_loss:.7f},  ', end='')
            print(f'Validation Loss:{epoch_val_loss:.7f},  ', end='')
            print(f'Time:{time_taken:.2f}secs', end='')

            # Save the best model with lowest validation loss feature.
            if save_best != None and last_loss > epoch_val_loss:
                self.save_model(save_best)
                last_loss = epoch_val_loss
                print(f'\tSaved at validation loss: {epoch_val_loss:.10f}')
            else:
                print()
        return history

    def save_model(self, path):
        """ Saves the currently used model to the path specified.
        Follows the best method recommended by Pytorch
        Link: https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended
        Parameters:
            path(str): The file location where the model needs to be saved.
        Returns:
            None
        """
        torch.save(self.model.state_dict(), path)

    def restore_model(self, path):
        """ Loads the saved model and restores it to the "model" object.
        Loads the model based on device used for computation.(CPU/GPU) 
        Follows the best method recommended by Pytorch
        Link: https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended
        Parameters:
            path(str): The file location where the model is saved.
        Returns:
            None
        """
        if self.device == 'cpu':
            self.model.load_state_dict(torch.load(path, map_location=device))
        else:
            self.model.load_state_dict(torch.load(path))
            self.model.to(self.device)

    def test(self, testloader, threshold=0.5):
        """ To test the performance of model on testing dataset.
        Parameters:
            testloader(torch.utils.data.Dataloader): Testing data
                        loader for the optimizer.
            threshold(float): Threshold value after which value will be part 
                              of output.
                              Default: 0.5

        Returns:
            mean_val_score(float): The mean Sørensen–Dice Coefficient for the 
                                    whole test dataset.
        """
        # Putting the model to evaluation mode
        self.model.eval()
        # Getting test data indices for dataloading
        test_data_indexes = testloader.sampler.indices[:]
        # Total testing data used.
        data_len = len(test_data_indexes)
        # Score after testing on dataset.
        mean_dice_score = 0
        mean_iou_score = 0
        # Error checking to set testloader batch size to 1.
        batch_size = testloader.batch_size
        if batch_size != 1:
            raise Exception("Set batch size to 1 for testing purpose")
        # Converting to iterator to get data in loops.
        testloader = iter(testloader)
        # Running the loop until no more data is left to test.
        while len(test_data_indexes) != 0:
            # Getting a data sample.
            data = next(testloader)
            # Getting the data index
            index = int(data['index'])
            # Removing the data index from total data indices
            # to indicate this data score has been included.
            if index in test_data_indexes:
                test_data_indexes.remove(index)
            else:
                continue
            # Data prepared to be given as input to model.
            image = data['image'].view((1, 1, 512, 512)).to(self.device)
            mask = data['mask']

            # Predicted output from the input sample.
            mask_pred = self.model(image).cpu()
            # Threshold elimination.
            mask_pred = (mask_pred > threshold)
            mask_pred = mask_pred.numpy()
            
            mask = np.resize(mask, (1, 512, 512))
            mask_pred = np.resize(mask_pred, (1, 512, 512))
            
            # Calculating the dice score for original and 
            # constructed image mask.
            mean_dice_score += self.dice_coefficient(mask_pred, mask)
            mean_iou_score += self._calculate_iou(mask_pred,mask)

        # Calculating the mean score for the whole test dataset.
        mean_dice_score = mean_dice_score / data_len
        mean_iou_score = mean_iou_score/ data_len
        # Putting the model back to training mode.
        self.model.train()
        return mean_dice_score, mean_iou_score

    def predict(self, data, threshold=0.5):
        """ Calculate the output mask on a single input data.
        Parameters:
            data(dict): Contains the index, image, mask torch.Tensor.
                        'index': Index of the image.
                        'image': Contains the tumor image torch.Tensor.
                        'mask' : Contains the mask image torch.Tensor.
            threshold(float): Threshold value after which value will be part of output.
                                Default: 0.5

        Returns:
            image(numpy.ndarray): 512x512 Original brain scanned image.
            mask(numpy.ndarray): 512x512 Original mask of scanned image.
            output(numpy.ndarray): 512x512 Generated mask of scanned image.
            score(float): Sørensen–Dice Coefficient for mask and output.
                            Calculates how similar are the two images.
        """
        self.model.eval()
        image = data['image'].numpy()
        mask = data['mask'].numpy()

        image_tensor = torch.Tensor(data['image'])
        image_tensor = image_tensor.view((-1, 1, 512, 512)).to(self.device)
        output = self.model(image_tensor).detach().cpu()
        output = (output > threshold)
        output = output.numpy()

        image = np.resize(image, (512, 512))
        mask = np.resize(mask, (512, 512))
        output = np.resize(output, (512, 512))
        dice_score = self.dice_coefficient(output, mask)
        iou_score = self._calculate_iou(output, mask)
        return image, mask, output, dice_score, iou_score

    def _train_epoch(self, trainloader, mini_batch):
        """ Training each epoch.
        Parameters:
            trainloader(torch.utils.data.Dataloader): Training data
                        loader for the optimizer.
            mini_batch(int): Used to print logs for epoch batches.

        Returns:
            epoch_loss(float): Loss calculated for each epoch.
        """
        epoch_loss, batch_loss, batch_iteration = 0, 0, 0
        for batch, data in enumerate(trainloader):
            # Keeping track how many iteration is happening.
            batch_iteration += 1
            # Loading data to device used.
            image = data['image'].to(self.device)
            mask = data['mask'].to(self.device)
            # Clearing gradients of optimizer.
            self.optimizer.zero_grad()
            # Calculation predicted output using forward pass.
            output = self.model(image)
            # Calculating the loss value.
            loss_value = self.criterion(output, mask)
            # Computing the gradients.
            loss_value.backward()
            # Optimizing the network parameters.
            self.optimizer.step()
            # Updating the running training loss
            epoch_loss += loss_value.item()
            batch_loss += loss_value.item()

            # Printing batch logs if any.
            if mini_batch:
                if (batch+1) % mini_batch == 0:
                    batch_loss = batch_loss / \
                        (mini_batch*trainloader.batch_size)
                    print(
                        f'    Batch: {batch+1:02d},\tBatch Loss: {batch_loss:.7f}')
                    batch_loss = 0

        epoch_loss = epoch_loss/(batch_iteration*trainloader.batch_size)
        return epoch_loss

    def _validate_epoch(self, valloader):
        """ Validate each epoch.
        Parameters:
            valloader(torch.utils.data.Dataloader): Validation data
                        loader for evaluating the model performance.

        Returns:
            epoch_loss(float): Loss calculated for each epoch.
        """
        epoch_loss, batch_loss, batch_iteration = 0, 0, 0
        with torch.no_grad():
            for batch, data in enumerate(valloader):
                # Keeping track how many iteration is happening.
                batch_iteration += 1
                # Loading data to device used.
                image = data['image'].to(self.device)
                mask = data['mask'].to(self.device)
                # Calculation predicted output using forward pass.
                output = self.model(image)
                # Calculating the loss value.
                loss_value = self.criterion(output, mask)
                # Updating the running validation loss
                epoch_loss += loss_value.item()
                batch_loss += loss_value.item()

        epoch_loss = epoch_loss/(batch_iteration*valloader.batch_size)
        return epoch_loss

    def _plot_image(self, epoch, sample):
        """
        Parameters:
            epoch(int): Running epoch number used to plot on Tensorboard
            sample(list): Sample inputs used to visualize the progress of
                          training over epochs.
        Returns:
            None
        """
        inputs = list()
        mask = list()

        # Inputs seperated.
        for data in sample:
            inputs.append(data['image'])
        # Inputs stacked together in a single batch
        inputs = torch.stack(inputs).to(self.device)