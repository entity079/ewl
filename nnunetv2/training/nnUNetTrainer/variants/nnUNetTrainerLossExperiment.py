import torch # Import torch to access gradient and tensor functions
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer # Import the base trainer class

class nnUNetTrainerLossExperiment(nnUNetTrainer): # Define a new class inheriting from the standard trainer
    def train_step(self, batch: dict) -> dict: # Override the core training step
        data = batch['data'] # Extract image data from the current batch
        target = batch['target'] # Extract ground truth labels from the current batch

        data = data.to(self.device, non_blocking=True) # Move images to GPU/CPU
        if isinstance(target, list): # Check if targets are a list for deep supervision
            target = [i.to(self.device, non_blocking=True) for i in target] # Move all target levels to device
        else:
            target = target.to(self.device, non_blocking=True) # Move single target to device

        self.optimizer.zero_grad() # Clear gradients from the previous step
        
        # --- Loss Calculation ---
        output = self.network(data) # Pass data through the network to get predictions
        l = self.loss(output, target) # Calculate loss using the default Dice + CE combination
        
        # EXPERIMENT: Print the raw loss value to the console
        print(f"Current Loss: {l.item()}") 

        # --- Backpropagation ---
        l.backward() # Propagate the error backward through the network layers
        
        # --- Gradient Inspection ---
        for name, param in self.network.named_parameters(): # Iterate through all network layers
            if param.grad is not None: # Check if a layer has calculated gradients
                # EXPERIMENT: View the magnitude of what propagates backward
                print(f"Layer: {name} | Grad Norm: {param.grad.norm().item()}") 
                break # Only print the first layer for brevity

        self.optimizer.step() # Update model weights based on the gradients
        
        # --- Visual Output / Prediction ---
        # Returns loss for the logger which generates progress.png
        return {'loss': l.detach().cpu().numpy()}
