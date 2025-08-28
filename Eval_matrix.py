import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
class eval_matrix:
    def __init__(self,model):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model=model.to(self.device)
        total_parameter=np.sum([i.cpu().numel() for i in self.model.parameters()])
        trainable_parameter=np.sum([i.cpu().numel() if i.requires_grad else 0 for i in self.model.parameters()])
        print(f'total_parameter={total_parameter} | trainable_parameter={trainable_parameter}')
    
    def train_val_loss_acc(self,epoch,train_loss,train_acc,val_loss,val_acc,epoch_time,remaining_time):
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {epoch_time:.2f} sec | Estimated remaining time: {remaining_time/60:.2f} min")

    def confusion_matrix(self,y_pred,y_true,class_names,names=""):
        
        cm = confusion_matrix(y_true, y_pred)       
        # Normalize values (percentage instead of raw counts)
        #cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, None]       fmt=".2f" 
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d",cmap="Blues", 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=True, square=True, linewidths=.5)
        
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title(f"Confusion Matrix{' for ' + names if names else ''}")
        plt.show()
    def draw_loss_acc(self, loss_acc):
        loss_acc = np.array(loss_acc)
        epochs = np.arange(len(loss_acc))
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)  
        
        axs[0,0].plot(epochs, loss_acc[:,0], label="Train Loss", color='blue')
        axs[0,0].set_title("Train Loss")
        axs[0,0].set_xlabel("Epoch")
        axs[0,0].set_ylabel("Loss")
        axs[0,0].legend()
    
        axs[0,1].plot(epochs, loss_acc[:,1], label="Train Accuracy", color='green')
        axs[0,1].set_title("Train Accuracy")
        axs[0,1].set_xlabel("Epoch")
        axs[0,1].set_ylabel("Accuracy")
        axs[0,1].legend()
    
        axs[1,0].plot(epochs, loss_acc[:,2], label="Validation Loss", color='orange')
        axs[1,0].set_title("Validation Loss")
        axs[1,0].set_xlabel("Epoch")
        axs[1,0].set_ylabel("Loss")
        axs[1,0].legend()
    
        axs[1,1].plot(epochs, loss_acc[:,3], label="Validation Accuracy", color='red')
        axs[1,1].set_title("Validation Accuracy")
        axs[1,1].set_xlabel("Epoch")
        axs[1,1].set_ylabel("Accuracy")
        axs[1,1].legend()
    
        plt.show()
   
                 
                
        


        
        
