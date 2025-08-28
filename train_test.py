from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
class Train_eval:
    def __init__(self,model,train_dataloader,val_dataloader,criterion=None,optimizer=None,lr=0.001,epochs=100,swa_start=50,swa_lr=0.05):
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'currently available devie ={self.device}')
        self.model=model.to(self.device)
        self.train_dataloader=train_dataloader
        self.val_dataloader=val_dataloader
        self.criterion=nn.CrossEntropyLoss(label_smoothing=0.1) if criterion is None else criterion
        self.lr=lr
        self.swa_lr=swa_lr
        self.epochs=epochs
        self.optimizer=torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-4) if optimizer is None else optimizer
        # self.schedular= optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5, verbose=True)
        self.swa_start = swa_start
        self.swa_model = AveragedModel(self.model)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.swa_lr)

    def train(self,get_label=None):
        self.model.train()
        train_loss=0
        total_sample=0
        total_correct=np.zeros((8),dtype=int)
        pred_label = [np.array([], dtype=int) for _ in range(8)]
        true_label = [np.array([], dtype=int) for _ in range(8)]
        
        for der_data,cli_data,meta_data,labels in self.train_dataloader:
            self.optimizer.zero_grad()
            der_data=der_data.to(self.device)
            cli_data=cli_data.to(self.device)
            meta_data=meta_data.to(self.device)
            labels=torch.stack([labels[i].to(self.device) for i in np.arange(8)])
            output=self.model(meta_data,cli_data,der_data)
            #print(f'output.shape={output.shape} and labels.shape={labels.shape}')
            loss=torch.true_divide(torch.sum(torch.stack([self.criterion(output[i],labels[i].squeeze(1)) for i in np.arange(8)])),8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            batch_size=labels[0].shape[0]
            train_loss=train_loss+loss.item()*batch_size
            if get_label:
               # print(pred_label,torch.argmax(output,dim=1).detach().numpy(),labels.squeeze(1).detach().numpy())
                for i in range(8):
                    pred_label[i]=np.concatenate((pred_label[i],torch.argmax(output[i],dim=1).detach().cpu().numpy()))
                    true_label[i]=np.concatenate((true_label[i],labels[i].squeeze(1).detach().cpu().numpy()))
                
            for i in np.arange(8):
                total_correct[i]=total_correct[i]+(torch.argmax(output[i],dim=1)==labels[i].squeeze(1)).sum().item()
            
            total_sample+=batch_size
        if get_label:
            return pred_label,true_label  
            
        return train_loss/total_sample,(np.sum(total_correct)/8)/total_sample
    def evalute(self,get_label=None):
        self.model.eval()
        val_loss=0
        total_sample=0
        total_correct=np.zeros((8),dtype=int)
        pred_label = [np.array([], dtype=int) for _ in range(8)]
        true_label = [np.array([], dtype=int) for _ in range(8)]
        with torch.no_grad():
            for der_data,cli_data,meta_data,labels in self.val_dataloader:
                der_data=der_data.to(self.device)
                cli_data=cli_data.to(self.device)
                meta_data=meta_data.to(self.device)
                labels=torch.stack([labels[i].to(self.device) for i in np.arange(8)])
                output=self.model(meta_data,cli_data,der_data)
                loss=torch.true_divide(torch.sum(torch.stack([self.criterion(output[i],labels[i].squeeze(1)) for i in np.arange(8)])),8)
                batch_size=labels[0].size(0)
                val_loss=val_loss+loss.item()*batch_size
                total_sample+=batch_size
                #print(output.shape,labels.shape)
                for i in range(8):
                    pred_label[i]=np.concatenate((pred_label[i],torch.argmax(output[i],dim=1).detach().cpu().numpy()))
                    true_label[i]=np.concatenate((true_label[i],labels[i].squeeze(1).detach().cpu().numpy()))
                for i in np.arange(8):
                    total_correct[i]=total_correct[i]+(torch.argmax(output[i],dim=1)==labels[i].squeeze(1)).sum().item()
        if get_label:
            return pred_label,true_label
        return val_loss/total_sample,(np.sum(total_correct)/8)/total_sample
    def lr_schedular(self,val_loss):
        self.scheduler.step(val_loss)     
    def get_model(self):
        return self.model
    def set_model_state(self,best_model_state):
        self.model.load_state_dict(best_model_state)
    def get_train_dataloader(self):
        return self.train_dataloader
    def get_swa_model(self):
        return self.swa_model
        
        
