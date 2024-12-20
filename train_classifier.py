import torch
from tqdm import tqdm
import wandb


def poshumim(images):
    rnd_normal = torch.randn(images.shape[0], device=images.device)
    sigma = torch.exp(rnd_normal)[:, None, None, None].to(images.device) # [batch, 1, 1, 1]
    n = torch.randn_like(images) * sigma
    noisy_img = images + n
    return noisy_img, sigma.flatten()


def train_loop(dataloader_train, dataloader_val, model, n_epochs):
    wandb.login(key='bbe60953ed99662c4459f461386ecd58a2f2ee3a')
    
    run = wandb.init(
        project="EGSDE_class_train"
    )
    
    device = 'cuda'
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    ce_loss = torch.nn.CrossEntropyLoss()
    
    for epoch in tqdm(range(n_epochs)):
        model.train()
        for batch, labels in dataloader_train:
            optimizer.zero_grad()
            noisy_img, timestamps = poshumim(batch.to(device))
            _, y_pred = model(noisy_img, timestamps)
            loss = ce_loss(y_pred, labels.to(device))

            loss.backward()
            wandb.log({"train_loss": loss.item()})
            optimizer.step()
            
        model.eval()
        for batch, labels in dataloader_val:
            with torch.no_grad():
                noisy_img, timestamps = poshumim(batch.to(device))
                _, y_pred = model(noisy_img, timestamps)
                y_pred = torch.argmax(y_pred, dim=-1)
                acc = (y_pred == labels.to(device)).float().mean()
                wandb.log({"val_acc": acc})

        name = 'checkpoints/checkpoint_epoch_{}'.format(epoch)
        torch.save(model.state_dict(), name)
    return model
