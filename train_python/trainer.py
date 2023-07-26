import torch
import torch.nn.functional as F
import torch_optimizer as optim
from torch.optim import lr_scheduler
from torch.cuda.amp import autocast as autocast

from PTdata.Dataset import ImageDataset
from options.config import Config
from models.model_select import SelectTransforModel
from gaskLibs.utils.pytorchtools import EarlyStopping


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config("./options/config.yml")

scaler = torch.cuda.amp.GradScaler()


def trainer():

    loss = 0

    train_dataset = ImageDataset(csv_path=config.train_csv,
                                 img_size=config.image_size,
                                 is_aug=True)

    val_dataset = ImageDataset(csv_path=config.test_csv,
                               img_size=config.image_size,
                               is_aug=False)

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    select_transfor_model = SelectTransforModel(config)
    model = select_transfor_model("deeplab").to(DEVICE)
    # model = torch.compile(model)
    optimizer = optim.RangerQH(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

    early_stopping = EarlyStopping(patience=7,
                                   verbose=False)

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        for _, (data, target) in enumerate(data_loader_train):

            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            with autocast(dtype=torch.float16):
                output = model(data)

                loss = loss_fn(output, target.long())
                epoch_loss += loss.detach().cpu().numpy()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        train_loss = (epoch_loss / len(data_loader_train.dataset))

        # Validation of the model.
        model.eval()
        eval_loss = 0

        with torch.no_grad():
            for _, (data, target) in enumerate(data_loader_val):

                data, target = data.to(DEVICE), target.to(DEVICE)
                with autocast():
                    output = model(data)
                loss = loss_fn(output, target.long())
                eval_loss += loss.detach().cpu().numpy()

        valid_loss = (eval_loss / len(data_loader_val.dataset))

        print(" %d/%d loss = %.6f, valid_loss =  %.6f " %
              (epoch+1, config.epochs, train_loss, valid_loss))

        scheduler.step(loss)
        # pruner.finalize(inplace=True)
        early_stopping(loss, model)
        if early_stopping.early_stop:
            print("[INFO] Early stopping")
            break

    return model


def traced_func(model, saved_path, X):
    traced_model = torch.jit.trace(model, X)
    torch.jit.save(traced_model, saved_path)
    return traced_model


if __name__ == "__main__":

    best_model = trainer()

    torch.save(best_model, config.saved_path)

    best_traced_model = traced_func(
        model=best_model,
        saved_path=config.traced_path,
        X=torch.rand(1, 3, config.image_size[0],
                     config.image_size[1]).to(DEVICE)
    )
