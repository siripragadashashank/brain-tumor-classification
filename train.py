import torch
import pandas as pd
from utils import mri_types
from datasets import MRIDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model import Model, Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_mri_type(df_train, df_valid, mri_type):
    if mri_type == "all":
        train_list = []
        valid_list = []
        for mri_type in mri_types:
            df_train.loc[:, "MRI_Type"] = mri_type
            train_list.append(df_train.copy())
            df_valid.loc[:, "MRI_Type"] = mri_type
            valid_list.append(df_valid.copy())

        df_train = pd.concat(train_list)
        df_valid = pd.concat(valid_list)
    else:
        df_train.loc[:, "MRI_Type"] = mri_type
        df_valid.loc[:, "MRI_Type"] = mri_type

    print(df_train.shape, df_valid.shape)
    print(df_train.head())

    train_data_retriever = MRIDataset(
        df_train["BraTS21ID"].values,
        df_train["MGMT_value"].values,
        df_train["MRI_Type"].values,
        augment=True
    )

    valid_data_retriever = MRIDataset(
        df_valid["BraTS21ID"].values,
        df_valid["MGMT_value"].values,
        df_valid["MRI_Type"].values
    )

    train_loader = DataLoader(
        train_data_retriever,
        batch_size=4,
        shuffle=True,
        num_workers=8, pin_memory=True
    )

    valid_loader = DataLoader(
        valid_data_retriever,
        batch_size=4,
        shuffle=False,
        num_workers=8, pin_memory=True
    )

    model = Model()
    model.to(device)

    # checkpoint = torch.load("best-model-all-auc0.555.pth")
    # model.load_state_dict(checkpoint["model_state_dict"])

    # print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    criterion = F.binary_cross_entropy_with_logits

    trainer = Trainer(
        model,
        device,
        optimizer,
        criterion
    )

    history = trainer.fit(
        10,
        train_loader,
        valid_loader,
        f"{mri_type}",
        10,
    )

    return trainer.lastmodel


def predict(modelfile, df, mri_type, split):
    print("Predict:", modelfile, mri_type, df.shape)
    df.loc[:, "MRI_Type"] = mri_type

    data_retriever = MRIDataset(
        df.index.values,
        mri_type=df["MRI_Type"].values,
        split=split
    )

    data_loader = DataLoader(
        data_retriever,
        batch_size=4,
        shuffle=False,
        num_workers=8,
    )

    model = Model()
    model.to(device)

    checkpoint = torch.load(modelfile)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_pred = []
    ids = []

    for e, batch in enumerate(data_loader, 1):
        print(f"{e}/{len(data_loader)}", end="\r")
        with torch.no_grad():
            tmp_pred = torch.sigmoid(model(batch["X"].to(device))).cpu().numpy().squeeze()
            if tmp_pred.size == 1:
                y_pred.append(tmp_pred)
            else:
                y_pred.extend(tmp_pred.tolist())
            ids.extend(batch["id"].numpy().tolist())

    pred_df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": y_pred})
    pred_df = pred_df.set_index("BraTS21ID")
    return pred_df

