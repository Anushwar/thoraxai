import torch
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="ChestXray Training Script")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained weights (default: False)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=80000,
        help="Maximum number of training samples (default: 80000)",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=20000,
        help="Maximum number of test samples (default: 20000)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=30,
        help="Number of epochs to train (default: 30)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay (default: 0.01)"
    )

    return parser.parse_args()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Effusion": 2,
    "Infiltration": 3,
    "Mass": 4,
    "Nodule": 5,
    "Pneumonia": 6,
    "Pneumothorax": 7,
    "Consolidation": 8,
    "Edema": 9,
    "Emphysema": 10,
    "Fibrosis": 11,
    "Pleural_Thickening": 12,
    "Hernia": 13,
}
