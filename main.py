# main.py
import os
import timm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import NIH_Chest_Xray_Dataset
from loss import FocalLoss
from trainer import ModelTrainer
from config import DEVICE, get_args


def main():
    # Get command line arguments
    args = get_args()

    # Set up data transforms
    transform = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Data paths
    data_dir = "/data/courses/2024/class_ImageSummerFall2024_jliang12/chestxray14/"
    train_file = os.path.join(data_dir, "train_val_list.txt")
    test_file = os.path.join(data_dir, "test_list.txt")

    # Create datasets
    train_dataset = NIH_Chest_Xray_Dataset(
        data_dir=data_dir,
        train_test_file=train_file,
        transform=transform,
        max_samples=args.max_train_samples,
        include_no_finding=False,
    )

    test_dataset = NIH_Chest_Xray_Dataset(
        data_dir=data_dir,
        train_test_file=test_file,
        transform=transform,
        max_samples=args.max_test_samples,
        include_no_finding=True,
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(
        f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}"
    )
    print(f"Using pretrained weights: {args.pretrained}")

    # Create model
    model = timm.create_model(
        "convnext_base", pretrained=args.pretrained, num_classes=14
    )
    model = model.to(DEVICE)

    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        criterion=FocalLoss(),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Train and evaluate
    print("Training model...")
    best_metrics = trainer.train_and_evaluate(
        train_dataloader, test_dataloader, num_epochs=args.num_epochs
    )

    print("\nBest Metrics:", best_metrics)

    # Final evaluation
    model.load_state_dict(torch.load("best_model.pth"))
    final_metrics = trainer._evaluate(test_dataloader)
    print("\nFinal Metrics:", final_metrics)


if __name__ == "__main__":
    main()
