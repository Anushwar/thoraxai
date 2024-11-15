# NIH ChestXray14 Classification with ConvNext

This project implements a deep learning solution for multi-label classification of chest X-ray images using the NIH ChestXray14 dataset. The implementation uses the ConvNext architecture through the `timm` library with options for transfer learning.

## Dataset

The NIH Chest X-ray Dataset consists of 112,120 X-ray images from 30,805 unique patients with 14 disease labels:

- Atelectasis
- Cardiomegaly
- Effusion
- Infiltration
- Mass
- Nodule
- Pneumonia
- Pneumothorax
- Consolidation
- Edema
- Emphysema
- Fibrosis
- Pleural Thickening
- Hernia

## Requirements

Inside your virtual environment, run the following:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── config.py        # Configuration and argument parsing
├── dataset.py       # Dataset class implementation
├── loss.py          # Focal Loss implementation
├── metrics.py       # Evaluation metrics
├── trainer.py       # Training loop implementation
├── main.py         # Main execution script
└── README.md
```

## Usage

Basic usage:

```bash
python main.py
```

With pretrained weights:

```bash
python main.py --pretrained
```

### Command Line Arguments

| Argument              | Default | Description                    |
| --------------------- | ------- | ------------------------------ |
| `--pretrained`        | False   | Use pretrained weights         |
| `--batch_size`        | 64      | Batch size for training        |
| `--num_workers`       | 4       | Number of data loading workers |
| `--max_train_samples` | 80000   | Maximum training samples       |
| `--max_test_samples`  | 20000   | Maximum test samples           |
| `--num_epochs`        | 30      | Number of training epochs      |
| `--learning_rate`     | 0.0001  | Learning rate                  |
| `--weight_decay`      | 0.01    | Weight decay for optimization  |

Example with multiple arguments:

```bash
python main.py --pretrained --batch_size 32 --learning_rate 0.0005 --num_epochs 50
```

## Model Architecture

The project uses ConvNext Base by default, but you can easily modify the architecture by changing the model in `main.py`:

```python
model = timm.create_model('convnext_base', pretrained=args.pretrained, num_classes=14)
```

### Alternative Models

You can replace 'convnext_base' with other architectures available in timm. Some options include:

- `convnext_tiny`
- `convnext_small`
- `convnext_large`
- `swin_tiny_patch4_window7_224`
- `vit_base_patch16_224`
- `efficientnet_b0`

To see all available models:

```python
import timm
print(timm.list_models('*'))
```

For detailed model specifications, refer to the [timm documentation](https://huggingface.co/docs/timm/index).

## Training Details

- **Loss Function**: Focal Loss with α=0.25 and γ=2
- **Optimizer**: AdamW
- **Learning Rate Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Implemented with patience=15
- **Metrics**: Mean AUC-ROC across all classes
- **Image Size**: 384x384 pixels
- **Data Augmentation**: Basic normalization (can be extended in transforms)

## Model Saving

The best model based on validation AUC is automatically saved as 'best_model.pth'.

## Performance Monitoring

During training, the script outputs:

- Epoch progress with loss
- Current AUC score
- Best AUC score achieved
- Learning rate adjustments
- Early stopping notifications

## Customization

### Adding Data Augmentation

Modify the transform in `main.py`:

```python
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomHorizontalFlip(),  # add augmentations here
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
```

### Modifying Training Parameters

You can adjust training parameters either through command line arguments or by modifying the defaults in `config.py`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NIH for providing the ChestXray14 dataset
- timm library for model implementations
- PyTorch team for the deep learning framework

## References

1. NIH ChestXray14 Dataset: [Wang et al. 2017](https://arxiv.org/abs/1705.02315)
2. ConvNext Paper: [Liu et al. 2022](https://arxiv.org/abs/2201.03545)
3. timm Library: [Ross Wightman](https://github.com/huggingface/pytorch-image-models)

This README provides comprehensive information about:

- Project overview and dataset description
- Installation and usage instructions
- Detailed explanation of command-line arguments
- Model architecture options and customization
- Training details and monitoring
- Ways to extend and modify the implementation

It serves as both documentation and a quick-start guide for users wanting to work with the project. The references section provides links to relevant papers and resources for further reading.
