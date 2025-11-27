from super_gradients.training import Trainer
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training import models
from super_gradients.common.object_names import Models
import os

def train_yolo_nas():
    # Define parameters
    dataset_params = {
        'data_dir': 'data/yolo_dataset',
        'train_images_dir': 'train/images',
        'train_labels_dir': 'train/labels',
        'val_images_dir': 'val/images',
        'val_labels_dir': 'val/labels',
        'test_images_dir': 'test/images',
        'test_labels_dir': 'test/labels',
        'classes': ['fire']
    }

    # Initialize Trainer
    trainer = Trainer(experiment_name='yolo_nas_fire_det', ckpt_root_dir='checkpoints')
    
    # Dataloaders
    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['train_images_dir'],
            'labels_dir': dataset_params['train_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={'batch_size': 16, 'num_workers': 2}
    )

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['val_images_dir'],
            'labels_dir': dataset_params['val_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={'batch_size': 16, 'num_workers': 2}
    )

    # Load Model
    model = models.get(Models.YOLO_NAS_S, num_classes=1, pretrained_weights="coco")

    # Training Params
    train_params = {
        "max_epochs": 30,
        "lr_mode": "cosine",
        "initial_lr": 1e-4,
        "optimizer": "AdamW",
        "loss": "PPYoloELoss",
        "valid_metrics_list": [
            "DetectionMetrics_050"
        ],
        "metric_to_watch": "mAP@0.50"
    }

    print("Starting YOLO-NAS Training...")
    trainer.train(model=model, training_params=train_params, train_loader=train_data, valid_loader=val_data)
    print("YOLO-NAS Training Complete.")

if __name__ == "__main__":
    train_yolo_nas()
