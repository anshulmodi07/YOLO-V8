# Optimized training script targeting confusion matrix issues
# Focus: Reduce false negatives & improve background separation

from ultralytics import YOLO
import os

if __name__ == '__main__': 
    # --- Part 1: Training Execution ---
    this_dir = os.path.dirname(__file__)
    if this_dir:
        os.chdir(this_dir)

    # Upgrade to largest model for maximum performance
    model = YOLO("yolov8x.pt")  # CHANGED: Upgraded from yolov8m

    print("\n--- Confusion Matrix Optimized Training ---")
    print("Using model: yolov8x.pt")
    print("Targeting false negative reduction and background separation")
    print("Training for 60 epochs with extended patience")
    print("----------------------------------------------------\n")

    results = model.train(
        # Model and Data
        data="yolo_params.yaml",

        # Extended training for better convergence
        epochs=60,          # CHANGED: Increased from 50
        patience=25,        # CHANGED: Increased from 20
        batch=8,            # CHANGED: Reduced from 16 for better gradients

        # Optimizer (fine-tuned for object detection)
        optimizer='AdamW',
        lr0=0.0008,         # CHANGED: Slightly reduced for stability
        lrf=0.00008,        # CHANGED: Lower final LR
        momentum=0.937,
        weight_decay=0.0005,

        # CRITICAL: Loss weights targeting confusion matrix issues
        box=12.0,           # CHANGED: Increased from 7.5 - better localization
        cls=1.2,            # CHANGED: Increased from 0.5 - stronger classification
        dfl=2.5,            # CHANGED: Increased from 1.5 - better distribution
        kobj=2.5,           # CHANGED: Reduced from 5.0 - less aggressive FP penalty
        
        # Enhanced augmentations for better object-background separation
        hsv_h=0.02,         # CHANGED: Increased color variation
        hsv_s=0.8,          # CHANGED: Increased saturation variation
        hsv_v=0.5,          # CHANGED: Increased brightness variation
        degrees=20.0,       # CHANGED: Increased rotation
        translate=0.15,     # CHANGED: Increased translation
        scale=0.6,          # CHANGED: Increased scale variation
        shear=2.0,          # CHANGED: Added shear augmentation
        perspective=0.0001, # CHANGED: Added slight perspective
        flipud=0.1,         # CHANGED: Added vertical flips
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,         # CHANGED: Increased mixup
        copy_paste=0.15,    # CHANGED: Increased copy_paste
        
        # Enhanced regularization
        label_smoothing=0.15, # CHANGED: Increased from 0.1
        
        # Additional parameters for better performance
        imgsz=640,
        save_period=10,
        plots=True,
        val=True,
        
        # Early stopping based on mAP50
        monitor='metrics/mAP50(B)'
    )

    print("\n--- Confusion Matrix Optimized Training Finished ---")
    
    # Post-training validation with TTA
    print("Running validation with Test-Time Augmentation...")
    val_results = model.val(data="yolo_params.yaml", augment=True)
    print(f"Final mAP@0.5 with TTA: {val_results.box.map50}")