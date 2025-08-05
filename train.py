# This is the final, most optimized training script.
# It combines every lesson learned to target our specific problems
# and push the mAP score to its maximum potential.

from ultralytics import YOLO
import os

if __name__ == '__main__': 
    # --- Part 1: Training Execution ---
    # Ensure we are in the correct directory
    this_dir = os.path.dirname(__file__)
    if this_dir:
        os.chdir(this_dir)

    # Use the largest, most powerful model for maximum performance.
    model = YOLO("yolov8x.pt")

    print("\n--- Final All-In Optimized Training Run ---")
    print("Using model: yolov8x.pt")
    print("Targeting class confusion and hallucination reduction.")
    print("Training for 100 epochs with early stopping.")
    print("----------------------------------------------------\n")

    # This configuration is the result of our iterative analysis.
    results = model.train(
        # Model and Data
        data="yolo_params.yaml",

        # Training Duration: Long enough for the large model to converge.
        epochs=100,
        patience=30,  # Stop if no improvement for 30 epochs.

        # Batch Size: Smaller for more precise gradient updates.
        batch=8,

        # Optimizer: Stable settings for a long run.
        optimizer='AdamW',
        lr0=0.0008,
        lrf=0.00008,
        momentum=0.937,
        weight_decay=0.0005,

        # --- CRITICAL: Loss Weights Tuned to Our Specific Problems ---
        # We increase the penalty for bad boxes and wrong classes,
        # but slightly reduce the penalty for hallucinations to find a better balance.
        box=10.0,     # Heavily penalize inaccurate bounding boxes.
        cls=1.0,      # Increase penalty for misclassifications.
        dfl=2.0,      # Increase penalty for distribution errors.
        kobj=3.0,     # A balanced penalty for hallucinations.

        # --- CRITICAL: Full Suite of Advanced Augmentations ---
        # This is an aggressive strategy to build a highly robust model.
        hsv_h=0.020,
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=20.0,
        translate=0.15,
        scale=0.6,
        shear=2.0,
        perspective=0.0001,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.15, # Use a slightly higher copy_paste to fight confusion.

        # Regularization: Prevent the large model from overfitting.
        label_smoothing=0.1,
        
        # Other settings
        imgsz=640,
        plots=True,
        val=True
    )

    print("\n--- Final Training Finished ---")
    
    # Automatically run a final validation with Test-Time Augmentation
    print("\n--- Running Final Validation with TTA ---")
    final_metrics = model.val(augment=True)
    print(f"Final mAP@0.5 with TTA: {final_metrics.box.map50:.4f}")
