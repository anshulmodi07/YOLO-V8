from ultralytics import YOLO
import os

if __name__ == '__main__': 

    this_dir = os.path.dirname(__file__)
    if this_dir:
        os.chdir(this_dir)


    model = YOLO("yolov8x.pt")

    print("\n--- Final All-In Optimized Training Run ---")
    print("Using model: yolov8x.pt")
    print("Targeting class confusion and hallucination reduction.")
    print("Training for 100 epochs with early stopping.")
    print("----------------------------------------------------\n")

   
    results = model.train(
       
        data="yolo_params.yaml",

    
        epochs=100,
        patience=30,  

     
        batch=8,

       
        optimizer='AdamW',
        lr0=0.0008,
        lrf=0.00008,
        momentum=0.937,
        weight_decay=0.0005,

     
        box=10.0,     
        cls=1.0,      
        dfl=2.0,      
        kobj=3.0,    


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
        copy_paste=0.15,

      
        label_smoothing=0.1,
        
      
        imgsz=640,
        plots=True,
        val=True
    )

    print("\n--- Final Training Finished ---")
    
   
    print("\n--- Running Final Validation with TTA ---")
    final_metrics = model.val(augment=True)
    print(f"Final mAP@0.5 with TTA: {final_metrics.box.map50:.4f}")
