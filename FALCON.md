# "ISS Guardian" - Use Case & Model Maintenance Plan

This document outlines the real-world application of our object detection model and the strategy for maintaining and updating it using the **Duality Falcon** platform.

---

## Application Use Case: "ISS Guardian"

### 1. The Real-World Problem

On the International Space Station (ISS), the safety of the crew and the integrity of the station are paramount. Manually tracking the location and status of critical equipment—such as fire extinguishers, oxygen tanks, and essential toolboxes—is a time-consuming task prone to human error.

- A misplaced tool can become a dangerous projectile in zero gravity.
- An untracked, depleted fire extinguisher is a major safety hazard.

### 2. Our Solution: The "ISS Guardian" Application

**ISS Guardian** is a web-based tool designed for rapid, automated safety and inventory checks. An astronaut can take a photo of a module or storage area with a tablet and upload it to the app.

Our optimized **YOLOv8** object detection model instantly analyzes the image, identifies critical equipment, and provides visual confirmation with bounding boxes.

#### Key Benefits:
- **Instant Safety Audits:** Quickly verify that all fire extinguishers are in their designated locations.
- **Efficient Inventory Management:** Get an immediate count of oxygen tanks or toolboxes before mission-critical tasks.
- **Reduced Crew Workload:** Automate tedious tasks, allowing astronauts to focus on research and operations.

---

## Maintaining the Model with Falcon

A model is only as good as its data. As the ISS evolves, our model must evolve too. The **Falcon digital twin** platform provides a cost-effective and efficient way to simulate new conditions and update the model accordingly—eliminating the need for real-world space data collection.

---

### 🔧 Scenario 1: An Object’s Appearance Changes

**Problem:** A new model of fire extinguisher, with a different shape and color, is introduced. The current model may fail to recognize it.

**Falcon Solution:**

1. **Update Digital Twin:** Modify the 3D asset for the "FireExtinguisher" class in Falcon to reflect the new design.
2. **Generate Synthetic Data:** Use Falcon to produce thousands of images with the updated extinguisher under varied lighting, angles, and occlusions.
3. **Fine-Tune Model:** Retrain YOLOv8 with this new dataset so it recognizes both old and new extinguisher types.

---

### 🚫 Scenario 2: A New Object Causes Confusion

**Problem:** A similarly shaped medical supply kit is introduced, leading to false positives—it's misidentified as a "ToolBox."

**Falcon Solution:**

1. **Update Digital Twin:** Add the 3D model of the new medical kit to the Falcon simulation.
2. **Generate "Hard Negative" Data:** Create a synthetic dataset where the medical kit appears near toolboxes but is explicitly not labeled as one.
3. **Fine-Tune Model:** Train the model on this targeted dataset to improve classification accuracy and reduce false positives.

---

## Conclusion

By integrating **Falcon** into our long-term model maintenance strategy, the **"ISS Guardian"** application remains a reliable, adaptive, and critical safety tool for the ISS crew. It ensures continued accuracy, flexibility, and operational excellence as the station’s environment evolves.

---
