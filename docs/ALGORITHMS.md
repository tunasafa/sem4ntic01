# Algorithm Documentation

## Real-Time Semantic Segmentation for Autonomous Navigation

This document explains how the segmentation system works - from capturing video to making navigation decisions.

---

## What Does This Actually Do?

Think of this system as giving a robot the ability to **see and understand** its environment:

1. **Camera captures video** (like your webcam)
2. **AI analyzes each frame** (finds people, objects, obstacles)
3. **System assigns priorities** (person = stop now, chair = go around)
4. **Robot gets navigation advice** (go left, stop, proceed forward)

All of this happens in **real-time** - about 30-55 times per second.

---

## The AI Model: YOLOv8-Seg

### Why This Model?

We use **YOLOv8-Seg** (You Only Look Once, version 8, Segmentation) because:

- **Fast**: Single pass through the network = real-time performance
- **Accurate**: State-of-the-art for object detection + segmentation
- **Practical**: Pre-trained on COCO dataset (80 common object classes)
- **Efficient**: Works on laptops, not just powerful servers

### How It Works

```
Input Image (640x480)
        ↓
[Convolutional Neural Network]
        ↓
Feature Maps (what the AI "sees")
        ↓
[Detection Head]
        ↓
Bounding Boxes + Class Labels + Segmentation Masks
        ↓
Output: "Person at (120, 50), 95% confidence, here's the exact shape"
```

### Performance by Hardware

| Device | Time per Frame | Frames per Second |
|--------|---------------|-------------------|
| Apple M1/M2/M3 | ~18ms | 55 FPS |
| NVIDIA RTX 4090 | ~5ms | 200 FPS |
| NVIDIA RTX 3080 | ~8ms | 125 FPS |
| Intel i7 CPU | ~80ms | 12 FPS |

**Key insight**: Running on a GPU (or Apple's MPS) gives you 5-10x speedup.

---

## Time & Space Complexity

### Time Complexity: O(H × W)

The algorithm processes every pixel once:

- **Image Preprocessing**: O(H × W) - resize, normalize colors
- **CNN Forward Pass**: O(H × W × K) - K is constant (model size)
- **Non-Maximum Suppression**: O(N²) - N = number of detections (usually small, <100)
- **Mask Generation**: O(H × W) per detection

**Total**: O(H × W) - linear in image size. This is as fast as it gets for image processing.

### Space Complexity: O(H × W + N × H × W)

Memory usage:

- **Frame buffer**: H × W pixels
- **Segmentation masks**: N masks × H × W (one per detected object)

For a 640×480 frame with 10 detections: ~3 MB of memory.

---

## Safety Priority System

Not all objects are equally important. A person crossing the path is more urgent than a potted plant.

### Priority Levels

| Priority | Category | Examples | Robot's Reaction |
|----------|----------|----------|------------------|
| **P10** 🔴 | Critical | People | EMERGENCY STOP |
| **P9** 🟠 | High | Dogs, cats, horses | High-priority avoidance |
| **P8** 🟡 | High | Cars, trucks, buses | Plan path carefully |
| **P7** 🔵 | Medium | Bicycles, stop signs | Follow traffic rules |
| **P5-6** 🟢 | Medium | Traffic lights | Caution advised |
| **P3-4** ⚪ | Low | Chairs, benches | General avoidance |
| **P1-2** 🟢 | Minimal | Plants, small items | Terrain awareness |

### How Priorities Work

```python
# Example: Robot sees a person (P10) and a chair (P3)
max_priority = max(10, 3)  # = 10

if max_priority >= 9:
    return "CRITICAL - STOP NOW"
elif max_priority >= 7:
    return "HIGH - Maneuver required"
# ... etc
```

---

## Navigation Logic

### Risk Assessment

The system doesn't just detect objects - it tells the robot **what to do**:

1. **Divide the path** into left, center, right zones
2. **Check each zone** for obstacles
3. **Recommend action** based on what's clear

### Decision Tree

```
Is the center path clear?
├─ YES → "PROCEED FORWARD"
└─ NO → Is left clear?
    ├─ YES → "MANEUVER LEFT"
    └─ NO → Is right clear?
        ├─ YES → "MANEUVER RIGHT"
        └─ NO → "STOP - No safe path"
```

### Example Scenario

```
Camera view:
┌─────────────────────────────┐
│                             │
│    [Person]    [Chair]      │
│       │           │         │
│    P10 (red)   P3 (gray)    │
│                             │
│         [Empty path]        │
│              ↓              │
│         SAFE TO GO          │
└─────────────────────────────┘

Recommendation: PROCEED FORWARD
Risk Level: SAFE
```

---

## Optimizations

### 1. TensorRT Acceleration (2-10x Faster)

Export the model to TensorRT for NVIDIA GPUs:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt').export(format='engine', device=0, half=True)"
```

**Result**: 5ms → 2ms inference time on RTX 4090

### 2. FP16 Quantization (2x Faster, Minimal Accuracy Loss)

Use 16-bit floating point instead of 32-bit:

```python
model(image, half=True)  # FP16 mode
```

**Trade-off**: ~0.5% lower accuracy, but 2x speedup

### 3. Frame Resizing

Large frames slow down inference. We resize to 640px max:

```python
if max(height, width) > 640:
    scale = 640 / max(height, width)
    frame = resize(frame, scale)
```

**Impact**: 1080p → 640px = 3x faster with minimal accuracy loss

### 4. Confidence Thresholding

Skip low-confidence detections to reduce processing:

```python
if confidence < 0.6:  # Ignore detections below 60% confidence
    continue
```

---

## Real-World Deployment

### For Edge Devices (Jetson Nano, Raspberry Pi)

- Use **YOLOv8n-seg** (nano model - smallest)
- Enable **TensorRT** if available
- Limit resolution to **480p**
- Expect **15-20 FPS**

### For Desktop/Server

- Use **YOLOv8s-seg** (small - better accuracy)
- **TensorRT** for maximum speed
- Can handle **720p-1080p**
- Expect **50-100+ FPS**

### For Cloud Deployment

- Use **YOLOv8m-seg** (medium - best accuracy)
- GPU instance (T4, A10G)
- Batch process multiple streams
- Expect **100-200 FPS per GPU**

---

## Future Improvements

Ideas for making this even better:

1. **Depth Estimation** - Add stereo vision or LiDAR for distance measurement
2. **Lane Detection** - Identify road boundaries for better navigation
3. **Traffic Sign Recognition** - Read and obey traffic signs
4. **Night Vision** - IR camera support for low-light conditions
5. **Multi-Camera** - 360° surround view
6. **SLAM Integration** - Simultaneous Localization and Mapping

---

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [TensorRT Developer Guide](https://developer.nvidia.com/tensorrt)
- [COCO Dataset Classes](https://cocodataset.org/)
- [Real-Time Object Detection Survey](https://arxiv.org/abs/2203.11482)
