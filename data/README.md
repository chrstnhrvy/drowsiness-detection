# Data

Place your input data here.

## Directory Structure

```
data/
├── videos/                    # Raw video recordings
├── extracted_frames/          # Frames extracted from videos (every 5th frame)
├── processed_faces_224_rgb/   # MTCNN-cropped faces (224×224)
├── labeled_faces/             # Auto-labeled frames (blinking/yawning/normal)
├── undersampled_faces/        # Class-balanced dataset
├── augmented_faces/           # Augmented training images
└── face_metrics.csv           # Extracted facial landmark features
```

> **Note:** Data directories and large files are excluded from Git via `.gitignore`. Prepare your own dataset or download the original from the project documentation.
