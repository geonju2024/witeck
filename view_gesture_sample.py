import cv2 as cv
import numpy as np
import os

# =========================
# Change the sample path here
# =========================
SAMPLE_PATH = r"gesture_dataset\user1\ges3\sample_005.npy"

# MediaPipe hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

data = np.load(SAMPLE_PATH)

print("file:", SAMPLE_PATH)
print("shape:", data.shape)

if data.shape[1] != 63:
    raise ValueError("This file is not one-hand data with 63 features.")

# data shape: (60, 63)
# each frame: 21 points, each point has x, y, z
frames = data.reshape(data.shape[0], 21, 3)

# Use all x and y values to auto-scale the view
all_x = frames[:, :, 0]
all_y = frames[:, :, 1]

min_x, max_x = all_x.min(), all_x.max()
min_y, max_y = all_y.min(), all_y.max()

# Avoid division by zero
range_x = max_x - min_x if max_x - min_x != 0 else 1
range_y = max_y - min_y if max_y - min_y != 0 else 1

canvas_size = 600

while True:
    for idx, frame_landmarks in enumerate(frames):
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

        points = []

        for lm in frame_landmarks:
            x, y, z = lm

            # Normalize to canvas
            px = int((x - min_x) / range_x * 400 + 100)
            py = int((y - min_y) / range_y * 400 + 100)

            points.append((px, py))
        # Draw connections
        # Draw connections
        for start, end in HAND_CONNECTIONS:
            cv.line(canvas, points[start], points[end], (0, 0, 0), 2)
        # Draw landmark points
        # Draw landmark points
        for i, p in enumerate(points):
            cv.circle(canvas, p, 5, (0, 0, 255), -1)
            cv.putText(canvas, str(i), (p[0] + 5, p[1] - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        cv.putText(canvas, f"Frame: {idx + 1}/{len(frames)}", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv.putText(canvas, os.path.basename(SAMPLE_PATH), (20, 80),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv.imshow("Gesture Viewer", canvas)

        key = cv.waitKey(80) & 0xFF

        if key == ord('q'):
            cv.destroyAllWindows()
            exit()

        if key == ord('p'):
            cv.waitKey(0)
