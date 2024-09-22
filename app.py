import base64
import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import cv2

# Initialize Flask app
app = Flask(__name__)

# Load YOLOv8 model (ensure the best.pt is in the same folder as this script)
model = YOLO("best.pt")

# Ensure the output folder exists
if not os.path.exists("static/output"):
    os.makedirs("static/output")

# Route to process the video and detect the best license plate
@app.route('/detect_video', methods=['POST'])
def detect_best_plate_from_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'})

    # Save uploaded video file
    video_file = request.files['video']
    video_path = os.path.join('static/output', video_file.filename)
    video_file.save(video_path)

    # Process the video using OpenCV
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, first_frame = cap.read()
    cap.release()  # Release the video capture object as we only need the first frame

    if not ret:
        return jsonify({'error': 'Failed to read video'})

    best_plate = None
    best_confidence = 0
    best_box = None

    # Run YOLOv8 inference on the first frame
    results = model(first_frame)

    # Check for detections
    if len(results) > 0 and results[0].boxes is not None:
        for i, box in enumerate(results[0].boxes.xyxy):
            confidence = results[0].boxes.conf[i].item()  # Confidence score
            x1, y1, x2, y2 = map(int, box.cpu().numpy())  # Bounding box

            # Get the size of the bounding box (area)
            box_area = (x2 - x1) * (y2 - y1)

            # Select the largest plate with the highest confidence
            if confidence > best_confidence and box_area > 1000:  # Adjust area threshold if needed
                best_confidence = confidence
                best_box = (x1, y1, x2, y2)

    # Ensure we found a plate
    if best_box:
        # Crop the best detected license plate
        x1, y1, x2, y2 = best_box
        cropped_plate = first_frame[y1:y2, x1:x2]

        # Convert cropped image to base64
        cropped_image_pil = Image.fromarray(cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB))
        buffered = BytesIO()
        cropped_image_pil.save(buffered, format="JPEG")
        cropped_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Return the base64 image of the cropped license plate
        return jsonify({
            'cropped_image_base64': cropped_base64
        })
    else:
        return jsonify({'error': 'No license plate detected'})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to port 5000 for local development
    app.run(host="0.0.0.0", port=port)
