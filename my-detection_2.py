import jetson.inference
import jetson.utils

# Initialize the detection network
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# Load the image (replace 'your_image.jpg' with your image path)
img = jetson.utils.loadImage("images.jpg")

# Perform detection
detections = net.Detect(img)

# Process detections and filter for "person" class (ClassID 1 in ssd-mobilenet-v2)
target_class = "person"  # Change this to your desired class
for detection in detections:
    class_name = net.GetClassDesc(detection.ClassID)
    
    # Only process if the detection is of the target class
    if class_name == target_class:
        # Get the coordinates
        left = detection.Left
        top = detection.Top
        right = detection.Right
        bottom = detection.Bottom
        center = detection.Center
        
        # Get confidence score
        confidence = detection.Confidence
        
        # Print the information
        print(f"\nDetected {class_name}:")
        print(f"Confidence: {confidence:.2f}")
        print(f"Bounding Box: left={left:.1f}, top={top:.1f}, right={right:.1f}, bottom={bottom:.1f}")
        print(f"Center: x={center[0]:.1f}, y={center[1]:.1f}")

# Save the output image with detections drawn (optional)
jetson.utils.saveImage("output_images.jpg", img)
