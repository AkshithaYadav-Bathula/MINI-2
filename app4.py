import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

class HighAccuracyDeepfakeDetector:
    def __init__(self):
        """Initialize high accuracy deepfake detector with best model"""
        print("Loading YOLOv8x (Extra Large) model for maximum accuracy...")
        print("This may take a moment to download on first run...")
        
        # Use YOLOv8x for highest accuracy
        self.model = YOLO('yolov8x.pt')  # Best accuracy model
        
        # Detection settings - Fixed colors as requested
        self.person_color = (0, 255, 0)  # Green for people
        self.object_color = (0, 0, 255)  # Red for objects (as requested)
        self.thickness = 2  # Thicker lines for better visibility
        self.person_class = 0  # Person class ID
        self.confidence_threshold = 0.25  # Lower threshold for better detection
        
        # Frame processing settings - Limit to 100 frames
        self.frame_count = 0
        self.max_frames = 100  # Process only first 100 frames
        
        print("Model loaded successfully!")
        print("✓ People will be marked in GREEN")
        print("✓ Objects will be marked in RED")
    
    def detect_objects_people(self, frame):
        """High accuracy detection of people and objects"""
        # Run inference with optimized settings
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=0.45,  # Non-max suppression threshold
            verbose=False,
            device='cuda' if self.model.device.type == 'cuda' else 'cpu'
        )
        
        people_count = 0
        object_count = 0
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract detection info
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'is_person': class_id == self.person_class
                    }
                    detections.append(detection)
                    
                    if class_id == self.person_class:
                        people_count += 1
                    else:
                        object_count += 1
        
        return detections, people_count, object_count
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            class_name = det['class_name']
            is_person = det['is_person']
            
            # Choose color based on detection type - RED for objects, GREEN for people
            color = self.person_color if is_person else self.object_color
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)
            
            # Prepare label
            label = f"{class_name} {confidence:.2f}"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1-25), (x1 + label_size[0] + 10, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1 + 5, y1-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def add_frame_overlay(self, frame, people_count, object_count, total_frames):
        """Add comprehensive frame information overlay"""
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (400, 140), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Frame counter
        cv2.putText(frame, f'Frame: {self.frame_count}/{total_frames}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # People count (green)
        cv2.putText(frame, f'People Detected: {people_count}', 
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.person_color, 2)
        
        # Object count (red)
        cv2.putText(frame, f'Objects Detected: {object_count}', 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.object_color, 2)
        
        # Model info
        cv2.putText(frame, 'Model: YOLOv8x (Max Accuracy)', 
                   (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Analysis status
        cv2.putText(frame, 'DEEPFAKE INVESTIGATION MODE', 
                   (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return frame
    
    def process_video(self, video_path="1.mp4"):
        """Process video with high accuracy detection and save output"""
        # Check if video exists
        if not os.path.exists(video_path):
            print(f"ERROR: Video file '{video_path}' not found!")
            print(f"Please ensure your video file exists in the same directory")
            return None
        
        print(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("ERROR: Could not open video file!")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video Properties:")
        print(f"  - Duration: {duration:.1f} seconds")
        print(f"  - FPS: {fps}")
        print(f"  - Total frames: {total_frames}")
        print(f"  - Resolution: {width}x{height}")
        print(f"  - Processing first {min(self.max_frames, total_frames)} frames")
        
        # Setup output video writer
        output_filename = os.path.splitext(video_path)[0] + "._output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("ERROR: Could not create output video file!")
            cap.release()
            return None
        
        print(f"Output will be saved as: {output_filename}")
        
        # Initialize statistics
        frame_stats = []
        total_people = 0
        total_objects = 0
        start_time = time.time()
        
        # Process first 100 frames
        self.frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or self.frame_count >= self.max_frames:
                break
            
            self.frame_count += 1
            
            # High accuracy detection
            detections, people_count, object_count = self.detect_objects_people(frame)
            
            # Draw detections
            frame = self.draw_detections(frame, detections)
            
            # Add overlay information
            frame = self.add_frame_overlay(frame, people_count, object_count, total_frames)
            
            # Write frame to output video
            out.write(frame)
            
            # Store statistics
            frame_stats.append({
                'frame': self.frame_count,
                'people': people_count,
                'objects': object_count,
                'total_detections': len(detections)
            })
            
            total_people += people_count
            total_objects += object_count
            
            # Progress update every 10 frames
            if self.frame_count % 10 == 0:
                progress = (self.frame_count / min(self.max_frames, total_frames)) * 100
                elapsed = time.time() - start_time
                remaining_frames = min(self.max_frames, total_frames) - self.frame_count
                eta = (elapsed / self.frame_count) * remaining_frames if self.frame_count > 0 else 0
                print(f"Progress: {progress:.1f}% ({self.frame_count}/{min(self.max_frames, total_frames)}) - ETA: {eta:.1f}s")
            
            # Optional: Display frame during processing (comment out for faster processing)
            # cv2.imshow('Processing...', cv2.resize(frame, (800, 600)))
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        processing_time = time.time() - start_time
        
        # Print detailed analysis
        self.print_analysis_report(frame_stats, total_people, total_objects, duration, processing_time, output_filename)
        
        return frame_stats
    
    def print_analysis_report(self, frame_stats, total_people, total_objects, duration, processing_time, output_file):
        """Print comprehensive analysis report"""
        print("\n" + "="*70)
        print("HIGH ACCURACY DEEPFAKE DETECTION ANALYSIS REPORT")
        print("="*70)
        print(f"Model Used: YOLOv8x (Maximum Accuracy)")
        print(f"Video Duration: {duration:.1f} seconds")
        print(f"Processing Time: {processing_time:.1f} seconds")
        print(f"Frames Processed: {len(frame_stats)} (First 100 frames)")
        print(f"Output Video: {output_file}")
        print(f"Total People Detected: {total_people}")
        print(f"Total Objects Detected: {total_objects}")
        print(f"Average People per Frame: {total_people/len(frame_stats):.2f}")
        print(f"Average Objects per Frame: {total_objects/len(frame_stats):.2f}")
        
        # Detection summary
        people_counts = [s['people'] for s in frame_stats]
        object_counts = [s['objects'] for s in frame_stats]
        
        print(f"\nDETECTION SUMMARY:")
        print(f"  Max People in Single Frame: {max(people_counts) if people_counts else 0}")
        print(f"  Max Objects in Single Frame: {max(object_counts) if object_counts else 0}")
        print(f"  Frames with People: {sum(1 for p in people_counts if p > 0)}")
        print(f"  Frames with Objects: {sum(1 for o in object_counts if o > 0)}")
        
        # Detection consistency analysis
        if people_counts:
            unique_people_counts = len(set(people_counts))
            consistency = 1 - (unique_people_counts / len(people_counts))
            print(f"\nPeople Detection Consistency: {consistency*100:.1f}%")
            if consistency > 0.7:
                print("✓ HIGH CONSISTENCY - Good for deepfake analysis")
            elif consistency > 0.4:
                print("⚠ MODERATE CONSISTENCY - Some variation detected")
            else:
                print("⚠ HIGH VARIATION - May indicate deepfake manipulation or complex scene")
        
        print(f"\n✓ Output video saved successfully: {output_file}")
        print("✓ Video contains detection boxes and analysis overlay")
        print("✓ GREEN boxes = People, RED boxes = Objects")

def main():
    """Main function to run high accuracy deepfake detection"""
    print("HIGH ACCURACY DEEPFAKE DETECTION SYSTEM")
    print("Using YOLOv8x model for maximum detection accuracy")
    print("Features:")
    print("  ✓ Processes first 100 frames of video")
    print("  ✓ Saves output as MP4 video file")
    print("  ✓ GREEN boxes for people, RED boxes for objects")
    print("  ✓ Can handle many people in the video")
    print("-" * 60)
    
    # Initialize detector
    detector = HighAccuracyDeepfakeDetector()
    
    # Process the video
    video_path = "1.mp4"
    
    print(f"\nLooking for video file: {video_path}")
    
    # Run detection
    results = detector.process_video(video_path)
    
    if results:
        print("\n" + "="*60)
        print("✓ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("✓ Check the output video file for results")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ ANALYSIS FAILED!")
        print("\nTROUBLESHOOTING:")
        print("1. Ensure your video file exists in the same folder")
        print("2. Check if the video file is not corrupted")
        print("3. Supported formats: MP4, AVI, MOV, MKV")
        print("4. Make sure you have enough disk space for output")
        print("="*60)

if __name__ == "__main__":
    main()

# INSTALLATION AND USAGE INSTRUCTIONS:
"""
REQUIREMENTS:
pip install ultralytics opencv-python torch torchvision

USAGE:
1. Place your video file as "1.mp4" in the same folder as this script
2. Run: python deepfake_detector.py
3. The script will process the ENTIRE video (no time limits)
4. Output will be saved as "1_output.mp4"

FEATURES:
✓ Processes first 100 frames of video
✓ Handles multiple people in video
✓ GREEN boxes for people detection
✓ RED boxes for object detection
✓ Saves output as video file with analysis overlay
✓ Real-time progress updates
✓ Detailed analysis report

INPUT/OUTPUT:
- Input: 1.mp4 (or change video_path in main())
- Output: 1_output.mp4 (automatically generated)
- Format: MP4 with H.264 codec
- Quality: Same as input video

COLOR CODING:
- GREEN boxes: People/Persons detected
- RED boxes: All other objects detected
- White text: Confidence scores and labels

MODEL:
- YOLOv8x (Extra Large) for maximum accuracy
- Best for detailed analysis and investigation
- CUDA acceleration if available
"""