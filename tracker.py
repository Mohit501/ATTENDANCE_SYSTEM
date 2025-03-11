import cv2
import numpy as np
import torch
import pickle
import os
from torch import nn
from transformers import ViTFeatureExtractor, ViTModel
from ultralytics import YOLO
import time
import supervision as sv
from .attendance import AttendanceManager


class PersonTracker:
    def __init__(self):
        # Existing initializations
        self.person_model = YOLO('yolov5n.pt')
        self.last_frame = None 
        
        self.face_model = YOLO('yolov8n-face.pt')
        self.auth_data = self.load_authorization_data("auth_data.pkl")
        print("Loaded authorization data:", self.auth_data)
        
        # Add processed IDs tracking
        self.processed_ids = set()
        
        # Rest of existing initializations
        self.person_tracker = sv.ByteTrack()
        self.face_tracker = sv.ByteTrack()
        self.faces_folder = 'faces'
        self.num_classes = self.get_num_classes()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        if not os.path.exists("vit_full_model.pth"):
            print("❌ vit_full_model.pth is missing. Ensure the model is in the correct path.")
        else:
            self.full_vit_model = self.load_full_vit_model("vit_full_model.pth")
        self.full_vit_model = self.load_full_vit_model("vit_full_model.pth")
        self.full_vit_model.eval()
        self.attendance_manager = AttendanceManager()
        self.total_people = 0
        self.unauthorized_count = 0
        self.manually_authorized = {} 
        
        # Existing face names initialization
        face_names = []
        faces_folder = 'faces'

        if not os.path.exists(faces_folder):
            print("Warning: 'faces' folder not found. Proceeding without preloaded face data.")
            os.makedirs(faces_folder, exist_ok=True)  # Create the folder if missing
        else:
            for person in os.listdir(faces_folder):
                person_path = os.path.join(faces_folder, person)
                if os.path.isdir(person_path):
                    face_names.append(person)

        face_names.sort()
        self.label_to_name = {i: name for i, name in enumerate(face_names)}

        self.tracks = {
            "persons": [],
            "faces": []
        }
        self.authorized_persons = {}
        self.max_frames_missing = 30
        self.smoothing_factor = 0.3
        self.stable_recognition_threshold = 3
        self.recognition_confidence_threshold = 0.7
        self.tracking_history = {}

    def capture_current_frame(self):
        """Return the latest processed frame."""
        if self.last_frame is not None:
            return True, self.last_frame.copy()
        return False, None
    
    def get_num_classes(self):
        """Count the number of unique faces (directories) in the 'faces' folder."""
        if not os.path.exists(self.faces_folder):
            os.makedirs(self.faces_folder, exist_ok=True)
            return 0  # No classes initially

        return len([person for person in os.listdir(self.faces_folder) if os.path.isdir(os.path.join(self.faces_folder, person))])


    def get_counts(self):
        """Return the number of detected people and unauthorized persons."""
        return self.total_people, self.unauthorized_count
        
    def load_authorization_data(self, file_path):
        """Load authorization data from a pickle file and normalize keys to lowercase."""
        try:
            with open(file_path, "rb") as f:
                auth_data = pickle.load(f)
            # Convert dictionary keys to lowercase
            return {key.lower(): value for key, value in auth_data.items()}
        except FileNotFoundError:
            print("Authorization file not found. Using empty dictionary.")
            return {}

    
    def smooth_bbox(self, track_id, new_bbox):
        """
        Smooth bounding box coordinates using exponential moving average.
        
        Args:
            track_id (int): Unique track identifier
            new_bbox (np.ndarray): New bounding box coordinates
        
        Returns:
            np.ndarray: Smoothed bounding box coordinates
        """
        if track_id not in self.tracking_history:
            self.tracking_history[track_id] = {
                'bbox': new_bbox,  # Initialize bbox
                'stable_count': 0,
                'recognition_count': {},
                'last_recognized_name': None,
                'last_seen': time.time()  # Ensure last_seen is also initialized
            }
            return new_bbox  # Return the original bbox if it's the first time

        # Exponential moving average for smoothing
        prev_bbox = self.tracking_history[track_id].get('bbox', new_bbox)  # Default to new_bbox if missing
        smoothed_bbox = (1 - self.smoothing_factor) * prev_bbox + self.smoothing_factor * new_bbox

        # Update tracking history
        self.tracking_history[track_id]['bbox'] = smoothed_bbox
        self.tracking_history[track_id]['last_seen'] = time.time()  # Update last_seen when detected

        return smoothed_bbox


    def process_frame(self, frame, frame_num):
        """Process a single frame with tracking, recognition, and authorization checking."""
        person_tracks, face_tracks = self.detect_and_track(frame, frame_num)
        
        self.unauthorized_count = 0
        self.total_people = 0
        
        face_recognized = {}  # Track recognized faces per track_id
        
        # Process face tracks first
        for i, track_id in enumerate(face_tracks.tracker_id):
            if track_id is not None:
                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, face_tracks.xyxy[i])
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    person_name = self.recognize_face(face_crop).strip().lower()
                    face_recognized[track_id] = person_name
                    
                    if person_name != "unauthorized":
                        authorization_status = self.auth_data.get(person_name, "Unauthorized")
                        if authorization_status == "Authorised":
                            self.authorized_persons[track_id] = {
                                'name': person_name,
                                'last_seen': frame_num,
                                'frames_missing': 0
                            }
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box for detected faces
        
        # Process person tracks
        for i, track_id in enumerate(person_tracks.tracker_id):
            if track_id is not None:
                self.total_people += 1
                track_id = int(track_id)
                str_track_id = str(track_id)

                bbox = person_tracks.xyxy[i]
                smoothed_bbox = self.smooth_bbox(track_id, bbox)
                x1, y1, x2, y2 = map(int, smoothed_bbox)

                # Check if person is manually authorized
                if str_track_id in self.manually_authorized:
                    auth_info = self.manually_authorized[str_track_id]
                    color = (0, 255, 0)  # Green for authorized
                    label_text = f"{auth_info['name']} - Authorised"
                    self.authorized_persons[track_id] = {
                        'name': auth_info['name'],
                        'last_seen': frame_num,
                        'frames_missing': 0
                    }
                # Check if person is authorized through face recognition
                elif track_id in self.authorized_persons:
                    auth_info = self.authorized_persons[track_id]
                    color = (0, 255, 0)  # Green for authorized
                    label_text = f"{auth_info['name']} - Authorised"
                else:
                    # Assign colors based on recognition status
                    if track_id in face_recognized:
                        if face_recognized[track_id] != "unauthorized":
                            color = (0, 255, 0)  # Green for authorized
                            label_text = f"{face_recognized[track_id]} - Authorised"
                        else:
                            color = (0, 0, 255)  # Red for unauthorized
                            label_text = "Unauthorized"
                            self.unauthorized_count += 1
                    else:
                        color = (0, 255, 255)  # Yellow for detected but not yet recognized
                        label_text = "Identifying..."
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                font_scale, thickness = 1.0, 2
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

                # Update tracking history
                self.tracking_history[track_id] = self.tracking_history.get(track_id, {})
                self.tracking_history[track_id]['last_seen'] = time.time()

                # Process attendance if recognized
                if label_text != "Identifying..." and label_text != "Unauthorized":
                    status_msg, success = self.attendance_manager.process_recognition(
                        employee_id=str(track_id),
                        name=label_text.split(" - ")[0],
                        confidence=0.9
                    )
                    if success:
                        print(f"Attendance updated: {status_msg}")
        
        self.last_frame = frame.copy()
        return frame
    def load_full_vit_model(self, model_path):
        """Load and initialize the ViT model with classifier, handling missing file."""
        class FullViTModel(nn.Module):
            def __init__(self, num_classes):
                super(FullViTModel, self).__init__()
                self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
                self.classifier = nn.Sequential(
                    nn.Dropout(0.3),  # Dropout layer
                    nn.Linear(self.vit.config.hidden_size, num_classes)  # Fully connected layer
                )

            def forward(self, pixel_values):
                outputs = self.vit(pixel_values=pixel_values)
                cls_token_output = outputs.last_hidden_state[:, 0, :]
                logits = self.classifier(cls_token_output)
                return logits

        model = FullViTModel(self.num_classes)  # Use dynamically calculated num_classes

        if not os.path.exists(model_path):
            print(f"Warning: Model file '{model_path}' not found. Proceeding without loading weights.")
            return model  # Return model without loading weights

        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error loading model: {e}. Proceeding with untrained model.")

        return model
    
    def load_model(self, model_path):
        """Load a newly trained model."""
        self.full_vit_model = self.load_full_vit_model(model_path)
        self.full_vit_model.eval()
        print(f"Model {model_path} loaded successfully!")

    def preprocess_face(self, face_image):
            """Preprocess face image for recognition."""
            if face_image.size == 0:
                return None
                
            try:
                # Resize face image to expected size (224x224)
                face_image = cv2.resize(face_image, (224, 224))
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                inputs = self.feature_extractor(images=face_rgb, return_tensors="pt")
                return inputs["pixel_values"]
            except Exception as e:
                print(f"Error preprocessing face: {e}")
                return None

    def recognize_face(self, face_image):
        """Recognize face using ViT model."""
        try:
            pixel_values = self.preprocess_face(face_image)
            if pixel_values is None:
                return "Unauthorized"

            with torch.no_grad():
                logits = self.full_vit_model(pixel_values)
                probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

            label_index = np.argmax(probabilities)
            confidence = probabilities[label_index]

            if confidence > 0.8:
                return self.label_to_name.get(label_index, "Unauthorized")
            return "Unauthorized"
        except Exception as e:
            print(f"Error recognizing face: {e}")
            return "Unauthorized"
        
    def filter_boxes(self, boxes, scores, aspect_ratio_range=(0.1, 10.0), 
                 size_range=(100, 100000), 
                 min_width=0, 
                 min_height=0, 
                 detection_type='person'):

        filtered_boxes = []
        filtered_scores = []

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            # Width and height checks
            if width <= min_width or height <= min_height:
                continue
                    
            area = width * height
            aspect_ratio = height / width

            # Face-specific filtering
            if detection_type == 'face':
                # Very strict face criteria
                if (0.8 <= aspect_ratio <= 1.2 and  # Almost square
                    500 <= area <= 20000 and  # Moderate face size
                    width >= 50 and height >= 50):  # Minimum dimensions
                    filtered_boxes.append(box)
                    filtered_scores.append(score)
            
            # Person-specific filtering
            elif detection_type == 'person':
                # Person detection criteria
                if (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1] and
                    size_range[0] <= area <= size_range[1]):
                    filtered_boxes.append(box)
                    filtered_scores.append(score)

        return (np.array(filtered_boxes, dtype=np.float32),
                np.array(filtered_scores, dtype=np.float32))

    def detect_and_track(self, frame, frame_num):
        """Comprehensive detection and tracking method."""
        # Detection configuration
        person_confidence_threshold = 0.3
        face_confidence_threshold = 0.3
        iou_threshold = 0.4

        # Person detection
        person_results = self.person_model.predict(
            frame, 
            conf=person_confidence_threshold, 
            iou=iou_threshold, 
            max_det=100,
            verbose=False
        )[0]
        
        # Face detection 
        face_results = self.face_model.predict(
            frame, 
            conf=face_confidence_threshold, 
            iou=iou_threshold, 
            max_det=50,
            verbose=False
        )[0]

        # Process person detections
        person_mask = person_results.boxes.cls.cpu().numpy() == 0
        person_boxes = person_results.boxes.xyxy.cpu().numpy()[person_mask]
        person_scores = person_results.boxes.conf.cpu().numpy()[person_mask]

        # Process face detections
        face_boxes = face_results.boxes.xyxy.cpu().numpy()
        face_scores = face_results.boxes.conf.cpu().numpy()

        # Debug detections
        

        # Filter person boxes
        if len(person_boxes) > 0:
            person_boxes, person_scores = self.filter_boxes(
                person_boxes,
                person_scores,
                detection_type='person',
                aspect_ratio_range=(0.2, 4.0),
                size_range=(1000, 300000)
            )

        # Filter face boxes
        if len(face_boxes) > 0:
            face_boxes, face_scores = self.filter_boxes(
                face_boxes,
                face_scores,
                detection_type='face'
            )

        # Debug filtered detections
        

        # Create person detections
        person_detections = sv.Detections.empty()
        if len(person_boxes) > 0:
            person_detections = sv.Detections(
                xyxy=np.array(person_boxes),
                confidence=np.array(person_scores)
            )

        # Create face detections
        face_detections = sv.Detections.empty()
        if len(face_boxes) > 0:
            face_detections = sv.Detections(
                xyxy=np.array(face_boxes),
                confidence=np.array(face_scores)
            )

        # Update trackers
        person_tracks = self.person_tracker.update_with_detections(detections=person_detections)
        face_tracks = self.face_tracker.update_with_detections(detections=face_detections)
        print(f"Detected {len(face_tracks.tracker_id)} faces in the frame.")
        current_person_ids = set()
        self.tracks["persons"].append({})
        self.tracks["faces"].append({})

        # Process person tracks
        for i, track_id in enumerate(person_tracks.tracker_id):
            if track_id is not None:
                track_id = int(track_id)
                current_person_ids.add(track_id)
                bbox = person_tracks.xyxy[i]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area > 0:
                    self.tracks["persons"][frame_num][track_id] = {"bbox": bbox.tolist()}
                self.tracking_history[track_id] = self.tracking_history.get(track_id, {})
                self.tracking_history[track_id]['last_seen'] = time.time()

        # Process face tracks
        for i, track_id in enumerate(face_tracks.tracker_id):
            if track_id is not None:
                bbox = face_tracks.xyxy[i]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area > 0:
                    self.tracks["faces"][frame_num][int(track_id)] = {"bbox": bbox.tolist()}

        # Authorized persons management
        to_remove = []
        for track_id in self.authorized_persons:
            if track_id not in current_person_ids:
                self.authorized_persons[track_id]['frames_missing'] = \
                    self.authorized_persons[track_id].get('frames_missing', 0) + 1
                if self.authorized_persons[track_id]['frames_missing'] > self.max_frames_missing:
                    to_remove.append(track_id)
            else:
                self.authorized_persons[track_id]['frames_missing'] = 0
                self.authorized_persons[track_id]['last_seen'] = frame_num

        # Remove old tracks
        for track_id in to_remove:
            del self.authorized_persons[track_id]

        return person_tracks, face_tracks
    def process_video(self, input_path, output_path):
        """Process entire video file."""
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.process_frame(frame, frame_num)
            out.write(processed_frame)

            # Display frame (optional)
            cv2.imshow("Processed Video", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_num += 1

        # Clean up
        cap.release()
        out.release()
        cv2.destroyAllWindows()




    def get_current_tracks(self):
        """Return only currently detected people (who have been seen within the last 3 seconds)."""
        tracks = {}
        current_time = time.time()
        
        for track_id, info in self.tracking_history.items():
            last_seen = info.get('last_seen', 0)

            # 🔥 Only keep tracks that have been detected in the last 3 seconds
            if (current_time - last_seen) < 3:
                tracks[str(track_id)] = {
                    'name': info.get('name', 'Unknown'),
                    'authorized': info.get('authorized', False)
                }

        return tracks
    def update_authorization(self, track_id, name):
        """Update authorization status and mark as processed."""
        track_id = str(track_id)
        self.manually_authorized[track_id] = {
            'name': name,
            'authorized': True
        }
        print(f"Updated manual authorization for track_id {track_id}: {name}")
        
        # If the track_id exists in numeric form, update it as well
        numeric_track_id = int(track_id)
        if numeric_track_id in self.authorized_persons:
            self.authorized_persons[numeric_track_id]['name'] = name
        else:
            self.authorized_persons[numeric_track_id] = {
                'name': name,
                'last_seen': 0,
                'frames_missing': 0
            }



    def process_checkouts(self):
        """Process checkouts for persons no longer detected."""
        to_remove = []
        
        for track_id in self.authorized_persons:
            if self.authorized_persons[track_id]['frames_missing'] > self.max_frames_missing:
                # Person has been missing for too long, check them out
                checkout_msg = self.attendance_manager.process_checkout(str(track_id))
                if checkout_msg:
                    print(f"Checkout processed: {checkout_msg}")
                to_remove.append(track_id)
        
        # Remove processed checkouts
        for track_id in to_remove:
            del self.authorized_persons[track_id]
            if track_id in self.tracking_history:
                del self.tracking_history[track_id]