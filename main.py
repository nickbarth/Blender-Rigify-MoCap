"""
Blender Webcam Motion Capture for Rigify Meta Rig - Upper Body Only
Captures motion from webcam using MediaPipe and applies to upper body Rigify bones
"""

import bpy
import subprocess
import sys
import json
import socket
import threading
import time
import os
import tempfile
from mathutils import Vector, Euler, Quaternion, Matrix
from typing import Dict, Optional, Tuple
import math

# Motion capture subprocess code (will be written to temp file and executed)
MOCAP_SUBPROCESS_CODE = '''
import cv2
import mediapipe as mp
import numpy as np
import math
import json
import socket
import sys
import signal

# Handle graceful shutdown
running = True
def signal_handler(sig, frame):
    global running
    running = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class MediaPipeSolver:
    """Use MediaPipe's built-in pose detection and calculations"""
    
    @staticmethod
    def mediapipe_to_blender(vec):
        """Convert MediaPipe coordinates to Blender coordinate system
        MediaPipe: X right, Y down, Z forward
        Blender: X right, Y forward, Z up
        Conversion: (x, y, z) -> (x, -z, y)
        """
        return np.array([vec[0], -vec[2], vec[1]])
    
    @staticmethod
    def calculate_angle(a, b, c):
        """Calculate angle at point b between points a and c"""
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return angle
    
    @staticmethod
    def calc_head_rotation(pose_landmarks, face_landmarks=None) -> dict:
        """Calculate head rotation using pose landmarks"""
        if not pose_landmarks:
            return {"x": 0, "y": 0, "z": 0}
        
        landmarks = pose_landmarks.landmark
        
        # Use ear and nose landmarks for head orientation
        nose = landmarks[0]
        left_ear = landmarks[7] if len(landmarks) > 7 else None
        right_ear = landmarks[8] if len(landmarks) > 8 else None
        
        head_rot = {"x": 0, "y": 0, "z": 0}
        
        # Calculate head yaw from ear positions relative to nose
        if left_ear and right_ear:
            # Yaw estimation (left/right turn)
            ear_center_x = (left_ear.x + right_ear.x) / 2
            head_rot["y"] = (nose.x - ear_center_x) * 1.5
            
            # Roll estimation from ear height difference
            head_rot["z"] = (right_ear.y - left_ear.y) * 0.5
        
        # Use shoulder-to-nose vector for pitch
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        if left_shoulder and right_shoulder:
            shoulder_center = np.array([
                (left_shoulder.x + right_shoulder.x) / 2,
                (left_shoulder.y + right_shoulder.y) / 2,
                (left_shoulder.z + right_shoulder.z) / 2
            ])
            nose_pos = np.array([nose.x, nose.y, nose.z])
            
            # Pitch based on nose height relative to shoulders
            vertical_diff = nose_pos[1] - shoulder_center[1]
            head_rot["x"] = vertical_diff * 0.5
        
        return head_rot
    
    @staticmethod
    def calc_arm_rotations(landmarks) -> dict:
        """Calculate arm rotations from pose landmarks for Rigify with proper coordinate conversion"""
        result = {
            "upper_arm_L": {"x": 0, "y": 0, "z": 0},
            "forearm_L": {"x": 0, "y": 0, "z": 0},
            "upper_arm_R": {"x": 0, "y": 0, "z": 0},
            "forearm_R": {"x": 0, "y": 0, "z": 0}
        }
        
        if not landmarks:
            return result
        
        lm = landmarks.landmark
        
        # Left arm
        if all(lm[i] for i in [11, 13, 15]):  # shoulder, elbow, wrist
            shoulder = lm[11]
            elbow = lm[13]
            wrist = lm[15]
            
            # Get MediaPipe vectors
            shoulder_vec = np.array([shoulder.x, shoulder.y, shoulder.z])
            elbow_vec = np.array([elbow.x, elbow.y, elbow.z])
            wrist_vec = np.array([wrist.x, wrist.y, wrist.z])
            
            # Convert to Blender coordinate system
            shoulder_bl = MediaPipeSolver.mediapipe_to_blender(shoulder_vec)
            elbow_bl = MediaPipeSolver.mediapipe_to_blender(elbow_vec)
            wrist_bl = MediaPipeSolver.mediapipe_to_blender(wrist_vec)
            
            # Calculate shoulder to elbow vector in Blender space
            shoulder_to_elbow = elbow_bl - shoulder_bl
            
            # Normalize
            length = np.linalg.norm(shoulder_to_elbow)
            if length > 0.01:
                shoulder_to_elbow = shoulder_to_elbow / length
                
                # For Rigify, calculate rotation from T-pose
                # In T-pose, left arm points to the left (-X direction)
                rest_direction = np.array([-1, 0, 0])
                
                # Calculate rotation needed to align rest_direction with shoulder_to_elbow
                # Using spherical coordinates for intuitive control
                
                # Forward/back rotation (around Z axis)
                forward_angle = np.arctan2(shoulder_to_elbow[1], -shoulder_to_elbow[0])
                
                # Up/down rotation (around Y axis)
                horizontal_length = np.sqrt(shoulder_to_elbow[0]**2 + shoulder_to_elbow[1]**2)
                vertical_angle = np.arctan2(shoulder_to_elbow[2], horizontal_length)
                
                # Apply corrective rotations for Rigify bone orientation
                result["upper_arm_L"]["x"] = vertical_angle  # Up/down
                result["upper_arm_L"]["y"] = 0  # No twist for now
                result["upper_arm_L"]["z"] = forward_angle - math.pi/2  # Forward/back (adjusted for T-pose)
            
            # Forearm: elbow bend
            elbow_angle = MediaPipeSolver.calculate_angle(shoulder, elbow, wrist)
            # Rigify expects positive values for elbow bend
            result["forearm_L"]["y"] = max(0, math.pi - elbow_angle) * 0.8
        
        # Right arm
        if all(lm[i] for i in [12, 14, 16]):  # shoulder, elbow, wrist
            shoulder = lm[12]
            elbow = lm[14]
            wrist = lm[16]
            
            # Get MediaPipe vectors
            shoulder_vec = np.array([shoulder.x, shoulder.y, shoulder.z])
            elbow_vec = np.array([elbow.x, elbow.y, elbow.z])
            wrist_vec = np.array([wrist.x, wrist.y, wrist.z])
            
            # Convert to Blender coordinate system
            shoulder_bl = MediaPipeSolver.mediapipe_to_blender(shoulder_vec)
            elbow_bl = MediaPipeSolver.mediapipe_to_blender(elbow_vec)
            wrist_bl = MediaPipeSolver.mediapipe_to_blender(wrist_vec)
            
            # Calculate shoulder to elbow vector in Blender space
            shoulder_to_elbow = elbow_bl - shoulder_bl
            
            # Normalize
            length = np.linalg.norm(shoulder_to_elbow)
            if length > 0.01:
                shoulder_to_elbow = shoulder_to_elbow / length
                
                # For Rigify, calculate rotation from T-pose
                # In T-pose, right arm points to the right (+X direction)
                rest_direction = np.array([1, 0, 0])
                
                # Forward/back rotation (around Z axis)
                forward_angle = np.arctan2(shoulder_to_elbow[1], shoulder_to_elbow[0])
                
                # Up/down rotation (around Y axis)
                horizontal_length = np.sqrt(shoulder_to_elbow[0]**2 + shoulder_to_elbow[1]**2)
                vertical_angle = np.arctan2(shoulder_to_elbow[2], horizontal_length)
                
                # Apply corrective rotations for Rigify bone orientation
                result["upper_arm_R"]["x"] = vertical_angle  # Up/down
                result["upper_arm_R"]["y"] = 0  # No twist for now
                result["upper_arm_R"]["z"] = forward_angle + math.pi/2  # Forward/back (adjusted for T-pose)
            
            # Forearm: elbow bend
            elbow_angle = MediaPipeSolver.calculate_angle(shoulder, elbow, wrist)
            # Rigify expects positive values for elbow bend
            result["forearm_R"]["y"] = max(0, math.pi - elbow_angle) * 0.8
        
        return result
    
    @staticmethod
    def calc_spine_rotation(landmarks) -> dict:
        """Calculate spine/torso rotation"""
        spine_rot = {"x": 0, "y": 0, "z": 0}
        
        if not landmarks:
            return spine_rot
        
        lm = landmarks.landmark
        
        # Check if we have required landmarks
        if all(lm[i] for i in [11, 12]):
            left_shoulder = lm[11]
            right_shoulder = lm[12]
            
            # Calculate shoulder vector
            shoulder_vec = np.array([
                right_shoulder.x - left_shoulder.x,
                right_shoulder.y - left_shoulder.y,
                right_shoulder.z - left_shoulder.z
            ])
            
            # Convert to Blender space
            shoulder_vec_bl = MediaPipeSolver.mediapipe_to_blender(shoulder_vec)
            
            # Calculate lean from shoulder tilt
            spine_rot["z"] = np.arctan2(shoulder_vec_bl[2], abs(shoulder_vec_bl[0]) + 0.1) * 0.3
            
            # Forward/backward lean from shoulder depth
            shoulder_center_z = (left_shoulder.z + right_shoulder.z) / 2
            spine_rot["x"] = -shoulder_center_z * 0.5
            
            # Rotation from shoulder line
            spine_rot["y"] = np.arctan2(shoulder_vec_bl[1], shoulder_vec_bl[0]) * 0.3
        
        return spine_rot


def format_rotation_string(rotation_dict):
    """Format rotation dictionary as string with degrees"""
    if not rotation_dict:
        return "X:0° Y:0° Z:0°"
    
    x_deg = math.degrees(rotation_dict.get('x', 0))
    y_deg = math.degrees(rotation_dict.get('y', 0))
    z_deg = math.degrees(rotation_dict.get('z', 0))
    
    return f"X:{x_deg:+.1f}° Y:{y_deg:+.1f}° Z:{z_deg:+.1f}°"


def main(port):
    global running
    
    # Setup socket server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("localhost", port))
    server_socket.listen(1)
    server_socket.settimeout(1.0)  # Non-blocking accept
    
    print(f"Motion capture server started on port {port}")
    
    # Wait for connection
    client_socket = None
    while running and client_socket is None:
        try:
            client_socket, address = server_socket.accept()
            client_socket.settimeout(0.1)
            print(f"Connected to Blender at {address}")
        except socket.timeout:
            continue
    
    if not client_socket:
        return
    
    # Initialize capture
    cap = cv2.VideoCapture(0)
    solver = MediaPipeSolver()
    
    # Store current rotations for display
    current_rotations = {}
    
    # Use Holistic for combined face and pose tracking
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=False) as holistic:
        
        while running:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Process with holistic model
            results = holistic.process(rgb_frame)
            
            motion_data = {}
            
            # Extract pose data using MediaPipe's landmarks
            if results.pose_world_landmarks:
                # Calculate rotations using MediaPipe solver
                motion_data["arms"] = solver.calc_arm_rotations(results.pose_world_landmarks)
                motion_data["spine"] = solver.calc_spine_rotation(results.pose_world_landmarks)
                motion_data["head"] = solver.calc_head_rotation(
                    results.pose_world_landmarks, 
                    results.face_landmarks
                )
                
                # Store for display
                current_rotations = motion_data
            
            # Send data to Blender
            if motion_data:
                try:
                    data_str = json.dumps(motion_data)
                    message = f"{len(data_str):08d}{data_str}"
                    client_socket.sendall(message.encode())
                except (socket.error, BrokenPipeError):
                    print("Connection lost")
                    break
            
            # Check for stop command
            try:
                command = client_socket.recv(4, socket.MSG_DONTWAIT)
                if command == b"STOP":
                    break
            except:
                pass
            
            # Display preview
            rgb_frame.flags.writeable = True
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Draw landmarks using MediaPipe's drawing utilities
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            
            # Add bone labels with XYZ rotations if pose detected
            if results.pose_landmarks:
                height, width = frame.shape[:2]
                landmarks = results.pose_landmarks.landmark
                
                # Define which bones to show with their rotation data
                bone_rotations = {
                    0: ("head", current_rotations.get("head", {})),
                    11: ("shoulder.L", {}),
                    12: ("shoulder.R", {}),
                    13: ("upper_arm.L", current_rotations.get("arms", {}).get("upper_arm_L", {})),
                    14: ("upper_arm.R", current_rotations.get("arms", {}).get("upper_arm_R", {})),
                    15: ("forearm.L", current_rotations.get("arms", {}).get("forearm_L", {})),
                    16: ("forearm.R", current_rotations.get("arms", {}).get("forearm_R", {})),
                }
                
                for idx, (bone_name, rotation) in bone_rotations.items():
                    if idx < len(landmarks):
                        landmark = landmarks[idx]
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        
                        # Draw bone name
                        cv2.putText(frame, bone_name, (x + 5, y - 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                                  (0, 255, 0), 1, cv2.LINE_AA)
                        
                        # Draw XYZ rotation values
                        rot_str = format_rotation_string(rotation)
                        cv2.putText(frame, rot_str, (x + 5, y - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, 
                                  (255, 255, 0), 1, cv2.LINE_AA)
            
            # Add spine rotation info
            if "spine" in current_rotations:
                spine_str = f"Spine: {format_rotation_string(current_rotations['spine'])}"
                cv2.putText(frame, spine_str, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Add status info
            status_text = "Tracking: "
            if results.pose_landmarks:
                status_text += "Upper Body "
            if results.face_landmarks:
                status_text += "Face "
            
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Motion Capture Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if client_socket:
        client_socket.close()
    server_socket.close()
    print("Motion capture server stopped")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5555
    main(port)
'''

class WebcamMotionCapture:
    def __init__(self, armature_name="metarig"):
        self.armature_name = armature_name
        self.armature = None
        self.subprocess = None
        self.socket_client = None
        self.receive_thread = None
        self.is_capturing = False
        self.port = 5555
        
        # Smoothing parameters
        self.smoothing = 0.3
        self.prev_rotations = {}
        
        # Motion data buffer
        self.latest_motion_data = None
        self.data_lock = threading.Lock()
        
        # Store current bone rotations for display
        self.current_bone_rotations = {}
        
        # Arm correction quaternions for T-pose alignment
        self.arm_corrections = {
            'upper_arm.L': Quaternion((1.0, 0.0, 0.0), math.radians(0)),  # Adjust if needed
            'upper_arm.R': Quaternion((1.0, 0.0, 0.0), math.radians(0)),  # Adjust if needed
        }
        
        # Debug counter
        self.debug_counter = 0
        
        # Get armature
        if armature_name in bpy.data.objects:
            self.armature = bpy.data.objects[armature_name]
            if self.armature.type != 'ARMATURE':
                raise ValueError(f"{armature_name} is not an armature")
        else:
            raise ValueError(f"Armature {armature_name} not found")
    
    def start_subprocess(self):
        """Start the motion capture subprocess"""
        # Write subprocess code to temp file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        temp_file.write(MOCAP_SUBPROCESS_CODE)
        temp_file.close()
        
        # Start subprocess
        self.subprocess = subprocess.Popen(
            [sys.executable, temp_file.name, str(self.port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Give subprocess time to start server
        time.sleep(2)
        
        # Connect to subprocess
        self.socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket_client.connect(("localhost", self.port))
            self.socket_client.settimeout(0.1)
            print(f"Connected to motion capture server on port {self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to motion capture server: {e}")
            self.stop_subprocess()
            return False
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file.name)
            except:
                pass
    
    def receive_data(self):
        """Receive data from subprocess"""
        while self.is_capturing and self.socket_client:
            try:
                # Read message length
                length_data = self.socket_client.recv(8)
                if not length_data:
                    break
                
                message_length = int(length_data.decode())
                
                # Read message
                message_data = b""
                while len(message_data) < message_length:
                    chunk = self.socket_client.recv(min(4096, message_length - len(message_data)))
                    if not chunk:
                        break
                    message_data += chunk
                
                # Parse JSON data
                if message_data:
                    motion_data = json.loads(message_data.decode())
                    with self.data_lock:
                        self.latest_motion_data = motion_data
            
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_capturing:
                    print(f"Error receiving data: {e}")
                break
    
    def stop_subprocess(self):
        """Stop the motion capture subprocess"""
        # Send stop command
        if self.socket_client:
            try:
                self.socket_client.send(b"STOP")
            except:
                pass
            self.socket_client.close()
            self.socket_client = None
        
        # Terminate subprocess
        if self.subprocess:
            self.subprocess.terminate()
            try:
                self.subprocess.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.subprocess.kill()
            self.subprocess = None
    
    def smooth_rotation(self, bone_name: str, new_rotation: Dict[str, float]) -> Tuple[float, float, float]:
        """Apply smoothing to rotations"""
        if bone_name not in self.prev_rotations:
            self.prev_rotations[bone_name] = new_rotation.copy()
            return (new_rotation['x'], new_rotation['y'], new_rotation['z'])
        
        prev = self.prev_rotations[bone_name]
        smoothed = {}
        for axis in ['x', 'y', 'z']:
            smoothed[axis] = prev[axis] + (new_rotation[axis] - prev[axis]) * (1 - self.smoothing)
        
        self.prev_rotations[bone_name] = smoothed
        return (smoothed['x'], smoothed['y'], smoothed['z'])
    
    def apply_motion_to_bones(self):
        """Apply captured motion to Rigify bones (UPPER BODY ONLY) with proper corrections"""
        with self.data_lock:
            motion_data = self.latest_motion_data
            self.latest_motion_data = None
        
        if not self.armature or not motion_data:
            return
        
        # Ensure we're in pose mode
        if bpy.context.active_object != self.armature:
            bpy.context.view_layer.objects.active = self.armature
        if bpy.context.mode != 'POSE':
            bpy.ops.object.mode_set(mode='POSE')
        
        pose_bones = self.armature.pose.bones
        
        # Apply head rotation
        if 'head' in motion_data and 'spine.006' in pose_bones:
            rot = motion_data['head']
            smoothed = self.smooth_rotation('head', {
                'x': rot['x'] * 0.5,
                'y': rot['y'] * 0.5,
                'z': rot['z'] * 0.5
            })
            bone = pose_bones['spine.006']
            bone.rotation_mode = 'XYZ'
            bone.rotation_euler = Euler(smoothed, 'XYZ')
            self.current_bone_rotations['spine.006'] = smoothed
        
        # Apply spine rotation
        if 'spine' in motion_data:
            spine_rot = motion_data['spine']
            smoothed = self.smooth_rotation('spine', {
                'x': spine_rot['x'] * 0.3,
                'y': spine_rot['y'] * 0.3,
                'z': spine_rot['z'] * 0.3
            })
            # Distribute rotation across spine bones
            spine_bones = ['spine', 'spine.001', 'spine.002', 'spine.003']
            for i, bone_name in enumerate(spine_bones):
                if bone_name in pose_bones:
                    bone = pose_bones[bone_name]
                    bone.rotation_mode = 'XYZ'
                    factor = (i + 1) / len(spine_bones) * 0.5
                    euler_rot = (smoothed[0] * factor, smoothed[1] * factor, smoothed[2] * factor)
                    bone.rotation_euler = Euler(euler_rot, 'XYZ')
                    self.current_bone_rotations[bone_name] = euler_rot
        
        # Apply arm rotations with proper Rigify mapping and corrections
        if 'arms' in motion_data:
            arms = motion_data['arms']
            
            # Left arm
            if 'upper_arm.L' in pose_bones and 'upper_arm_L' in arms:
                arm_rot = arms['upper_arm_L']
                smoothed = self.smooth_rotation('upper_arm.L', arm_rot)
                bone = pose_bones['upper_arm.L']
                bone.rotation_mode = 'QUATERNION'
                
                # Create rotation quaternion from Euler angles
                euler_rot = Euler(smoothed, 'XYZ')
                rot_quat = euler_rot.to_quaternion()
                
                # Apply arm correction for T-pose alignment
                corrected_quat = self.arm_corrections['upper_arm.L'] @ rot_quat
                bone.rotation_quaternion = corrected_quat
                
                self.current_bone_rotations['upper_arm.L'] = smoothed
            
            if 'forearm.L' in pose_bones and 'forearm_L' in arms:
                arm_rot = arms['forearm_L']
                smoothed = self.smooth_rotation('forearm.L', arm_rot)
                bone = pose_bones['forearm.L']
                bone.rotation_mode = 'XYZ'
                bone.rotation_euler = Euler(smoothed, 'XYZ')
                self.current_bone_rotations['forearm.L'] = smoothed
            
            # Right arm
            if 'upper_arm.R' in pose_bones and 'upper_arm_R' in arms:
                arm_rot = arms['upper_arm_R']
                smoothed = self.smooth_rotation('upper_arm.R', arm_rot)
                bone = pose_bones['upper_arm.R']
                bone.rotation_mode = 'QUATERNION'
                
                # Create rotation quaternion from Euler angles
                euler_rot = Euler(smoothed, 'XYZ')
                rot_quat = euler_rot.to_quaternion()
                
                # Apply arm correction for T-pose alignment
                corrected_quat = self.arm_corrections['upper_arm.R'] @ rot_quat
                bone.rotation_quaternion = corrected_quat
                
                self.current_bone_rotations['upper_arm.R'] = smoothed
            
            if 'forearm.R' in pose_bones and 'forearm_R' in arms:
                arm_rot = arms['forearm_R']
                smoothed = self.smooth_rotation('forearm.R', arm_rot)
                bone = pose_bones['forearm.R']
                bone.rotation_mode = 'XYZ'
                bone.rotation_euler = Euler(smoothed, 'XYZ')
                self.current_bone_rotations['forearm.R'] = smoothed
        
        # Force update
        self.armature.update_tag()
        bpy.context.view_layer.update()
    
    def initialize_bones(self):
        """Initialize bones for motion capture (UPPER BODY ONLY)"""
        if not self.armature:
            return
        
        # List of bones we'll be controlling (upper body only)
        target_bones = [
            'spine', 'spine.001', 'spine.002', 'spine.003', 'spine.004', 'spine.005', 'spine.006',
            'upper_arm.L', 'forearm.L', 'hand.L',
            'upper_arm.R', 'forearm.R', 'hand.R'
        ]
        
        # Set rotation mode for all target bones
        initialized = []
        for bone_name in target_bones:
            if bone_name in self.armature.pose.bones:
                bone = self.armature.pose.bones[bone_name]
                # Use quaternion for upper arms to apply corrections
                if bone_name.startswith('upper_arm'):
                    bone.rotation_mode = 'QUATERNION'
                else:
                    bone.rotation_mode = 'XYZ'
                initialized.append(bone_name)
        
        print(f"Initialized {len(initialized)} upper body bones for motion capture: {initialized}")
    
    def start_capture(self):
        """Start motion capture"""
        if not self.is_capturing:
            # Initialize bones
            self.initialize_bones()
            
            # Find available port
            while self.port < 5600:
                try:
                    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    test_socket.bind(("localhost", self.port))
                    test_socket.close()
                    break
                except:
                    self.port += 1
            
            self.is_capturing = True
            
            # Start subprocess
            if not self.start_subprocess():
                self.is_capturing = False
                return False
            
            # Start receive thread
            self.receive_thread = threading.Thread(target=self.receive_data)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
            print("Motion capture started (Upper Body Only)")
            return True
        return False
    
    def stop_capture(self):
        """Stop motion capture"""
        self.is_capturing = False
        
        # Stop receive thread
        if self.receive_thread:
            self.receive_thread.join(timeout=2)
        
        # Stop subprocess
        self.stop_subprocess()
        
        print("Motion capture stopped")


# Global instance
mocap_instance = None


# Blender Operators
class MOCAP_OT_start_capture(bpy.types.Operator):
    """Start webcam motion capture"""
    bl_idname = "mocap.start_capture"
    bl_label = "Start Motion Capture"
    
    def execute(self, context):
        global mocap_instance
        
        # Create instance if needed
        if mocap_instance is None:
            try:
                mocap_instance = WebcamMotionCapture("metarig")
            except Exception as e:
                self.report({'ERROR'}, str(e))
                return {'CANCELLED'}
        
        # Start capture
        if mocap_instance.start_capture():
            # Start timer for updating bones
            context.window_manager.modal_handler_add(self)
            self._timer = context.window_manager.event_timer_add(0.033, window=context.window)
            return {'RUNNING_MODAL'}
        else:
            self.report({'ERROR'}, "Failed to start motion capture")
            return {'CANCELLED'}
    
    def modal(self, context, event):
        global mocap_instance
        
        if event.type == 'TIMER':
            if mocap_instance and mocap_instance.is_capturing:
                # Apply motion data
                mocap_instance.apply_motion_to_bones()
                
                # Force redraw of all 3D viewports
                for window in context.window_manager.windows:
                    for area in window.screen.areas:
                        if area.type == 'VIEW_3D':
                            area.tag_redraw()
                
                # Also update depsgraph
                context.view_layer.depsgraph.update()
            else:
                return self.cancel(context)
        
        if event.type == 'ESC':
            return self.cancel(context)
        
        return {'PASS_THROUGH'}
    
    def cancel(self, context):
        global mocap_instance
        if mocap_instance:
            mocap_instance.stop_capture()
        context.window_manager.event_timer_remove(self._timer)
        return {'CANCELLED'}


class MOCAP_OT_stop_capture(bpy.types.Operator):
    """Stop webcam motion capture"""
    bl_idname = "mocap.stop_capture"
    bl_label = "Stop Motion Capture"
    
    def execute(self, context):
        global mocap_instance
        if mocap_instance:
            mocap_instance.stop_capture()
        return {'FINISHED'}


class MOCAP_OT_reset_pose(bpy.types.Operator):
    """Reset armature to rest pose"""
    bl_idname = "mocap.reset_pose"
    bl_label = "Reset Pose"
    
    def execute(self, context):
        global mocap_instance
        if mocap_instance and mocap_instance.armature:
            # Set armature as active and switch to pose mode
            context.view_layer.objects.active = mocap_instance.armature
            bpy.ops.object.mode_set(mode='POSE')
            
            # Reset all bone transformations
            for bone in mocap_instance.armature.pose.bones:
                # Reset transformations
                bone.rotation_euler = Euler((0, 0, 0), 'XYZ')
                bone.rotation_quaternion = Quaternion((1, 0, 0, 0))
                bone.location = Vector((0, 0, 0))
                bone.scale = Vector((1, 1, 1))
            
            # Clear smoothing cache and rotation display
            if mocap_instance:
                mocap_instance.prev_rotations.clear()
                mocap_instance.current_bone_rotations.clear()
            
            # Update
            mocap_instance.armature.update_tag()
            context.view_layer.update()
            
            self.report({'INFO'}, "Pose reset successfully")
        return {'FINISHED'}


# Properties
class MocapProperties(bpy.types.PropertyGroup):
    smoothing: bpy.props.FloatProperty(
        name="Smoothing",
        description="Motion smoothing factor (0 = no smoothing, 1 = max smoothing)",
        min=0.0,
        max=1.0,
        default=0.3,
        update=lambda self, context: setattr(mocap_instance, 'smoothing', self.smoothing) if mocap_instance else None
    )


# UI Panel
class MOCAP_PT_panel(bpy.types.Panel):
    """Motion Capture Panel"""
    bl_label = "Webcam Motion Capture"
    bl_idname = "MOCAP_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Motion Capture"
    
    def draw(self, context):
        layout = self.layout
        mocap_props = context.scene.mocap_props
        
        col = layout.column(align=True)
        
        # Status
        if mocap_instance and mocap_instance.is_capturing:
            col.label(text="Status: Capturing", icon='REC')
            col.operator("mocap.stop_capture", icon='PAUSE')
        else:
            col.label(text="Status: Stopped", icon='NONE')
            col.operator("mocap.start_capture", icon='PLAY')
        
        col.separator()
        
        # Controls
        col.operator("mocap.reset_pose", icon='LOOP_BACK')
        
        col.separator()
        
        # Settings
        col.prop(mocap_props, "smoothing", slider=True)
        
        col.separator()
        
        # Bone Rotation Display
        if mocap_instance and mocap_instance.current_bone_rotations:
            box = col.box()
            box.label(text="Current Bone Rotations:", icon='BONE_DATA')
            
            for bone_name, rotation in mocap_instance.current_bone_rotations.items():
                sub_box = box.box()
                sub_box.label(text=f"{bone_name}:", icon='BONE_DATA')
                
                # Convert radians to degrees for display
                x_deg = math.degrees(rotation[0])
                y_deg = math.degrees(rotation[1])
                z_deg = math.degrees(rotation[2])
                
                row = sub_box.row(align=True)
                row.label(text=f"X: {x_deg:+.1f}°")
                row.label(text=f"Y: {y_deg:+.1f}°")
                row.label(text=f"Z: {z_deg:+.1f}°")
        
        # Info
        col.separator()
        box = col.box()
        box.label(text="Controls:", icon='INFO')
        box.label(text="• ESC to stop capture")
        box.label(text="• Q in preview to quit")
        box.label(text="• Upper body tracking only")
        
        # Connection info
        if mocap_instance and mocap_instance.is_capturing:
            box.separator()
            box.label(text=f"Port: {mocap_instance.port}")


# Registration
classes = [
    MocapProperties,
    MOCAP_OT_start_capture,
    MOCAP_OT_stop_capture,
    MOCAP_OT_reset_pose,
    MOCAP_PT_panel,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.mocap_props = bpy.props.PointerProperty(type=MocapProperties)
    print("Webcam Motion Capture addon registered (Upper Body Only)")

def unregister():
    global mocap_instance
    if mocap_instance:
        mocap_instance.stop_capture()
        mocap_instance = None
    
    del bpy.types.Scene.mocap_props
    
    for cls in classes:
        bpy.utils.unregister_class(cls)
    print("Webcam Motion Capture addon unregistered")

if __name__ == "__main__":
    # Unregister if already registered
    try:
        unregister()
    except:
        pass
    
    register()
