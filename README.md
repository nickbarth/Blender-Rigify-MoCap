# Blender-Rigify-MoCap
### v0.1

Blender Webcam Motion Capture for Rigify Meta Rig - Upper Body Only
Captures motion from webcam using MediaPipe and applies to upper body Rigify bones

<img width="793" height="526" alt="image" src="https://github.com/user-attachments/assets/3f24c235-254c-464d-9fdb-138235f4bcee" />



    # Instructions
    
    SETUP:
    1. Install required packages in Blender's Python:
       - opencv-python: pip install opencv-python
       - mediapipe: pip install mediapipe
       - numpy: pip install numpy
    2. Add a Rigify Basic Human armature to your scene
    3. Run this script in Blender's Script Editor
    
    USAGE:
    1. Find "Motion Capture" panel in 3D Viewport sidebar (press N)
    2. Click "Reset Pose" to return to rest position
    3. Click "Start Motion Capture" to begin capturing
    4. Stand in front of camera with upper body visible
    5. Arms should now move correctly:
       - Up = arms go up
       - Forward = arms go forward
       - Sideways = arms go to the sides
    6. Press ESC or click "Stop Motion Capture" to stop


