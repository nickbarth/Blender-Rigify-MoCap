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

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org>
