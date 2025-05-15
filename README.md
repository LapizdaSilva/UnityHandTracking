# UnityHandTracking
Mediapipe Python integrated with Unity project made for an academic fair

# Unity Hand Tracking + MediaPipe + Physics Interaction

Real time hand tracking with **MediaPipe** (Python) + **Unity** integrated by socket TCP, **allowing to phisically interact with 3D objects**.

---

## 🧠 Funcionallities

- 3D tracking of up to **Two Hands** 
- Interaction with phisical objects by colliders and rigibodies
  
---

## Requirements

### Python
- `mediapipe`
- `opencv-python`
- `numpy`

### Unity
- Unity 2021+ (or higher)
- A standard 3D project
- Scripts `HandTrackingServer.cs`

---

## How to use

### 1. Clone the repository

```
git clone https://github.com/seu-usuario/unity-handtracking.git
```
### 2. Run the python script

```
cd python
python HT_Unity.py
```
This script uses the camera to track your hand and then send the data to Unity via socket.

### 3. Open the Unity Project

Download the unity project and load on unity

### 4. Press play and Have fun

---

# Credits

## Mediapipe - Hand Tracking
## Unity - Engine
## Made by **Gabriel Vinicius dos Santos**
