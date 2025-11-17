# **SkyScout — Autonomous Drone Navigation with Unity ML-Agents**

SkyScout is a reinforcement-learning project built in Unity that trains a quadcopter-style drone to autonomously navigate toward dynamic goals in a 3D environment. Using Unity ML-Agents, a custom reward structure, and realistic physics, the drone learns stable flight, height control, and target-seeking behaviors. The project includes training, inference, and visualization components, with support for ONNX models and Unity Sentis for efficient deployment.

---

## **Features**

### 🚁 **Autonomous Drone Agent**
- Uses ML-Agents PPO to learn stable flight behavior.  
- Observes relative position, velocity, rotation, and directional cues.  
- Accepts continuous actions for pitch, roll, yaw, and throttle.

### 🎯 **Dynamic Goal Generation**
- Target spawns at randomized distances, angles, and heights every episode.  
- Reward shaping encourages vertical alignment, horizontal approach, and efficient control use.

### 🧠 **Custom Reward Structure**
Implemented entirely from scratch to guide learning, including:
- Height alignment rewards  
- Horizontal distance improvement rewards  
- Throttle incentives when below target  
- Collision penalties & step penalties  
- Sparse success reward upon reaching the goal  

### 🧪 **Training Pipeline**
- Train in Unity using ML-Agents CLI.
- Supports curriculum-style scene variations.
- Outputs a `.nn` or `.onnx` policy for later inference.

### ⚙️ **Inference & Deployment**
- Inference supported through:
  - **Barracuda** (for older ML-Agents versions)
  - **Unity Sentis** (for ONNX runtime)
- Pretrained models can be swapped in the `Behavior Parameters` component.

---

## **Tech Stack**
- **Unity** (Physics & simulation)  
- **C#** (Agent behavior, reward logic, environment setup)  
- **Unity ML-Agents** (Reinforcement learning framework)  
- **PyTorch** (Training backend via ML-Agents)  
- **ONNX + Unity Sentis** (Model deployment)  
- **TensorBoard** (Training monitoring)

---

## **Project Structure**

```text
SkyScout/
├── Assets/
│   ├── Agents/            # DroneAgent scripts & prefabs
│   ├── Environments/      # Training scenes and goal spawners
│   ├── Models/            # Trained .nn/.onnx models
│   └── Scripts/           # Utilities, camera follow, config helpers
├── config/                # ML-Agents training configuration (YAML)
└── README.md


