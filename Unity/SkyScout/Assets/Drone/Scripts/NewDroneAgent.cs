using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine.InputSystem;

public class NewDroneAgent : Agent
{
    [SerializeField] private Renderer droneRenderer;
    [SerializeField] private float environmentSize = 10f;
    [SerializeField] private InputActionAsset inputActions;

    private Rigidbody rb;
    private InputAction thrustAction;
    private InputAction moveAction;
    private Bounds bounds;
    private Vector3 initialPosition;
    private Quaternion initialRotation;

    // Drone physics parameters
    [Header("Drone Physics")]
    [SerializeField] private float maxThrust = 100f;
    [SerializeField] private float maxAngularSpeed = 100f;
    [SerializeField] private float maxTiltAngle = 30f;
    [SerializeField] private float maxRotationRate = 30f; // Maximum degrees per second of rotation change
    [SerializeField] private float rotationSmoothing = 0.1f; // Lower = smoother rotation

    // Training parameters
    [Header("Training Parameters")]
    [SerializeField] private float targetHeight = 2f;
    [SerializeField] private float maxHorizontalDeviation = 1f;

    private Vector3 currentAngularVelocity;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
        initialPosition = transform.localPosition;
        initialRotation = transform.localRotation;
        currentAngularVelocity = Vector3.zero;
        
        // Create bounds relative to this environment
        bounds = new Bounds(Vector3.zero, Vector3.one * environmentSize);

        // Set up input actions for manual control
        if (inputActions != null)
        {
            var actionMap = inputActions.FindActionMap("Drone");
            thrustAction = actionMap.FindAction("Thrust");
            moveAction = actionMap.FindAction("Move");
            
            thrustAction.Enable();
            moveAction.Enable();
        }
    }

    private void OnDestroy()
    {
        thrustAction?.Disable();
        moveAction?.Disable();
    }

    public override void OnEpisodeBegin()
    {
        // Reset position and rotation
        transform.localPosition = initialPosition;
        transform.localRotation = initialRotation;
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        currentAngularVelocity = Vector3.zero;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Drone's rotation (3 values)
        sensor.AddObservation(transform.localRotation.eulerAngles / 360f);
        
        // Drone's velocity (3 values)
        sensor.AddObservation(rb.linearVelocity / maxThrust);
        
        // Drone's angular velocity (3 values)
        sensor.AddObservation(rb.angularVelocity / maxAngularSpeed);
        
        // Position relative to start position (3 values)
        Vector3 localPosition = transform.localPosition;
        sensor.AddObservation(localPosition / environmentSize);
    }

    public override void OnActionReceived(ActionBuffers actionsOut)
    {
        // Get actions
        float thrust = actionsOut.ContinuousActions[0];
        float pitch = actionsOut.ContinuousActions[1];
        float roll = actionsOut.ContinuousActions[2];
        float yaw = actionsOut.ContinuousActions[3];

        // Apply thrust
        rb.AddForce(transform.up * thrust * maxThrust);

        // Calculate target angular velocity with rate limiting
        Vector3 targetAngularVelocity = new Vector3(pitch, yaw, roll) * maxAngularSpeed;
        
        // Rate limit the angular velocity changes
        Vector3 maxDelta = Vector3.one * (maxRotationRate * Time.fixedDeltaTime);
        currentAngularVelocity = Vector3.MoveTowards(
            currentAngularVelocity,
            targetAngularVelocity,
            maxDelta.magnitude
        );

        // Apply smoothed rotation
        rb.angularVelocity = Vector3.Lerp(
            rb.angularVelocity,
            currentAngularVelocity,
            rotationSmoothing
        );

        // Calculate rewards
        if (bounds.Contains(transform.localPosition))
        {
            // Reward for maintaining stability (reduced penalty for angular velocity)
            float stabilityReward = 1f - (rb.angularVelocity.magnitude / (maxAngularSpeed * 0.5f));
            AddReward(stabilityReward * 0.2f);

            /*
            // Penalty for distance to goal
            float distanceToGoal = Vector3.Distance(transform.localPosition, goal.localPosition);
            AddReward(-distanceToGoal * 0.1f);
            */

            // Penalty for horizontal distance to start position
            float horizontalDistance = new Vector3(transform.localPosition.x, 0f, transform.localPosition.z).magnitude;
            if (horizontalDistance > maxHorizontalDeviation)
            {
                AddReward(-horizontalDistance * 0.2f);
            }

            // Height reward - exponential reward for getting closer to target height
            float heightDiff = Mathf.Abs(transform.localPosition.y - targetHeight);
            float heightReward = Mathf.Exp(-heightDiff);
            AddReward(heightReward * 0.3f);

            // Bonus reward for maintaining target height
            if (Mathf.Abs(transform.localPosition.y - targetHeight) < 0.2f)
            {
                AddReward(0.1f);
            }
        }
        else
        {
            // Penalty for going out of bounds
            AddReward(-1f);
            EndEpisode();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        if (inputActions == null) return;

        var continuousActionsOut = actionsOut.ContinuousActions;
        
        // Get thrust input (0 to 1)
        float thrust = thrustAction.ReadValue<float>();
        continuousActionsOut[0] = thrust;
        
        // Get movement input for pitch and roll
        Vector2 moveInput = moveAction.ReadValue<Vector2>();
        continuousActionsOut[1] = moveInput.y;  // Pitch
        continuousActionsOut[2] = moveInput.x;  // Roll
        continuousActionsOut[3] = 0f;          // Yaw (could add Q/E keys for this)
    }

    private void OnCollisionEnter(Collision collision)
    {
        AddReward(-0.5f);
        if (droneRenderer != null)
        {
            droneRenderer.material.color = Color.red;
        }
    }

    private void OnCollisionExit(Collision collision)
    {
        if (droneRenderer != null)
        {
            droneRenderer.material.color = Color.blue;
        }
    }
}
