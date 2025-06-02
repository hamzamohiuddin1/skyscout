using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Linq;

namespace MBaske
{
    public class DroneAgentTerrain : Agent
    {

        // [Header("Heuristic Obstacle Avoidance")]
        // [SerializeField]
        // private float avoidRayLength = 5f;
        // [SerializeField]
        // private float avoidYawTurnAngle = 20f;
        // [SerializeField]
        // private string[] detectableTags = new[] { "Terrain", "Wall" };


        [SerializeField]
        private Multicopter multicopter;

        [SerializeField] private Transform _goal;
        [SerializeField] private Renderer _renderer;
        [SerializeField] private float environmentSize = 50f;

        private Vector3 previousPosition;

        private Bounds bounds;
        private Resetter resetter;

        private float lastHorizontalDistanceToGoal;
        private int noProgressSteps;

        private Rigidbody rb;

        public override void Initialize()
        {
            rb = GetComponent<Rigidbody>();
            multicopter.Initialize();
            bounds = new Bounds(Vector3.zero, Vector3.one * environmentSize);
            resetter = new Resetter(transform);
        }


        // private void HeuristicAvoidance(ref float[] thrusts)
        // {
        //     // 1) Ray origins & directions
        //     Vector3 origin = multicopter.Frame.position;
        //     Vector3 forward = multicopter.Frame.forward;
        //     Vector3 back = -forward;
        //     Vector3 upDir = multicopter.Frame.up;
        //     Vector3 downDir = -upDir;

        //     // 2) Cast each primary direction with the same length
        //     bool hitFront = Physics.Raycast(origin, forward, out RaycastHit frontHit, avoidRayLength);
        //     bool hitBack = Physics.Raycast(origin, back, out RaycastHit backHit, avoidRayLength);
        //     bool hitUp = Physics.Raycast(origin, upDir, out RaycastHit upHit, avoidRayLength);
        //     bool hitDown = Physics.Raycast(origin, downDir, out RaycastHit downHit, avoidRayLength);

        //     // Helper to check tags quickly
        //     bool IsBad(RaycastHit h)
        //         => h.collider != null && detectableTags.Contains(h.collider.tag);

        //     // 3) Vertical avoidance: if ground (down) is too close, boost upward
        //     if (hitDown && IsBad(downHit))
        //     {
        //         // Add a small upward thrust tweak on all four motors
        //         float liftKick = 0.1f;
        //         for (int i = 0; i < thrusts.Length; i++)
        //             thrusts[i] = Mathf.Clamp(thrusts[i] + liftKick, 0f, 1f);

        //         // Skip other avoidance this frame
        //         return;
        //     }

        //     // 4) Ceiling avoidance: if an overhang is too close above, push down (reduce thrust)
        //     if (hitUp && IsBad(upHit))
        //     {
        //         float dropKick = 0.1f;
        //         for (int i = 0; i < thrusts.Length; i++)
        //             thrusts[i] = Mathf.Clamp(thrusts[i] - dropKick, 0f, 1f);
        //         // Skip horizontal avoidance this frame
        //         return;
        //     }

        //     // 5) Horizontal avoidance:
        //     if (hitFront && IsBad(frontHit))
        //     {
        //         // yaw-right: decrease left motors (0 & 2), increase right (1 & 3)
        //         float yawKick = avoidYawTurnAngle / 100f;
        //         thrusts[0] = Mathf.Clamp(thrusts[0] - yawKick, 0f, 1f);
        //         thrusts[2] = Mathf.Clamp(thrusts[2] - yawKick, 0f, 1f);
        //         thrusts[1] = Mathf.Clamp(thrusts[1] + yawKick, 0f, 1f);
        //         thrusts[3] = Mathf.Clamp(thrusts[3] + yawKick, 0f, 1f);
        //         return;
        //     }
        //     if (hitBack && IsBad(backHit))
        //     {
        //         // yaw-left: decrease right motors, increase left
        //         float yawKick = avoidYawTurnAngle / 100f;
        //         thrusts[1] = Mathf.Clamp(thrusts[1] - yawKick, 0f, 1f);
        //         thrusts[3] = Mathf.Clamp(thrusts[3] - yawKick, 0f, 1f);
        //         thrusts[0] = Mathf.Clamp(thrusts[0] + yawKick, 0f, 1f);
        //         thrusts[2] = Mathf.Clamp(thrusts[2] + yawKick, 0f, 1f);
        //         return;
        //     }
        //     // If no hits, do nothing—keep the network’s original thrusts
        // }


        public override void OnEpisodeBegin()
        {
            resetter.Reset();
            SpawnObjects();

            lastHorizontalDistanceToGoal = Vector3.Distance(
                new Vector3(transform.localPosition.x, 0, transform.localPosition.z),
                new Vector3(_goal.localPosition.x, 0, _goal.localPosition.z));
            noProgressSteps = 0;
        }

        private void SpawnObjects()
        {

            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;

            // Set drone position to (0, 2.34, -17)
            transform.localPosition = new Vector3(20f, 4.1f, 0.0f);
            transform.localRotation = Quaternion.identity;

            // Find a random position on the terrain
            bool foundValidPosition = false;
            int maxAttempts = 10;
            int attempts = 0;
            Vector3 goalPosition = Vector3.zero;

            while (!foundValidPosition && attempts < maxAttempts)
            {
                // Random position within bounds
                float randomX = Random.Range(-environmentSize / 2, environmentSize / 2);
                float randomZ = Random.Range(-environmentSize / 2, environmentSize / 2);

                // Start raycast from high above the terrain
                Vector3 rayStart = new Vector3(randomX, 100f, randomZ);
                RaycastHit hit;

                // Cast ray downward to find terrain
                if (Physics.Raycast(rayStart, Vector3.down, out hit, 200f, LayerMask.GetMask("Default")))
                {
                    if (hit.collider.CompareTag("Terrain"))
                    {
                        // Place goal slightly above the terrain surface
                        goalPosition = hit.point + Vector3.up * 1.0f; // 1 unit above terrain
                        foundValidPosition = true;
                    }
                }
                attempts++;
            }

            // If we couldn't find a valid position, use a default position
            if (!foundValidPosition)
            {
                Debug.LogWarning("Could not find valid terrain position for goal, using default position");
                goalPosition = new Vector3(13.0f, 7.0f, 4.39f);
            }

            _goal.localPosition = goalPosition;
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            sensor.AddObservation(multicopter.Inclination);
            sensor.AddObservation(Normalization.Sigmoid(
                multicopter.LocalizeVector(multicopter.Rigidbody.linearVelocity), 0.25f));
            sensor.AddObservation(Normalization.Sigmoid(
                multicopter.LocalizeVector(multicopter.Rigidbody.angularVelocity)));

            foreach (var rotor in multicopter.Rotors)
            {
                sensor.AddObservation(rotor.CurrentThrust);
            }

            Vector3 goalPos = _goal.position;
            Vector3 dronePos = multicopter.Frame.position;

            sensor.AddObservation(goalPos.x / environmentSize);
            sensor.AddObservation(goalPos.y / environmentSize);
            sensor.AddObservation(goalPos.z / environmentSize);
            sensor.AddObservation(dronePos.x / environmentSize);
            sensor.AddObservation(dronePos.y / environmentSize);
            sensor.AddObservation(dronePos.z / environmentSize);

            // --- NEW OBSERVATIONS ---

            // Distance to goal (scalar, normalized by environment size)
            float distanceToGoal = Vector3.Distance(dronePos, goalPos) / environmentSize;
            sensor.AddObservation(distanceToGoal);

            // Direction to goal in local drone coordinates (normalized vector)
            Vector3 directionToGoalWorld = (goalPos - dronePos).normalized;
            Vector3 directionToGoalLocal = multicopter.Frame.InverseTransformDirection(directionToGoalWorld);
            sensor.AddObservation(directionToGoalLocal);
        }

        // Add this field at the top of the class
        private int stepCounter = 0;
        private const int logEvery = 100;  // <-- print every 100 Agent decisions

        public override void OnActionReceived(ActionBuffers actionBuffers)
        {

            if (!bounds.Contains(multicopter.Frame.position)){
                AddReward(-2.0f); // Penalty for going out of bounds
                EndEpisode();
                return; // Important to stop further processing for this step
            }

            float[] actions = actionBuffers.ContinuousActions.Array;
            // HeuristicAvoidance(ref actions);
            multicopter.UpdateThrust(actions);

            Vector3 pos = multicopter.Frame.position;
            Vector3 goalPos = _goal.position;

            float totalDistance = Vector3.Distance(pos, goalPos);

            Vector3 dirToGoal = (goalPos - pos).normalized;
            Vector3 upVector = multicopter.Rigidbody.rotation * Vector3.up;

            Vector3 velocity = multicopter.Rigidbody.linearVelocity;

            // Enhanced upright reward/penalty
            float uprightReward = Vector3.Dot(upVector, Vector3.up);
            // Add strong penalty for being upside down (uprightReward < 0)
            if (uprightReward < 0)
            {
                AddReward(-0.5f); // Immediate penalty for being upside down
            }
            // Normal upright reward for being right-side up
            else
            {
                AddReward(0.1f * uprightReward);
            }

            

            // float tiltAlignment = Vector3.Dot(upVector, dirToGoal);
            // tiltAlignment = Mathf.Max(tiltAlignment, 0f);
            // if (uprightReward > 0.5f)
            // {
            //     tiltAlignment *= uprightReward;
            // }
            // else
            // {
            //     tiltAlignment = 0f;
            // }

            float velocityTowardsGoal = Vector3.Dot(velocity.normalized, dirToGoal);

            float reward = 0f;
            // Distance penalties (scale down)
            reward += -0.05f * totalDistance;

            // Positive incentives
            // reward += 0.3f * tiltAlignment;
            reward += 0.2f * velocityTowardsGoal;

            float previousDistance = Vector3.Distance(previousPosition, goalPos);
            float currentDistance = Vector3.Distance(pos, goalPos);

            float distanceDelta = previousDistance - currentDistance;
            reward += distanceDelta * 0.5f;
            AddReward(reward);

            previousPosition = pos;

            // Time penalty
            AddReward(-0.005f);

            /* ------------ LOGGING ------------ */
            if (stepCounter % logEvery == 0)
            {
                Debug.Log(
                    $"Step {StepCount} | " +
                    $"Dist:{totalDistance:F2} | " +
                    $"DronePos:{pos:F2} | " +
                    $"GoalPos:{goalPos:F2} | " +
                    $"Dist:{totalDistance:F2} | " +
                    $"Δd:{distanceDelta:F3} | " +
                    $"Vel:{velocity.magnitude:F2} " +
                    $"(→Goal:{velocityTowardsGoal:F2}) | " +
                    $"UpDot:{uprightReward:F2} | " +
                    $"RewardThisStep:{reward:F3}"
                );
            }
            stepCounter++;
            /* ------------ END LOG ------------ */
        }

        private void OnTriggerEnter(Collider other)
        {
            if (other.gameObject.CompareTag("Goal"))
            {
                AddReward(10.0f); // Increased reward for success
                EndEpisode();
            }
        }

        private void OnCollisionEnter(Collision collision)
        {
            if (collision.gameObject.CompareTag("Terrain"))
            {
                AddReward(-500f);
                EndEpisode();
            }
        }

        private void OnCollisionStay(Collision collision)
        {
            if (collision.gameObject.CompareTag("Wall"))
            {
                AddReward(-0.1f * Time.fixedDeltaTime);
            }
        }

        private void OnCollisionExit(Collision collision)
        {
            if (collision.gameObject.CompareTag("Wall"))
            {
                if (_renderer != null)
                {
                    _renderer.material.color = Color.blue;
                }
            }
        }
    }
}
