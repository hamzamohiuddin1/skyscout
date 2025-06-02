using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

namespace MBaske
{
    public class DroneAgentTerrain : Agent
    {
        [SerializeField]
        private Multicopter multicopter;

        [SerializeField] private Transform _goal;
        [SerializeField] private Renderer _renderer;
        [SerializeField] private float environmentSize = 10f;

        public MeshCollider terrainMeshCollider;// Assign this in the Inspector

        private Vector3 previousPosition;

        private Bounds bounds;
        private Resetter resetter;

        private float lastHorizontalDistanceToGoal;
        private int noProgressSteps;

        public override void Initialize()
        {
            multicopter.Initialize();
            bounds = new Bounds(Vector3.zero, Vector3.one * environmentSize);
            resetter = new Resetter(transform);
        }

        public override void OnEpisodeBegin()
        {
            resetter.Reset();
            SpawnObjects();

            lastHorizontalDistanceToGoal = Vector3.Distance(
                new Vector3(transform.localPosition.x, 0, transform.localPosition.z),
                new Vector3(_goal.localPosition.x, 0, _goal.localPosition.z));
            noProgressSteps = 0;
        }

        
        public float minX = -20f, maxX = 20f;
        public float minZ = -20f, maxZ = 20f;

        private void SpawnObjects()
        {
            // Reset drone
            transform.localPosition = new Vector3(20f, 4f, 0f);
            transform.localRotation = Quaternion.identity;

            // Random XZ position
            float randX = Random.Range(minX, maxX);
            float randZ = Random.Range(minZ, maxZ);

            Vector3 rayOrigin = new Vector3(randX, 100f, randZ);
            if (Physics.Raycast(rayOrigin, Vector3.down, out RaycastHit hitInfo, 200f))
            {
                if (hitInfo.collider.gameObject.CompareTag("Terrain"))
                {
                    float slope = Vector3.Angle(hitInfo.normal, Vector3.up);
                    if (slope < 45f) // Acceptable slope
                    {
                        _goal.localPosition = hitInfo.point + hitInfo.normal * 2.0f;
                        return;
                    }
                }
            }

            // Fallback position if no valid hit
            _goal.localPosition = new Vector3(randX, 16f, randZ);
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

            float terrainHeight = 0f;
            float altitude = 0f;
            if (Physics.Raycast(dronePos, Vector3.down, out RaycastHit hitInfo, 100f))
            {
                if (hitInfo.collider.gameObject.CompareTag("Terrain")) // Optional
                {
                    terrainHeight = hitInfo.point.y;
                    altitude = dronePos.y - terrainHeight;
                }
            }

            float maxAltitude = environmentSize; // or use 50f, etc.
            float normalizedAltitude = Mathf.Clamp01(altitude / maxAltitude);
            sensor.AddObservation(normalizedAltitude);
        }

        // Add this field at the top of the class
        private int stepCounter = 0;
        private const int logEvery = 100;  // <-- print every 100 Agent decisions

        public override void OnActionReceived(ActionBuffers actionBuffers)
        {
            float[] actions = actionBuffers.ContinuousActions.Array;
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

            float tiltAlignment = Vector3.Dot(upVector, dirToGoal);
            tiltAlignment = Mathf.Max(tiltAlignment, 0f);
            if (uprightReward > 0.5f)
            {
                tiltAlignment *= uprightReward;
            }
            else
            {
                tiltAlignment = 0f;
            }

            float velocityTowardsGoal = Vector3.Dot(velocity.normalized, dirToGoal);

            float reward = 0f;
            // Distance penalties (scale down)
            reward += -0.1f * totalDistance;

            // Positive incentives
            reward += 0.2f * velocityTowardsGoal;

            float previousDistance = Vector3.Distance(previousPosition, goalPos);
            float currentDistance = Vector3.Distance(pos, goalPos);

            float distanceDelta = previousDistance - currentDistance;
            reward += distanceDelta * 0.5f;

            previousPosition = pos;
            /*
            float droneAltitude = 0f;
            if (Physics.Raycast(pos, Vector3.down, out RaycastHit hitInfo, 100f))
            {
                // Check if hit terrain
                if (hitInfo.collider.gameObject.CompareTag("Terrain")) // Optional tag check
                {
                    float groundY = hitInfo.point.y;
                    droneAltitude = pos.y - groundY;
                }
            }

            float minSafeAltitude = 5.0f; // Minimum safe altitude above terrain
            float groundPenalty = 0f;

            if (droneAltitude < minSafeAltitude)
            {
                groundPenalty = -0.2f * (minSafeAltitude - droneAltitude);
                reward += groundPenalty;
            }
            */

            // === Obstacle avoidance + climb encouragement ===
            bool frontBlocked = false;
            float forwardObstacleDist = 0f;

            Vector3 directionToGoal = (goalPos - pos).normalized;
            if (Physics.Raycast(pos, directionToGoal, out RaycastHit wallHit, 3f))
            {
                if (wallHit.collider.CompareTag("Terrain"))
                {
                    frontBlocked = true;
                    forwardObstacleDist = wallHit.distance;
                }
            }

            float climbReward = 0f;
            if (frontBlocked && distanceDelta < 0.01f)
            {
                float upwardVelocity = Mathf.Max(0f, velocity.y);
                climbReward = 0.1f * upwardVelocity;
                reward += climbReward;
            }
            AddReward(reward);
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
                    $"UpDot:{uprightReward:F2} | ClimbReward:{climbReward:F2} | " +
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
                AddReward(3.0f); // Increased reward for success
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
