using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

namespace MBaske
{
    public class DroneAgent : Agent
    {
        [SerializeField]
        private Multicopter multicopter;

        [SerializeField] private Transform _goal;
        [SerializeField] private Renderer _renderer;
        [SerializeField] private float environmentSize = 10f;

        private Vector3 previousPosition;
        private float previousHeight = 0.0f;
        private float previousHorizontalDist;

        private Bounds bounds;
        private Resetter resetter;

        private float lastHorizontalDistanceToGoal;
        private int noProgressSteps;

        private string trainingStage = "takeoff"; // 'takeoff' | 'findTarget' | 'descend'

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

        private void SpawnObjects()
        {
            transform.localPosition = new Vector3(0f, 0.5f, 0f);
            transform.localRotation = Quaternion.identity;

            float randomAngle = Random.Range(0f, 360f);
            Vector3 randomDirection = Quaternion.Euler(0f, randomAngle, 0f) * Vector3.forward;
            float randomDistance = Random.Range(1f, 3f);
            Vector3 goalPosition = transform.localPosition + randomDirection * randomDistance;
            _goal.localPosition = new Vector3(goalPosition.x, 2f, goalPosition.z);
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
            float totalReward = 0f;
            float[] actions = actionBuffers.ContinuousActions.Array;
            multicopter.UpdateThrust(actions);

            Vector3 pos = multicopter.Frame.position;
            Vector3 goalPos = _goal.position;
            Rigidbody rb = multicopter.Rigidbody;

            Vector2 posXZ = new Vector2(pos.x, pos.z);
            Vector2 goalXZ = new Vector2(goalPos.x, goalPos.z);

            Vector3 upVector = rb.rotation * Vector3.up;
            float angularVelocityPenalty = -0.1f * rb.angularVelocity.magnitude;

            float horizontalDistance = Vector2.Distance(posXZ, goalXZ);
            float horizontalProgress = previousHorizontalDist - horizontalDistance;

            switch (trainingStage)
            {
                case "takeoff":
                    if (pos.y >= 3.9f)
                    {
                        totalReward += 2.0f;
                        trainingStage = "findTarget";

                        previousHorizontalDist = Vector2.Distance(new Vector2(pos.x, pos.z), new Vector2(goalPos.x, goalPos.z));
                    }
                    else
                    {
                        // Encourage upward motion
                        float heightDelta = pos.y - previousHeight;
                        previousHeight = pos.y;

                        float upwardVelocity = Mathf.Clamp(rb.linearVelocity.y, 0f, 1f);
                        float upwardVelocityReward = 0.3f * upwardVelocity;

                        // Uprightness: only reward if reasonably upright
                        float uprightness = Vector3.Dot(rb.rotation * Vector3.up, Vector3.up);
                        float uprightReward = Mathf.Clamp01(uprightness);

                        // Small constant reward for climbing, penalize stagnation
                        float heightProgressReward = Mathf.Max(0f, heightDelta * 2f);

                        float timePenalty = -0.002f;

                        totalReward += heightProgressReward + upwardVelocityReward + 0.1f * uprightReward + timePenalty;

                        // Fail-safe (but don't kill reward too hard)
                        if (StepCount > 400 && pos.y < 0.5f)
                        {
                            totalReward -= 0.5f;
                            EndEpisode();
                        }
                    }
                    break;

                case "findTarget":
                    bool isAboveTargetXZ = horizontalDistance < 0.5f;

                    if (isAboveTargetXZ)
                    {
                        trainingStage = "descend";
                        totalReward += 3.0f;
                    }
                    else
                    {
                        float lateralVelocity = new Vector2(rb.linearVelocity.x, rb.linearVelocity.z).magnitude;
                        previousHorizontalDist = horizontalDistance;

                        float altitudePenalty = Mathf.Abs(pos.y - 4.0f) > 0.2f ? -0.2f : 0.1f;
                        float uprightness = Vector3.Dot(upVector, Vector3.up);

                        totalReward += (0.5f * horizontalProgress) + (0.2f * lateralVelocity) + altitudePenalty + angularVelocityPenalty;
                        totalReward += uprightness > 0.9f ? 0.05f : -0.05f;
                        totalReward -= 0.01f; // time penalty
                    }
                    break;

                case "descend":
                    Vector3 droneToGoal = goalPos - pos;
                    float distToGoal = droneToGoal.magnitude;

                    // Penalty for moving sideways
                    Vector2 horizontalVelocity = new Vector2(rb.linearVelocity.x, rb.linearVelocity.z);
                    float lateralPenalty = -0.4f * horizontalVelocity.magnitude;

                    // Reward for moving down (but not too fast) if above goal, else penalty for moving down
                    bool isAboveGoal = pos.y > goalPos.y + 0.3f;
                    float downwardVelocity = isAboveGoal ? Mathf.Clamp(rb.linearVelocity.y, 0f, 1f) : Mathf.Clamp(-rb.linearVelocity.y, 0f, 1f);
                    float descentReward = isAboveGoal ? 0.3f * downwardVelocity : -0.3f * downwardVelocity;

                    // Reward being close to horizontal alignment with the goal
                    float alignmentReward = horizontalDistance < 0.3f ? -0.3f : 0.1f;

                    // Penalty if far off-course
                    if (horizontalDistance > 0.6f)
                    {
                        totalReward -= 1.0f;
                        // EndEpisode(); // You can enable this if you want a hard fail
                    }

                    // Reward getting back on course
                    previousHorizontalDist = horizontalDistance;

                    if (horizontalProgress > 0)
                    {
                        totalReward += 0.3f * horizontalProgress;
                    }
                    else
                    {
                        totalReward += 0.1f * horizontalProgress; // smaller penalty if regressing
                    }

                    totalReward += descentReward + lateralPenalty + alignmentReward;

                    // Check if fully arrived at the goal (within small 3D range and mostly stopped)
                    if (distToGoal < 0.5f && Mathf.Abs(rb.linearVelocity.y) < 0.3f && rb.linearVelocity.magnitude < 1.0f)
                    {
                        totalReward += 3.0f;
                        EndEpisode();
                    }

                    // Fail-safe to stop very long descents
                    if (StepCount > 600)
                    {
                        totalReward -= 1.0f;
                        EndEpisode();
                    }
                    break;
            }

            AddReward(totalReward);

            /* ------------ LOGGING ------------ */
            if (stepCounter % logEvery == 0)
            {
                Debug.Log(
                    $"Step {StepCount} | " +
                    $"DronePos:{pos:F2} | " +
                    $"DronePos(RB):{rb.position:F2} | " +
                    $"GoalPos:{goalPos:F2} | " +
                    $"RewardThisStep:{totalReward:F3} | " +
                    $"TrainingStage: {trainingStage}"
                );
            }

            stepCounter++;
            /* ------------ END LOG ------------ */
        }

            /* ------------ LOGGING ------------ */
            // if (stepCounter % logEvery == 0)
            // {
            //     Debug.Log(
            //         $"Step {StepCount} | " +
            //         $"Dist:{totalDistance:F2} | " +
            //         $"DronePos:{pos:F2} | " +
            //         $"GoalPos:{goalPos:F2} | " +
            //         $"Dist:{totalDistance:F2} | " +
            //         $"Δd:{distanceDelta:F3} | " +
            //         $"Vel:{velocity.magnitude:F2} " +
            //         $"(→Goal:{velocityTowardsGoal:F2}) | " +
            //         $"UpDot:{uprightReward:F2} | TiltAlign:{tiltAlignment:F2} | " +
            //         $"RewardThisStep:{reward:F3}"
            //     );
            // }
            // stepCounter++;
            /* ------------ END LOG ------------ */

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
            AddReward(-0.5f);

            if (_renderer != null)
            {
                _renderer.material.color = Color.red;
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
