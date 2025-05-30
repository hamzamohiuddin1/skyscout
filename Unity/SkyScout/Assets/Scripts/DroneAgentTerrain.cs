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

        private void SpawnObjects()
        {
            // Set drone position to (0, 2.34, -17)
            transform.localPosition = new Vector3(0f, 2.34f, -17f);
            transform.localRotation = Quaternion.identity;

            // Set goal to fixed position (9, 3, -5)
            _goal.localPosition = new Vector3(9f, 3f, -5f);
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
            reward += 0.3f * tiltAlignment;
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
                    $"UpDot:{uprightReward:F2} | TiltAlign:{tiltAlignment:F2} | " +
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
