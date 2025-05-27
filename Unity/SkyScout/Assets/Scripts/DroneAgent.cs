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

            Vector3 goalPos = _goal.localPosition;
            Vector3 dronePos = transform.localPosition;

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
            Vector3 directionToGoalLocal = transform.InverseTransformDirection(directionToGoalWorld);
            sensor.AddObservation(directionToGoalLocal);
        }

        public override void OnActionReceived(ActionBuffers actionBuffers)
        {
            float[] actions = actionBuffers.ContinuousActions.Array;
            multicopter.UpdateThrust(actions);

            
            Vector3 localPos = transform.localPosition;

            Vector3 dirToGoal = (_goal.localPosition - localPos).normalized;
            Vector3 upVector = transform.up;
            Vector3 velocity = multicopter.Rigidbody.linearVelocity;

            float uprightReward = Vector3.Dot(upVector, Vector3.up);

            float tiltAlignment = Vector3.Dot(upVector, -dirToGoal);
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

            Vector3 horizontalDronePos = new Vector3(localPos.x, 0f, localPos.z);
            Vector3 horizontalGoalPos = new Vector3(_goal.localPosition.x, 0f, _goal.localPosition.z);
            float horizontalDistance = Vector3.Distance(horizontalDronePos, horizontalGoalPos);
            float verticalDistance = Mathf.Abs(localPos.y - _goal.localPosition.y);
            float totalDistance = Vector3.Distance(localPos, _goal.localPosition);

            float heightFromGround = localPos.y;

            float reward = 0f;
            // Distance penalties (scale down)
            // Note: instead of penalizing, try to reward it when the distance shrinks
            reward += -0.05f * horizontalDistance;
            reward += -0.05f * verticalDistance;

            // Positive incentives
            reward += 0.1f * uprightReward;
            reward += 0.3f * tiltAlignment;
            reward += 0.2f * velocityTowardsGoal;
            /*
            // --- NEW: Inclination alignment reward ---

            Vector3 horizontalDirToGoal = new Vector3(dirToGoal.x, 0f, dirToGoal.z).normalized;

            // Goal direction projected onto drone's local right and forward vectors
            float goalAlongRight = Vector3.Dot(horizontalDirToGoal, transform.right);
            float goalAlongForward = Vector3.Dot(horizontalDirToGoal, transform.forward);

            Vector3 inclination = multicopter.Inclination;  // (right.y, up.y, forward.y)
            float inclinationRight = inclination.x;
            float inclinationForward = inclination.z;

            // Reward if inclination matches goal direction in horizontal plane
            float inclinationAlignment = (inclinationRight * goalAlongRight) + (inclinationForward * goalAlongForward);

            reward += inclinationAlignment;

            // --- END NEW ---
            */
            Vector3 goalPos = _goal.localPosition;
            float previousDistance = Vector3.Distance(previousPosition, goalPos);
            float currentDistance = Vector3.Distance(transform.localPosition, goalPos);

            float distanceDelta = previousDistance - currentDistance;
            reward += distanceDelta * 0.1f - 0.1f;
            AddReward(reward);

            previousPosition = transform.localPosition;
            /*

            // Speed mismatch penalty (scale with distance)
            float idealSpeed = 1f * totalDistance;
            float speedMismatchPenalty = Mathf.Pow(velocity.magnitude - idealSpeed, 2) * (totalDistance / 5f);
            reward += -0.1f * speedMismatchPenalty;

            // Add total reward
            AddReward(reward);

            // Close to goal bonus
            if (horizontalDistance < 0.5f && verticalDistance < 0.5f)
            {
                AddReward(5f);
            }
            */
            // Time penalty
            AddReward(-0.005f);
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
