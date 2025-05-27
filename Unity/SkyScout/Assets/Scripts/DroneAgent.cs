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
        [SerializeField] private float environmentSize = 10f; // Size of each environment

        private Bounds bounds;
        private Resetter resetter;

        public override void Initialize()
        {
            multicopter.Initialize();
            
            // Create bounds relative to this environment
            bounds = new Bounds(Vector3.zero, Vector3.one * environmentSize);
            resetter = new Resetter(transform);
        }

        public override void OnEpisodeBegin()
        {
            resetter.Reset();
            SpawnObjects();
        }

        private void SpawnObjects()
        {
            // Reset drone position and rotation in local space
            transform.localPosition = new Vector3(0f, 0.5f, 0f);
            transform.localRotation = Quaternion.identity;

            // Spawn goal at random position within bounds
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

            // Add goal position observations using local positions
            Vector3 goalPos = _goal.localPosition;
            Vector3 dronePos = transform.localPosition;
            
            // Normalize positions relative to environment size
            sensor.AddObservation(goalPos.x / environmentSize);
            sensor.AddObservation(goalPos.y / environmentSize);
            sensor.AddObservation(goalPos.z / environmentSize);
            sensor.AddObservation(dronePos.x / environmentSize);
            sensor.AddObservation(dronePos.y / environmentSize);
            sensor.AddObservation(dronePos.z / environmentSize);
        }

        public override void OnActionReceived(ActionBuffers actionBuffers)
        {
            // Only use the first action value for thrust
            float thrust = actionBuffers.ContinuousActions[0];
            float[] synchronizedThrust = new float[multicopter.Rotors.Length];
            for (int i = 0; i < synchronizedThrust.Length; i++)
            {
                synchronizedThrust[i] = thrust;
            }
            multicopter.UpdateThrust(synchronizedThrust);

            if (bounds.Contains(transform.localPosition))
            {
                // Calculate horizontal and vertical distances separately
                Vector3 horizontalDronePos = new Vector3(transform.localPosition.x, 0, transform.localPosition.z);
                Vector3 horizontalGoalPos = new Vector3(_goal.localPosition.x, 0, _goal.localPosition.z);
                float horizontalDistanceToGoal = Vector3.Distance(horizontalDronePos, horizontalGoalPos);
                float verticalDistanceToGoal = Mathf.Abs(transform.localPosition.y - _goal.localPosition.y);

                // Higher penalty for vertical distance to encourage lifting
                AddReward(-0.02f * horizontalDistanceToGoal);
                //AddReward(-0.05f * verticalDistanceToGoal);

                float heightFromGround = transform.localPosition.y;
                AddReward(heightFromGround * 0.1f);
                // Additional reward for maintaining height
                if (heightFromGround > 1.0f)
                {
                    AddReward(multicopter.Frame.up.y);
                    AddReward(multicopter.Rigidbody.linearVelocity.magnitude * -0.2f);
                    AddReward(multicopter.Rigidbody.angularVelocity.magnitude * -0.1f);
                    AddReward(0.03f);
                }
            }
            else
            {
                resetter.Reset();
            }
        }

        private void OnTriggerEnter(Collider other)
        {
            if (other.gameObject.CompareTag("Goal"))
            {
                AddReward(1.0f);
                EndEpisode();
            }
        }

        private void OnCollisionEnter(Collision collision)
        {
            AddReward(-0.05f);

            if (_renderer != null)
            {
                _renderer.material.color = Color.red;
            }
        }

        private void OnCollisionStay(Collision collision)
        {
            if (collision.gameObject.CompareTag("Wall"))
            {
                AddReward(-0.01f * Time.fixedDeltaTime);
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