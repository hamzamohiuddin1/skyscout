using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

namespace MBaske
{
    public class DroneAgent : Agent
    {
        [SerializeField] private Transform _goal;
        [SerializeField] private Renderer _renderer;
        [SerializeField] private float environmentSize = 10f;
        [SerializeField] private float moveSpeed = 2f; // Units per second

        private Bounds bounds;
        private Resetter resetter;

        public override void Initialize()
        {
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
            // Reset drone
            transform.localPosition = new Vector3(0f, 0.5f, 0f);
            transform.localRotation = Quaternion.identity;

            // Place goal randomly within bounds
            float randomAngle = Random.Range(0f, 360f);
            Vector3 randomDirection = Quaternion.Euler(0f, randomAngle, 0f) * Vector3.forward;
            float randomDistance = Random.Range(1f, 3f);
            Vector3 goalPosition = transform.localPosition + randomDirection * randomDistance;
            _goal.localPosition = new Vector3(goalPosition.x, 2f, goalPosition.z);
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            Vector3 goalPos = _goal.localPosition / environmentSize;
            Vector3 dronePos = transform.localPosition / environmentSize;

            sensor.AddObservation(goalPos);
            sensor.AddObservation(dronePos);
        }

        public override void OnActionReceived(ActionBuffers actionBuffers)
        {
            Vector3 move = Vector3.zero;
            float delta = moveSpeed * Time.deltaTime;

            var act = actionBuffers.ContinuousActions;
            move.x = Mathf.Clamp(act[0], -1f, 1f);
            move.y = Mathf.Clamp(act[1], -1f, 1f);
            move.z = Mathf.Clamp(act[2], -1f, 1f);

            transform.localPosition += move * delta;

            // Keep within bounds
            if (!bounds.Contains(transform.localPosition))
            {
                AddReward(-1f);
                EndEpisode();
                return;
            }

            // Reward shaping
            float distance = Vector3.Distance(transform.localPosition, _goal.localPosition);
            AddReward(-distance * 0.001f);
        }

        private void OnTriggerEnter(Collider other)
        {
            if (other.CompareTag("Goal"))
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
            if (collision.gameObject.CompareTag("Wall") && _renderer != null)
            {
                _renderer.material.color = Color.blue;
            }
        }
    }
}
