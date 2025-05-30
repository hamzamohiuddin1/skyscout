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

        private string trainingStage = "takeoff"; // 'takeoff' | 'findTarget' | 'land'

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

            switch (trainingStage)
            {
                case "takeoff":
                    if (rb.position.y > 1.5f)
                    {
                        totalReward += 1.0f;
                        trainingStage = "findTarget";
                    }
                    else
                    {
                        totalReward = 0.1f * rb.linearVelocity.y;
                    }
                    break;

                case "findTarget":
                    if (Vector3.Distance(pos, goalPos) == 0f)
                    {
                        // Reward added in OnTriggerEnter
                        trainingStage = "land";
                        totalReward += 3.0f;
                    }
                    else
                    {
                        Vector3 dirToGoal = (goalPos - pos).normalized;
                        Vector3 upVector = rb.rotation * Vector3.up;
                        Vector3 velocity = rb.linearVelocity;

                        float uprightReward = Vector3.Dot(upVector, Vector3.up);

                        float tiltAlignment = Mathf.Max(0f, Vector3.Dot(upVector, dirToGoal));
                        if (uprightReward > 0.5f)
                        {
                            tiltAlignment *= uprightReward;
                        }
                        else
                        {
                            tiltAlignment = 0f;
                        }

                        float velocityTowardsGoal = Vector3.Dot(velocity.normalized, dirToGoal);
                        float totalDistance = Vector3.Distance(pos, goalPos);

                        totalReward += -0.1f * totalDistance;
                        totalReward += 0.1f * uprightReward;
                        totalReward += 0.3f * tiltAlignment;
                        totalReward += 0.2f * velocityTowardsGoal;

                        float distanceDelta = Vector3.Distance(previousPosition, goalPos) - totalDistance;
                        totalReward += distanceDelta * 0.5f;
                        previousPosition = pos;

                        totalReward -= 0.005f;
                    }
                    break;

                case "land":
                    if (rb.position.y < 0.1f)
                    {
                        trainingStage = "takeoff";
                        totalReward += 2.0f;
                        AddReward(totalReward);
                        EndEpisode();
                    }
                    else
                    {
                        float descentReward = 0.1f * -rb.linearVelocity.y;
                        float altitudePenalty = 0.1f * -rb.position.y;
                        totalReward += descentReward + altitudePenalty;
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
