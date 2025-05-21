using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine.InputSystem;

public class DroneAgent : Agent
{
   [SerializeField] private Transform _goal;
   [SerializeField] private float _moveSpeed = 1.5f;
   [SerializeField] private float _rotationSpeed = 180f;

   private Renderer _renderer;

   private int _currentEpisode = 0;
   private float _cumulativeReward = 0f;
   
    public override void Initialize()
    {
        Debug.Log("Initialize");
        _renderer = GetComponent<Renderer>();
        _currentEpisode = 0;
        _cumulativeReward = 0f;
    }
    
    public override void OnEpisodeBegin()
    {
        Debug.Log("OnEpisodeBegin");
        _currentEpisode++;
        _cumulativeReward = 0f;
        _renderer.material.color = Color.blue;

        SpawnObjects();
    }

    private void SpawnObjects() {
        transform.localRotation = Quaternion.identity;
        transform.localPosition = new Vector3(0f, 0.3f, 0f);

        float randomAngle = Random.Range(0f, 360f);
        Vector3 randomDirection = Quaternion.Euler(0f, randomAngle, 0f) * Vector3.forward;

        float randomDistance = Random.Range(1f, 2.5f);

        Vector3 goalPosition = transform.localPosition + randomDirection * randomDistance;

        _goal.localPosition = new Vector3(goalPosition.x, 0.3f, goalPosition.z);
    }
    
    public override void CollectObservations(VectorSensor sensor) {
        float goalPosX_normalized = _goal.localPosition.x / 5f;
        float goalPosZ_normalized = _goal.localPosition.z / 5f;

        float turtlePosX_normalized = transform.localPosition.x / 5f;
        float turtlePosZ_normalized = transform.localPosition.z / 5f;

        sensor.AddObservation(goalPosX_normalized);
        sensor.AddObservation(goalPosZ_normalized);
        sensor.AddObservation(turtlePosX_normalized);
        sensor.AddObservation(turtlePosZ_normalized);
    }

    public override void Heuristic(in ActionBuffers actionsOut) {
        var discreteActions = actionsOut.DiscreteActions;
        discreteActions[0] = 0;

        if (Keyboard.current != null && Keyboard.current.upArrowKey.isPressed) {
            discreteActions[0] = 1;
        } else if (Keyboard.current != null && Keyboard.current.leftArrowKey.isPressed) {
            discreteActions[0] = 2;
        } else if (Keyboard.current != null && Keyboard.current.rightArrowKey.isPressed) {
            discreteActions[0] = 3;
        }
    }

    public override void OnActionReceived(ActionBuffers actions) {
        MoveAgent(actions.DiscreteActions);

        AddReward(-2f / MaxStep);

        _cumulativeReward = GetCumulativeReward();
    }

    public void MoveAgent(ActionSegment<int> act) {
        var action = act[0];

        switch (action) {
            case 1:
                transform.position += transform.forward * _moveSpeed * Time.deltaTime;
                break;
            case 2:
                transform.Rotate(0f, -_rotationSpeed * Time.deltaTime, 0f);
                break;
            case 3:
                transform.Rotate(0f, _rotationSpeed * Time.deltaTime, 0f);
                break;
        }
    }

    public void OnTriggerEnter(Collider other) {
        if (other.gameObject.CompareTag("Goal")) {
            GoalReached();
        }
    }

    private void GoalReached() {
        AddReward(1.0f);
        _cumulativeReward += GetCumulativeReward();
        EndEpisode();
    }

    private float GetCumulativeReward() {
        return _cumulativeReward;
    }

    private void OnCollisionEnter(Collision collision) {
        AddReward(-0.05f);

        if (_renderer != null) {
            _renderer.material.color = Color.red;
        }
    }

    public void OnCollisionStay(Collision collision) {
        if (collision.gameObject.CompareTag("Wall")) {
            AddReward(-0.01f * Time.fixedDeltaTime);
        }
    }

    public void OnCollisionExit(Collision collision) {
        if (collision.gameObject.CompareTag("Wall")) {
            if (_renderer != null) {
                _renderer.material.color = Color.blue;
            }
        }
    }
}
