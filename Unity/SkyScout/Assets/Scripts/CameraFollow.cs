using UnityEngine;

public class CameraFollow : MonoBehaviour
{
    public Transform target;          // DroneAgent Transform
    public Vector3 offset = new Vector3(0, 5, -10);
    public float followSpeed = 5f;
    public bool lookAtTarget = true;

    void LateUpdate()
    {
        if (target == null) return;

        // Smooth follow
        Vector3 desiredPosition = target.position + offset;
        transform.position = Vector3.Lerp(transform.position, desiredPosition, followSpeed * Time.deltaTime);

        // Optional: look at the drone
        if (lookAtTarget)
        {
            transform.LookAt(target);
        }
    }
}
