using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using EasyRoads3Dv3;

public class SceneController : MonoBehaviour {
    private GameObject objectSegmentCamera;
    private ImageSynthesis imageSynthesis;


    // Use this for initialization
    void Start () {
        objectSegmentCamera = GameObject.FindWithTag("ObjectSegmentCamera");
        imageSynthesis = objectSegmentCamera.GetComponent<ImageSynthesis>();
    }
    
    // Update is called once per frame
    void Update () {
        imageSynthesis.OnSceneChange();
        if (Input.GetKey("e"))
        {
            Debug.Log("up arrow key is held down");
          
        }
    }
}
