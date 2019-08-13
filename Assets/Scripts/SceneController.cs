using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using EasyRoads3Dv3;

public class SceneController : MonoBehaviour {
    private GameObject objectSegmentCamera;
    private ImageSynthesis imageSynthesis;
    private GameObject[] trees;
    private ERSideObjectInstance[] sideObjectsInstance;

    private bool treesLabeled = false;

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
            //trees = GameObject.FindGameObjectsWithTag("Tree");
            //trees = (GameObject[])FindObjectsOfType(typeof(ERTreeInstance));
            sideObjectsInstance = (ERSideObjectInstance[])FindObjectsOfType(typeof(ERSideObjectInstance));
            //foreach (GameObject tree in trees)
            //{
            //    tree.layer = LayerMask.NameToLayer("Tree");
            //}
            foreach (ERSideObjectInstance sideObjectInstance in sideObjectsInstance)
            {
                sideObjectInstance.combined = false;
                sideObjectInstance.so.layer = 11;
            }
        }
    }
}
