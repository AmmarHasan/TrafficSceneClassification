using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using EasyRoads3Dv3;
using System.Linq;
using System;
using Assets.Scripts.Extensions;
using Assets.Scripts.Classes;
using System.IO;
using System.Text;
using System.Globalization;
using Assets.Scripts.Configuration;

public class EasyRoadsGenerator : MonoBehaviour
{
    public ERRoadNetwork network;
    public GameObject cameraCar;
    public GameObject car;
    [Range(0.001f, 0.2f)]
    public float percentageEven;
    public List<CustomEasyRoad> customEasyRoads;
    [HideInInspector]
    public bool isGenerated = false;
    public bool isSelfDriving = false;
    [HideInInspector]
    public bool isPlaced = false;
    public int carSpeed = 50;
    public int numberOfTracks = 0;
    private CultureInfo culture;
    private List<Vector3> lane1CenterPoints = new List<Vector3>();
    private int currentIndex = 0;
    void Start()
    {
        network = new ERRoadNetwork();
        network.BuildRoadNetwork();
        customEasyRoads = new List<CustomEasyRoad>();
        ProjectOnCamera2D projectOnCamera2D = car.GetComponent<ProjectOnCamera2D>();
        projectOnCamera2D.cam = Camera.main;

        culture = CultureInfo.CreateSpecificCulture("en-US");
        //if (this.isSelfDriving)
        //{

            InvokeRepeating("SimulateCar", 2.0f, 2.0f);
        //}
    }

    void Update()
    {
        
    }

    #region FixedUpdate
    void FixedUpdate()
    {
        if (isGenerated)
        {
            if (!isPlaced)
            {
                this.PlaceCameraCar();
            }

            // If the car does not drive itself, then simulate driving
            // if (this.isSelfDriving)
            // {
            //     SimulateCar();
            // }

            // Activate the ScreenRecorder
            ScreenRecorder screenRecorder = Camera.main.GetComponent<ScreenRecorder>();
            screenRecorder.isGenerated = true;
            screenRecorder.updateCounter++;

            if (screenRecorder.updateCounter % screenRecorder.takePictureEveryXFrame == 0 && screenRecorder.capture)
            {
                // Iterate over all cars and save the coordinates of the visible ones.
                //string textToAppend = "Picture " + screenRecorder.counter + ":";
                string textToAppend = string.Empty;
                List<Tuple<GameObject, int>> visibleCars = new List<Tuple<GameObject, int>>();
                foreach (CustomEasyRoad ceRoad in customEasyRoads)
                {
                    foreach (Tuple<GameObject, int> carOnLane in ceRoad.CarsOnLanes)
                    {
                        if (carOnLane == null || carOnLane.First == null)
                        {
                            continue;
                        }

                        ProjectOnCamera2D projectOnCamera2D = carOnLane.First.GetComponent<ProjectOnCamera2D>();

                        // If the car is visible on the screen, save the coordinates.
                        if (projectOnCamera2D.IsVisible)
                        {
                            visibleCars.Add(carOnLane);
                        }
                    }
                }

                if (visibleCars.Count < 1)
                {
                    return;
                }
                //textToAppend += "---\n";
                //foreach(Tuple<GameObject, int> visibleCar in visibleCars)
                //{
                //    textToAppend +=
                //        visibleCar.Second + "," +
                //        visibleCar.First.GetComponent<ProjectOnCamera2D>()
                //            .getRelativeBoxCoords()
                //            .Select(c => c.First.ToString("G", culture) + "," + c.Second.ToString("G", culture))
                //            .Aggregate((a, b) => a + "," + b)
                //        + ";";
                //    textToAppend += "\n";

                //    // Write Visible car's bounding rectangle pixel value
                //    Rect r = visibleCar.First.GetComponent<ProjectOnCamera2D>()
                //            .getCarPositionInFrame();
                //    textToAppend += r.center + ",";
                //    textToAppend += "(" + r.xMin + ",";
                //    textToAppend += r.yMin+ "),";
                //    textToAppend += "(" + r.xMax + ",";
                //    textToAppend += r.yMax + ")\n";
                //}

                //textToAppend += "---\n";
                screenRecorder.TakePicture(textToAppend);
            }

            DestroyColliderCars();
        }
    }
    #endregion

    #region SetUpRoadType
    /// <summary>
    /// Methode zum Aufsetzen des RoadTypes anhand der Lanes.
    /// </summary>
    public void SetUpRoadType()
    {
        // Die Strecke neu holen
        this.network = new ERRoadNetwork();
        this.network.BuildRoadNetwork();

        //foreach (ERRoadType roadType in this.network.GetRoadTypes()[0])
        //{
        ERRoadType roadType = this.network.GetRoadTypes()[0];
        roadType.layer = 10;

        switch (this.numberOfTracks)
        {
            case 8:
                // If eight tracks are needed (roadWidth has been adjusted)
                roadType.roadMaterial = Resources.Load<Material>("Road8Lanes");
                roadType.roadWidth = 24;
                roadType.Update();
                break;
            case 6:
                // Falls sechs Spuren benötigt werden (roadWidth wurde angepasst)
                roadType.roadMaterial = Resources.Load<Material>("Road6Lanes");
                roadType.roadWidth = 18;
                roadType.Update();
                break;
            case 4:
                // Falls vier Spuren benötigt werden
                roadType.roadMaterial = Resources.Load<Material>("Road4Lanes");
                roadType.roadWidth = 12;
                roadType.Update();
                break;
            case 3:
                // Falls zwei Spuren benötigt werden (roadMaterial stimmt schon)
                roadType.roadMaterial = Resources.Load<Material>("Road3Lanes");
                roadType.roadWidth = 9;
                roadType.Update();
                break;
            case 2:
                // If two tracks are needed (roadWidth has been adjusted)
                roadType.roadWidth = 6;
                roadType.Update();
                break;
            case 1:
            default:
                // Falls zwei Spuren benötigt werden (roadMaterial stimmt schon)
                roadType.roadMaterial = Resources.Load<Material>("Road1Lane");
                roadType.roadWidth = 3;
                roadType.Update();
                break;
        }
    }
    #endregion

    #region CreateCurve
    /// <summary>
    /// Creates a curve based on an angle, the length of the curve, and the positions of the current and previous road elements.
    /// </summary>
    /// <param name="angle">The angle.</param>
    /// <param name="length">The length of the road element.</param>
    /// <param name="heightDifference">The height difference for the section of the route.</param>
    /// <param name="minCars">Die minimale Anzahl an Autos auf diesem Streckenabschnitt.</param>
    /// <param name="maxCars">Die maximale Anzahl an Autos auf diesem Streckenabschnitt.</param>
    /// <param name="seed">Der Seed des Random-Generators.</param>
    /// <returns>Die Kurve.</returns>
    public ERRoad CreateCurve(float angle, float length, float? heightDifference, int minCars, int maxCars, string seed)
    {
        // Get the track new
        this.network = new ERRoadNetwork();
        this.network.BuildRoadNetwork();

        // get the height difference
        float fixHeightDifference = heightDifference ?? 0f;

        // Initialize the start position.
        Vector3 startPosition = new Vector3(0, 0, 0);

        // Initialize the alignment (default is z-direction).
        Vector3 heading = new Vector3(0, 0, 1);

        // Get the RoadType
        ERRoadType roadType = this.network.GetRoadTypes()[0];

        // Get the position of the last leg, if any.
        ERRoad lastRoad = null;
        if (network.GetRoads().Length > 0)
        {
            lastRoad = network.GetRoads().Last();
            Vector3[] markers = lastRoad.GetMarkerPositions();
            Vector3 lastPosition = markers.Last();

            // Adjust the starting position to the last section of track.
            startPosition = lastPosition;

            // Get the alignment with respect to the previous stretch.
            Vector3 secondToLast = markers[markers.Count() - 2];
            heading = lastPosition - secondToLast;
            heading.y = 0;
        }

        // Compute the (even) direction vector.
        Vector3 direction = heading / heading.magnitude;

        // The vector of the y-axis
        Vector3 yAxis = new Vector3(0, 1, 0);

        // The number of positions to calculate for the curve
        int numbPositions = Convert.ToInt32(Math.Abs(angle));
        float positionPercentage = numbPositions * percentageEven;

        // The array with the new positions.
        Vector3[] curvePositions = new Vector3[numbPositions];
        curvePositions[0] = startPosition;

        // positions are calculated in 1-degree increments.
        float anglePart = angle / Math.Abs(angle);
        float lengthPart = length / numbPositions;
        float heightPart = fixHeightDifference / (numbPositions - (2 * positionPercentage));

        // calculate the positions.
        for (int i = 1; i < numbPositions; i++)
        {
            // calculate the direction for the next step
            if (i > 1)
            {
                heading = curvePositions[i - 1] - curvePositions[i - 2];
                heading.y = 0;
                direction = heading / heading.magnitude;
            }

            // Get the last position.
            Vector3 oldPosition = curvePositions[i - 1];

            // within the percent range, apply the height.
            if (i > positionPercentage && i < (numbPositions - positionPercentage))
            {
                oldPosition.y += heightPart.Truncate(5);
            }

            // Calculate the new position.
            curvePositions[i] = oldPosition + Quaternion.AngleAxis(anglePart, yAxis) * direction * lengthPart;
            curvePositions[i].y = 0.01f;
        }

        // Create the curve.
        ERRoad thisRoad = this.network.CreateRoad("Curve" + network.GetRoads().Count(), roadType, curvePositions);
        customEasyRoads.Add(new CustomEasyRoad(car, thisRoad, minCars, maxCars, numberOfTracks, RoadPartType.Curve));
        return thisRoad;
    }
    #endregion

    #region CreateStraight
    /// <summary>
    /// Method for drawing a straight road.
    /// </summary>
    /// <param name="length">Die Länge der Straße.</param>
    /// <param name="minCars">Die minimale Anzahl der Autos auf dem Straßenteil.</param>
    /// <param name="maxCars">Die maximale Anzahl der Autos auf dem Straßenteil.</param>
    /// <param name="heightDifference">Der Höhenunterschied.</param>
    /// <param name="seed">Der Seed.</param>
    /// <returns>Die Straße.</returns>
    public ERRoad CreateStraight(float length, int minCars, int maxCars, float? heightDifference, string seed)
    {
        // Die Strecke neu holen
        this.network = new ERRoadNetwork();
        this.network.BuildRoadNetwork();

        // Den RoadType holen
        ERRoadType roadType = this.network.GetRoadTypes()[0];

        // Hole die akutellen Streckenteile
        ERRoad[] currentRoads = network.GetRoads();

        // Hole die Höhe der Strecke
        float fixHeightDifference = heightDifference ?? 0;

        // Lege die Positionen der Strecke an
        Vector3 startPosition = new Vector3(0, 0.01f, 0);
        Vector3 middlePosition = new Vector3(0, fixHeightDifference / 2, length / 2);
        Vector3 endPosition = new Vector3(0, fixHeightDifference, length);

        ERRoad lastRoad = null;
        ERRoad road = null;

        if (currentRoads.Length > 0)
        {
            // Hole die letzte Strecke
            lastRoad = currentRoads.Last();

            // Hole den letzten Punkt der Strecke
            Vector3[] markers = lastRoad.GetMarkerPositions();
            Vector3 lastMarker = markers.Last();

            // Die richtige Rotation ausrechnen
            Vector3 heading = (lastMarker - markers[markers.Length - 2]);
            Vector3 direction = heading / heading.magnitude;
            direction.y = 0;

            // Das Verhältnis zwischen x und z-Achse ausrechnen
            float x = direction.x / (direction.magnitude);
            float z = direction.z / (direction.magnitude);

            Vector3[] streetVectors = new Vector3[(int)length];
            float heightPart = fixHeightDifference / length;
            for (int lengthPart = 0; lengthPart < length; lengthPart++)
            {
                streetVectors[lengthPart] = lastMarker + new Vector3(x * lengthPart, heightPart * lengthPart, z * lengthPart);
            }

            // Generiere Straße
            road = network.CreateRoad("Straight" + currentRoads.Length, roadType, streetVectors);
        }
        else
        {
            // Generiere erste Straße
            road = network.CreateRoad("Straight" + currentRoads.Length, roadType, new Vector3[] { startPosition, middlePosition, endPosition });
        }

        // Erstelle die Strecke mit einem eindeutigen Namen
        customEasyRoads.Add(new CustomEasyRoad(car,road,minCars,maxCars, numberOfTracks, RoadPartType.Straight));
        return road;
    }
    #endregion

    #region CreateLane
    public void CreateLane(ERRoad originalRoadPart, RoadPartType roadPartType, int laneNumber)
    {
        Vector3[] originialRoadPostions = originalRoadPart.roadScript.splinePoints.ToArray();
        List<Vector3> meshVectorsOriginal = originalRoadPart.roadScript.meshVecs;
        ERRoadType roadTypeLane = this.network.GetRoadTypes()[laneNumber];
        int numbPositions = originialRoadPostions.Length;
        Vector3[] curvePositions = new Vector3[numbPositions];

        // Important: Step should be numberOfTracks+1. 
        // For this you must change number of segments for each road type
        int step = meshVectorsOriginal.Count / numbPositions;

        for (int m = 0, i = 0; m < meshVectorsOriginal.Count && i < numbPositions; m += step, i++)
        {
            Vector3 laneMiddle = Vector3.Lerp(meshVectorsOriginal[m + (step - laneNumber - 1)], meshVectorsOriginal[m + (step - laneNumber)], 0.5f);
            curvePositions[i] = laneMiddle;
            curvePositions[i].y = originialRoadPostions[i].y + 0.01f;
        }
        ERRoad newRoad = this.network.CreateRoad(roadPartType.ToString() + network.GetRoads().Count() + "-lane:"+laneNumber, roadTypeLane, curvePositions);
        if (laneNumber == 1) { 
            newRoad.SetTag("lane1");
            for (int i = 0; i < curvePositions.Length; i++)
            {
                lane1CenterPoints.Add(curvePositions[i]);
            }
        }
    }
    #endregion

    #region SimulateCar
    /// <summary>
    /// Methode zum Fahren des Autos (Simulation).
    /// </summary>
    public void SimulateCar()
    {
        // Bring the colliders of the car
        Collider[] colliders = Physics.OverlapBox(cameraCar.gameObject.transform.position, (cameraCar.gameObject.transform.localScale / 5f), cameraCar.gameObject.transform.rotation);
        ERRoad road = null;
        foreach (Collider collider in colliders)
        {
            if (collider.tag == "lane1")
            {
                road = network.GetRoadByName(collider.name);
            }
        }

        Vector3 heading = new Vector3(0, 0, 1);
        if (road != null)
        {
            // Get the last point of the track
            //Vector3[] markers = road.GetMarkerPositions();
            Vector3[] markers = road.GetSplinePointsCenter();
            Vector3 lastMarker = markers.Last();

            // Die richtige Rotation ausrechnen
            heading = (lastMarker - markers[markers.Length - 2]).normalized;
            heading.y = 0;
            CustomEasyRoad customEasy = null;
            foreach (CustomEasyRoad customEasyRoad in customEasyRoads)
            {
                if (customEasyRoad.Road.GetName() == road.GetName())
                {
                    customEasy = customEasyRoad;
                    break;
                }
            }

            //Tuple<Vector3, Vector3> markers = customEasy.GetIncludingMarkers(cameraCar.gameObject.transform.position);
            //heading = (markers.Second - markers.First).normalized;
            //heading.y = 0;
        }

        // Set speed
        Rigidbody rigidbody = cameraCar.GetComponent<Rigidbody>();
        // cameraCar.transform.Translate(Vector3.forward * (carSpeed / 3.6f) * Time.deltaTime);
        // //cameraCar.transform.rotation.SetLookRotation(heading);
        // cameraCar.transform.rotation = Quaternion.Slerp(transform.rotation, Quaternion.LookRotation(heading), 2.5f * Time.deltaTime);
        Vector3 nextPoint= lane1CenterPoints[currentIndex];
        nextPoint.y = nextPoint.y + 0.01f;
        cameraCar.transform.position = nextPoint;

        Vector3 targetPosition = new Vector3(lane1CenterPoints[currentIndex+1].x, lane1CenterPoints[currentIndex + 1].y, lane1CenterPoints[currentIndex + 1].z);
        cameraCar.transform.LookAt(targetPosition);
        currentIndex+=4;
        Debug.Log("currentIndex: "+ currentIndex);
        if (currentIndex - 12 > lane1CenterPoints.Count) {
            Debug.Log(currentIndex);
            Debug.Log("lane1CenterPointsCOunt: " + lane1CenterPoints.Count);
            UnityEditor.EditorApplication.isPlaying = false;
        }
    }
    #endregion

    #region PlaceCameraCar
    /// <summary>
    /// Method of placing the car in the beginning.
    /// </summary>
    private void PlaceCameraCar()
    {
        // Place the car on the right first lane
        ERRoad firstRoad = network.GetRoads().First();
        Vector3 firstSplineRightSide = firstRoad.GetSplinePointsRightSide()[0];
        Vector3 firstMarker = firstRoad.GetMarkerPosition(0);
        Vector3 rightMiddleLane = Vector3.zero;

        // Interpolate the position according to the number of tracks
        switch (this.numberOfTracks)
        {
            case 8:
                rightMiddleLane = Vector3.Slerp(firstMarker, firstSplineRightSide, 0.85f);
                break;
            case 6:
                rightMiddleLane = Vector3.Slerp(firstMarker, firstSplineRightSide, 0.825f);
                break;
            case 4:
                rightMiddleLane = Vector3.Slerp(firstMarker, firstSplineRightSide, 0.75f);
                break;
            case 3:
                rightMiddleLane = Vector3.Slerp(firstMarker, firstSplineRightSide, 0.6f);
                break;
            case 2:
                rightMiddleLane = Vector3.Slerp(firstMarker, firstSplineRightSide, 0.5f);
                break;
            case 1:
            default:
                rightMiddleLane = Vector3.Slerp(firstMarker, firstSplineRightSide, 0.25f);
                break;
        }

        // Put the car in the right position
        cameraCar.gameObject.transform.position = rightMiddleLane;
        cameraCar.gameObject.transform.rotation = Quaternion.identity;

        // Ist platziert setzen
        isPlaced = true;
    }
    #endregion

    #region DestroyColliderCars
    /// <summary>
    /// Methode zum Zerstören der Autos im Weg.
    /// </summary>
    private void DestroyColliderCars()
    {
        // Bring the colliders of the car
        Collider[] colliders = Physics.OverlapBox(cameraCar.gameObject.transform.position, (cameraCar.gameObject.transform.localScale / 2.5f), cameraCar.gameObject.transform.rotation);
        List<Collider> carColliders = new List<Collider>();
        foreach (Collider collider in colliders)
        {
            if (collider.tag == "Car")
            {
                carColliders.Add(collider);
            }
        }

        // Remove the cars in the way
        foreach (Collider collider in carColliders)
        {
            Destroy(collider.gameObject);
        }
    }

    #endregion
}
