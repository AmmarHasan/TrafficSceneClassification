using UnityEngine;
using EasyRoads3Dv3;
using System.Collections;
using Assets.Scripts.Classes;
using System.Collections.Generic;


public class CustomEasyRoad
{
    /// <summary>
    /// All cars on the track.
    /// </summary>
    public List<Tuple<GameObject, int>> CarsOnLanes  { get; private set; }

    /// <summary>
    /// Das RoadNetwork.
    /// </summary>
    private readonly ERRoadNetwork Network;
    
    /// <summary>
    /// Die Road.
    /// </summary>
    public ERRoad Road { get; private set; }

    /// <summary>
    /// The cars per track.
    /// </summary>
    public Dictionary<int, List<GameObject>> CarsPerLane { get; private set; }

    /// <summary>
    /// Method for creating a CustomEasyRoad.
    /// </summary>
    /// <param name="car">The Gameobject of the Autos.</param>
    /// <param name="road">The street.</param>
    /// <param name="minCars">The minimum number of cars on the track.</param>
    /// <param name="maxCars">The maximum number of cars on the track.</param>
    /// <param name="numberOfTracks">The number of tracks.</param>
    public CustomEasyRoad(GameObject car, ERRoad road, int minCars, int maxCars, int numberOfTracks)
    {
        road.SetTag("Street");

        this.Road = road;
        this.CarsOnLanes = new List<Tuple<GameObject, int>>();
        this.CarsPerLane = new Dictionary<int, List<GameObject>>();
        for (int i = 0; i < numberOfTracks; i++)
        {
            List<GameObject> carsOnLane = new List<GameObject>();
            CarsPerLane.Add(i, carsOnLane);
        }

        Vector3[] markers = road.GetSplinePointsCenter();
        Vector3[] markersR = road.GetSplinePointsRightSide();
        Vector3[] markersL = road.GetSplinePointsLeftSide();
        
        int carCount = Random.Range(minCars, maxCars);
        if (carCount > 0)
        {      
            int increment = markers.Length / carCount;
            Vector3 look = Vector3.zero;
            GameObject newCar = null;

            for (int i = 0; i < markers.Length; i+= increment)
            {
                // Determine the track
                int lane = Random.Range(0, numberOfTracks);
                Vector3[] directionMarkers = null;

                // Bring and set the direction of the car / lane
                if (numberOfTracks==3) {
                    if (lane <= 1)
                    {
                        directionMarkers = markersL;
                        look = (markers[Mathf.Max(0, i - 1)] - markers[Mathf.Min(markers.Length - 1, i + 1)]);
                    }
                    else
                    {
                        directionMarkers = markersR;
                        look = (markers[Mathf.Min(markers.Length - 1, i + 1)] - markers[Mathf.Max(0, i - 1)]);
                    }
                } else {
                    if (lane < (numberOfTracks / 2))
                    {
                        directionMarkers = markersL;
                        look = (markers[Mathf.Max(0, i - 1)] - markers[Mathf.Min(markers.Length - 1, i + 1)]);
                    }
                    else
                    {
                        directionMarkers = markersR;
                        look = (markers[Mathf.Min(markers.Length - 1, i + 1)] - markers[Mathf.Max(0, i - 1)]);
                    }
                }

                // Get the RoadSlerp
                float roadSlerp = RoadUtils.GetRoadSlerpByLane(numberOfTracks, lane);

                // Spawn the car with the direction and the trail
                newCar = GameObject.Instantiate(car, Vector3.Slerp(markers[i], directionMarkers[i], roadSlerp) + new Vector3(0, 1, 0), Quaternion.LookRotation(look));

                // Add the car to the lists
                this.AddToIndexOnLane(lane, newCar);
                CarsOnLanes.Add(new Tuple<GameObject, int>(newCar, numberOfTracks - lane - 1));
            }
        }
    }

    #region GetIncludingMarkers
    public Tuple<Vector3, Vector3> GetIncludingMarkers(Vector3 position)
    {
        Vector3[] markers = Road.GetMarkerPositions();

        // TODO: Return markers between positions
        for (int i = 0; i < markers.Length - 1; i++)
        {
            Vector3 currentMarker = markers[i];
            Vector3 nextMarker = markers[i + 1];
            if (currentMarker.x <= position.x && currentMarker.y <= position.z && nextMarker.x >= position.x && nextMarker.z >= position.z
                || currentMarker.x >= position.x && currentMarker.y >= position.z && nextMarker.x <= position.x && nextMarker.z <= position.z)
            {
                return new Tuple<Vector3, Vector3>(currentMarker, nextMarker);
            }
        }

        return new Tuple<Vector3, Vector3>(Vector3.zero, Vector3.one);
    }
    #endregion

    #region AddToIndexOnLane
    /// <summary>
    /// Method for adding a car to the dictionary on a specific route.
    /// </summary>
    /// <param name="i">Der Index der Lane.</param>
    /// <param name="car">Das Auto.</param>
    private void AddToIndexOnLane(int i, GameObject car)
    {
        // Get the list of the track
        List<GameObject> carsOnLane = CarsPerLane[i];

        // Add the car
        carsOnLane.Add(car);

        // Reset the cars of the track
        CarsPerLane[i] = carsOnLane;
    }
    #endregion
}
