namespace Assets.Scripts.Configuration
{
    using System;
    using System.IO;
    using UnityEngine;
    using EasyRoads3Dv3;

    public class ConfigurationLoader : MonoBehaviour
    {
        private const string configName = "config-big.json";

        public RoadConfig Config { get; set; }

        // Use this for initialization
        void Start() {
            string configPath = Path.Combine(Application.streamingAssetsPath, configName);
            EasyRoadsGenerator generator = GetComponent<EasyRoadsGenerator>();


            using (StreamReader r = new StreamReader(configPath))
            {
                string json = r.ReadToEnd();
                Debug.Log("Config loaded: " + json);
                this.Config = JsonUtility.FromJson<RoadConfig>(json);

                // Set the number of lanes
                generator.numberOfTracks = this.Config.NumberOfTracks;
                generator.SetUpRoadType();

                foreach (RoadPartConfig roadPartConfig in Config.RoadItems)
                {
                    switch(roadPartConfig.Type)
                    {
                        case RoadPartType.Straight:
                            generator.CreateStraight(roadPartConfig.Length, roadPartConfig.MinCars, roadPartConfig.MaxCars, roadPartConfig.HeightDifference, roadPartConfig.Seed);
                            break;
                        case RoadPartType.Curve:
                            generator.CreateCurve(roadPartConfig.Angle, roadPartConfig.Length, roadPartConfig.HeightDifference, roadPartConfig.MinCars, roadPartConfig.MaxCars, roadPartConfig.Seed);
                            break;
                        default:
                            throw new Exception("Undefined Road Type '" + roadPartConfig.Type + "'");
                    }
                }
                for (int laneNumber = 1; laneNumber <= generator.numberOfTracks; laneNumber++){
                    foreach (CustomEasyRoad customEasyRoad in generator.customEasyRoads){
                        generator.CreateLane(customEasyRoad.Road, customEasyRoad.Type, laneNumber);
                    }
                }
            }

            generator.isSelfDriving = Config.IsSelfDriving;
            generator.carSpeed = Config.CarSpeed;
            generator.isGenerated = true;

            ScreenRecorder screenRecorder = Camera.main.GetComponent<ScreenRecorder>();
            screenRecorder.ObjectsToHide = GameObject.FindGameObjectsWithTag("ObjectToHide");
        }

        // Update is called once per frame
        void Update() {

        }
    }
}