namespace Assets.Scripts.Configuration
{
    using System.Collections.Generic;
    using System;

    /// <summary>
    /// Class represents the configuration of a route.
    /// </summary>
    [Serializable]
    public class RoadConfig
    {
        /// <summary>
        /// The number of lanes on the road (Current options: 2, 4, 6, 8).
        /// </summary>
        public int NumberOfTracks;

        /// <summary>
        /// The speed of the car in Km / h.
        /// </summary>
        public int CarSpeed;

        /// <summary>
        /// Whether the user wants to control the car himself.
        /// </summary>
        public bool IsSelfDriving;

        /// <summary>
        /// The optional seed, if the same route should be generated again.
        /// </summary>
        public string Seed;

        /// <summary>
        /// The sections of the route.
        /// </summary>
        public List<RoadPartConfig> RoadItems;
    }
}