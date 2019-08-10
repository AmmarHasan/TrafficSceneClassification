namespace Assets.Scripts.Configuration
{
    using System;

    /// <summary>
    /// Class represents a part of a route.
    /// </summary>
    [Serializable]
    public class RoadPartConfig
    {
        /// <summary>
        /// The length of the track part.
        /// </summary>
        public float Length;

        /// <summary>
        /// The type of track (0 = straight line, 1 = curve);
        /// </summary>
        public RoadPartType Type;

        /// <summary>
        /// The minimum number of cars on this section of the route.
        /// </summary>
        public int MinCars;

        /// <summary>
        /// The maximum number of cars on this section of the route.
        /// </summary>
        public int MaxCars;

        /// <summary>
        /// The optional height difference between start and end.
        /// </summary>
        public float HeightDifference;

        /// <summary>
        /// The optional angle if there is a curve.
        /// </summary>
        public int Angle;

        /// <summary>
        /// The optional seed, if the same route should be used again.
        /// </summary>
        public string Seed;
    }
}