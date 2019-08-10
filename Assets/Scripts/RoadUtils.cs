/// <summary>
/// Class for auxiliary functions.
/// </summary>
public static class RoadUtils
{
    /// <summary>
    /// Bring the slerp to a lane based on the number of tracks from the middle to the outer point.
    /// </summary>
    /// <param name="numberOfTracks">The number of tracks.</param>
    /// <param name="lane">The current track from the left(0 - 7).</param>
    /// <returns>The Slerp.</returns>
    public static float GetRoadSlerpByLane(int numberOfTracks, int lane)
    {
        // Depending on the number of tracks that set the slerp between center, 
        // outside (left or right markers) and correct track
        switch (numberOfTracks)
        {
            case 4:
                // Either outside (0.75) or inside (0.25)
                if (lane == 0 || lane == 3) return 0.75f;
                return 0.25f;
            case 6:
                // Either outside, middle or inside
                if (lane == 0 || lane == 5) return 0.825f;
                if (lane == 1 || lane == 4) return 0.5f;
                return 0.175f;
            case 8:
                // Either outside, left middle, right center, or inside
                if (lane == 0 || lane == 5) return 0.85f;
                if (lane == 1 || lane == 4) return 0.6f;
                if (lane == 2 || lane == 3) return 0.4f;
                return 0.15f;
            case 3:
                if (lane == 2 || lane == 3) return 0.7f;
                return 0.01f;
            case 2:
                return 0.5f;
            default:
                // Has only one track
                return 0.1f;
        }
    }
}
