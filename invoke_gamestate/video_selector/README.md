# Video Selector

A simple web application that allows you to rate videos as "good" or "bad".

## Setup

1. Make sure you have Node.js installed on your system
2. Install dependencies:
   ```
   npm install
   ```
3. Place your video files in the `pip_videos` folder (supported formats: mp4, webm, mov, avi)
4. Start the application:
   ```
   npm start
   ```
5. Open your browser and navigate to `http://localhost:3000`

## Usage

- Videos will be displayed one by one
- Click the "Good" or "Bad" button to rate the current video
- After rating, it will automatically move to the next video
- You can also use the "Previous" and "Next" buttons to navigate between videos
- All ratings are saved to a `ratings.json` file in the root directory

## Results

The ratings are saved to a JSON file called `ratings.json` in the following format:

```json
[
  {
    "videoName": "video1.mp4",
    "rating": "good",
    "timestamp": "2023-08-15T12:34:56.789Z"
  },
  {
    "videoName": "video2.mp4",
    "rating": "bad",
    "timestamp": "2023-08-15T12:35:10.123Z"
  }
]
``` 