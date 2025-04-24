const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(express.static('public'));

// Get list of videos from pip_videos folder
app.get('/api/videos', (req, res) => {
  const videoDir = path.join(__dirname, 'pip_videos');
  
  fs.readdir(videoDir, (err, files) => {
    if (err) {
      console.error('Error reading video directory:', err);
      return res.status(500).json({ error: 'Failed to read videos' });
    }
    
    // Filter for video files
    const videoFiles = files.filter(file => {
      const ext = path.extname(file).toLowerCase();
      return ['.mp4', '.webm', '.mov', '.avi'].includes(ext);
    });
    
    res.json(videoFiles);
  });
});

// Save rating
app.post('/api/ratings', (req, res) => {
  const { videoName, rating } = req.body;
  
  if (!videoName || !rating) {
    return res.status(400).json({ error: 'Video name and rating are required' });
  }
  
  const ratingsFile = path.join(__dirname, 'ratings.json');
  
  // Read existing ratings or create new array
  let ratings = [];
  if (fs.existsSync(ratingsFile)) {
    try {
      const data = fs.readFileSync(ratingsFile, 'utf8');
      ratings = JSON.parse(data);
    } catch (err) {
      console.error('Error reading ratings file:', err);
    }
  }
  
  // Add new rating
  ratings.push({
    videoName,
    rating,
    timestamp: new Date().toISOString()
  });
  
  // Save updated ratings
  fs.writeFileSync(ratingsFile, JSON.stringify(ratings, null, 2), 'utf8');
  
  res.json({ success: true });
});

// Serve video files
app.use('/videos', express.static(path.join(__dirname, 'pip_videos')));

// Start server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
}); 