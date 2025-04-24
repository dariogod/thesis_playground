document.addEventListener('DOMContentLoaded', () => {
  // DOM elements
  const videoElement = document.getElementById('current-video');
  const loadingMessage = document.getElementById('loading-message');
  const currentIndexElement = document.getElementById('current-index');
  const totalVideosElement = document.getElementById('total-videos');
  const videoNameElement = document.getElementById('video-name');
  const goodBtn = document.getElementById('good-btn');
  const badBtn = document.getElementById('bad-btn');
  const prevBtn = document.getElementById('prev-btn');
  const nextBtn = document.getElementById('next-btn');
  const statusMessage = document.getElementById('status-message');

  // State
  let videos = [];
  let currentIndex = 0;
  let ratings = {};

  // Show loading message
  function showLoading(show) {
    loadingMessage.style.display = show ? 'block' : 'none';
  }

  // Update status message
  function showStatus(message, isError = false) {
    statusMessage.textContent = message;
    statusMessage.className = isError ? 'error' : 'success';
    
    // Clear message after 3 seconds
    setTimeout(() => {
      statusMessage.textContent = '';
      statusMessage.className = '';
    }, 3000);
  }

  // Fetch videos from the API
  async function fetchVideos() {
    showLoading(true);
    
    try {
      const response = await fetch('/api/videos');
      if (!response.ok) {
        throw new Error('Failed to fetch videos');
      }
      
      videos = await response.json();
      
      if (videos.length === 0) {
        showStatus('No videos found in the pip_videos folder', true);
        return;
      }
      
      totalVideosElement.textContent = videos.length;
      loadVideo(0);
    } catch (error) {
      console.error('Error fetching videos:', error);
      showStatus(`Error: ${error.message}`, true);
    } finally {
      showLoading(false);
    }
  }

  // Load video at specified index
  function loadVideo(index) {
    if (index < 0 || index >= videos.length) {
      return;
    }
    
    currentIndex = index;
    const videoName = videos[index];
    
    // Update UI
    currentIndexElement.textContent = index + 1;
    videoNameElement.textContent = videoName;
    
    // Load video
    videoElement.src = `/videos/${videoName}`;
    videoElement.load();
    
    // Update button states
    prevBtn.disabled = index === 0;
    nextBtn.disabled = index === videos.length - 1;
    
    // Check if this video has been rated
    if (ratings[videoName]) {
      showStatus(`This video was previously rated as "${ratings[videoName]}"`);
    }
  }

  // Save rating
  async function saveRating(videoName, rating) {
    try {
      const response = await fetch('/api/ratings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ videoName, rating })
      });
      
      if (!response.ok) {
        throw new Error('Failed to save rating');
      }
      
      // Store rating in local state
      ratings[videoName] = rating;
      
      // Show success message
      showStatus(`Video rated as "${rating}"`);
      
      // Automatically go to next video if not at the end
      if (currentIndex < videos.length - 1) {
        loadVideo(currentIndex + 1);
      }
    } catch (error) {
      console.error('Error saving rating:', error);
      showStatus(`Error: ${error.message}`, true);
    }
  }

  // Event listeners
  goodBtn.addEventListener('click', () => {
    const videoName = videos[currentIndex];
    saveRating(videoName, 'good');
  });
  
  badBtn.addEventListener('click', () => {
    const videoName = videos[currentIndex];
    saveRating(videoName, 'bad');
  });
  
  prevBtn.addEventListener('click', () => {
    loadVideo(currentIndex - 1);
  });
  
  nextBtn.addEventListener('click', () => {
    loadVideo(currentIndex + 1);
  });
  
  // Initialize app
  fetchVideos();
}); 