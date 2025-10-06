async function getCurrentTabUrl() {
  let queryOptions = { active: true, lastFocusedWindow: true };
  let [tab] = await chrome.tabs.query(queryOptions);
  return tab.url;
}

async function checkYouTubeUrl() {
  const urlInput = await getCurrentTabUrl();
  const messageDiv = document.getElementById('message');
  const total_div = document.getElementById('total_comments');
  const pattern = /https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)/;

  // adding event listener to positive negative neutral boxes so that clicking them filters comments
  // for positive box
  
  const match = urlInput.match(pattern);
  if (match && match[1]) {
    const videoId = match[1];
    let prediction = [];
    let total_comments = -1;
    async function getPrediction(videoId) {
      try {
        const res = await fetch(`http://127.0.0.1:8000/predict?video_id=${videoId}`);
        const data = await res.json(); // parse once

        // Extract fields
        const comments = data['comments'];
        const tot = data['total_comments'];

        // Assign to your variables
        prediction = comments;
        total_comments = tot;

      } catch (err) {
        console.error(err);
      }
    }

    // Usage
    async function main() {
      await getPrediction(videoId); // wait for fetch
      getPrediction(videoId);
      total_div.textContent = `Total Comments: ${total_comments}`;
      messageDiv.style.color = '#34d399';
      // console.log(prediction); // you can use it here safely
      const container = document.getElementById("comments-container");

      const counts = { positive: 0, negative: 0, neutral: 0 };

      prediction.forEach(comment => {
        const status = comment.status.toLowerCase();
        if (counts.hasOwnProperty(status)) {
          counts[status]++;
        }
      });

      function updateSentimentStats(positive, negative, neutral, total) {
        // Update text
        document.getElementById("positive-percent").textContent = Math.round(positive) + "%";
        document.getElementById("negative-percent").textContent = Math.round(negative) + "%";
        document.getElementById("neutral-percent").textContent = Math.round(neutral) + "%";

        // Update progress bars
        document.getElementById("positive-fill").style.width = positive + "%";
        document.getElementById("negative-fill").style.width = negative + "%";
        document.getElementById("neutral-fill").style.width = neutral + "%";

        // Update counts (optional)
        document.getElementById("positive-count").textContent = `${Math.round(total * positive / 100)} / ${total}`;
        document.getElementById("negative-count").textContent = `${Math.round(total * negative / 100)} / ${total}`;
        document.getElementById("neutral-count").textContent = `${Math.round(total * neutral / 100)} / ${total}`;
      }

      // Example usage:
      document.getElementById("stats-content").style.display = "block";
      updateSentimentStats(counts.positive * 100 / total_comments, counts.negative * 100 / total_comments, counts.neutral * 100 / total_comments, total_comments);

      prediction.forEach(c => {
        const div = document.createElement("div");

        div.classList.add("comment-box");
        if (c.status === "positive") div.classList.add("positive");
        else if (c.status === "negative") div.classList.add("negative");
        else if (c.status === "neutral") div.classList.add("neutral");

        const textDiv = document.createElement("div");
        textDiv.classList.add("comment-text");
        textDiv.textContent = c.text;

        const statusDiv = document.createElement("div");
        statusDiv.classList.add("comment-status", c.status);
        statusDiv.textContent = `${c.status.charAt(0).toUpperCase() + c.status.slice(1)}`;


        div.appendChild(textDiv);
        div.appendChild(statusDiv);
        container.appendChild(div);
      });
    }
    main();
    setTimeout(() => {
        setupFilterListeners();
      }, 100);
    
  }
  else {
    messageDiv.style.color = '#f87171'; // red for invalid
    messageDiv.textContent = 'Not a valid YouTube URL.';
    return null;
  }
}

// Separate function for event listeners
function setupFilterListeners() {
  const positiveBox = document.querySelector('.PositiveBox');
  const negativeBox = document.querySelector('.NegativeBox');
  const neutralBox = document.querySelector('.NeutralBox');
  const allCommentsBox = document.querySelector('.allComments');
  
  // Add cursor pointer style
  positiveBox.style.cursor = 'pointer';
  negativeBox.style.cursor = 'pointer';
  neutralBox.style.cursor = 'pointer';
  allCommentsBox.style.cursor = 'pointer';

  // Filter positive comments
  positiveBox.addEventListener('click', () => {
    const comments = document.querySelectorAll('.comment-box');
    comments.forEach(comment => {
      if (comment.classList.contains('positive')) {
        comment.style.display = 'flex';
      } else {
        comment.style.display = 'none';
      }
    });
  });

  // Filter negative comments
  negativeBox.addEventListener('click', () => {
    const comments = document.querySelectorAll('.comment-box');
    comments.forEach(comment => {
      if (comment.classList.contains('negative')) {
        comment.style.display = 'flex';
      } else {
        comment.style.display = 'none';
      }
    });
  });

  // Filter neutral comments
  neutralBox.addEventListener('click', () => {
    const comments = document.querySelectorAll('.comment-box');
    comments.forEach(comment => {
      if (comment.classList.contains('neutral')) {
        comment.style.display = 'flex';
      } else {
        comment.style.display = 'none';
      }
    });
  });

  // Show all comments
  allCommentsBox.addEventListener('click', () => {
    const comments = document.querySelectorAll('.comment-box');
    comments.forEach(comment => {
      comment.style.display = 'flex';
    });
  });
}


  
window.onload = () => {
  checkYouTubeUrl();
};
// Run when extension popup loads
// window.addEventListener('load', () => {
// });