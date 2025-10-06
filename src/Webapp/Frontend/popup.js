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

  const match = urlInput.match(pattern);
  if (!match || !match[1]) {
    messageDiv.style.color = '#f87171';
    messageDiv.textContent = 'Not a valid YouTube URL.';
    return;
  }

  const videoId = match[1];
  let prediction = [];
  let total_comments = -1;

  async function getPrediction(videoId) {
    try {
      const res = await fetch(`http://127.0.0.1:8000/predict?video_id=${videoId}`);
      const data = await res.json();
      prediction = data['comments'];
      total_comments = data['total_comments'];
    } catch (err) {
      console.error("Error fetching predictions:", err);
    }
  }

  async function main() {
    await getPrediction(videoId);

    total_div.textContent = `Total Comments: ${total_comments}`;
    messageDiv.style.color = '#34d399';

    const container = document.getElementById("comments-container");
    container.innerHTML = "";

    const counts = { positive: 0, negative: 0, neutral: 0 };

    prediction.forEach(comment => {
      const status = comment.status.toLowerCase();
      if (counts.hasOwnProperty(status)) counts[status]++;

      const div = document.createElement("div");
      div.classList.add("comment-box", status);

      const textDiv = document.createElement("div");
      textDiv.classList.add("comment-text");
      textDiv.textContent = comment.text;

      const statusDiv = document.createElement("div");
      statusDiv.classList.add("comment-status", status);
      statusDiv.textContent = status.charAt(0).toUpperCase() + status.slice(1);

      div.appendChild(textDiv);
      div.appendChild(statusDiv);
      container.appendChild(div);
    });

    document.getElementById("stats-content").style.display = "block";

    updateSentimentStats(
      (counts.positive * 100) / total_comments,
      (counts.negative * 100) / total_comments,
      (counts.neutral * 100) / total_comments,
      total_comments
    );

    setupFilterListeners();
  }

  function updateSentimentStats(positive, negative, neutral, total) {
    document.getElementById("positive-percent").textContent = Math.round(positive) + "%";
    document.getElementById("negative-percent").textContent = Math.round(negative) + "%";
    document.getElementById("neutral-percent").textContent = Math.round(neutral) + "%";

    document.getElementById("positive-fill").style.width = positive + "%";
    document.getElementById("negative-fill").style.width = negative + "%";
    document.getElementById("neutral-fill").style.width = neutral + "%";

    document.getElementById("positive-count").textContent = `${Math.round(total * positive / 100)} / ${total}`;
    document.getElementById("negative-count").textContent = `${Math.round(total * negative / 100)} / ${total}`;
    document.getElementById("neutral-count").textContent = `${Math.round(total * neutral / 100)} / ${total}`;
  }

  main();
}

function setupFilterListeners() {
  const positiveBox = document.querySelector('.PositiveBox');
  const negativeBox = document.querySelector('.NegativeBox');
  const neutralBox = document.querySelector('.NeutralBox');
  const allCommentsBox = document.querySelector('.allComments');

  if (!positiveBox || !negativeBox || !neutralBox || !allCommentsBox) {
    console.warn("Filter boxes not found in DOM");
    return;
  }

  [positiveBox, negativeBox, neutralBox, allCommentsBox].forEach(el => {
    el.style.cursor = 'pointer';
  });

  const filterComments = (filter) => {
    const comments = document.querySelectorAll('.comment-box');
    comments.forEach(comment => {
      if (filter === 'all' || comment.classList.contains(filter)) {
        comment.style.display = 'flex';
      } else {
        comment.style.display = 'none';
      }
    });
  };

  positiveBox.addEventListener('click', () => filterComments('positive'));
  negativeBox.addEventListener('click', () => filterComments('negative'));
  neutralBox.addEventListener('click', () => filterComments('neutral'));
  allCommentsBox.addEventListener('click', () => filterComments('all'));
}

window.onload = () => {
  checkYouTubeUrl();
};
