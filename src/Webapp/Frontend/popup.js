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
  let analysisData = null; // store stats & charts

  async function getPrediction(videoId) {
    try {
      const res = await fetch(`http://127.0.0.1:8000/predict?video_id=${videoId}`);
      const data = await res.json();
      prediction = data['comments'];
      total_comments = data['total_comments'];
      analysisData = data; // store for charts and stats
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
    updateAnalysisTab();
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

  function setupFilterListeners() {
    const positiveBox = document.querySelector('.PositiveBox');
    const negativeBox = document.querySelector('.NegativeBox');
    const neutralBox = document.querySelector('.NeutralBox');
    const allCommentsBox = document.querySelector('.allComments');

    if (!positiveBox || !negativeBox || !neutralBox || !allCommentsBox) return;

    [positiveBox, negativeBox, neutralBox, allCommentsBox].forEach(el => el.style.cursor = 'pointer');

    const filterComments = (filter) => {
      const comments = document.querySelectorAll('.comment-box');
      comments.forEach(comment => {
        comment.style.display = (filter === 'all' || comment.classList.contains(filter)) ? 'flex' : 'none';
      });
    };

    positiveBox.addEventListener('click', () => filterComments('positive'));
    negativeBox.addEventListener('click', () => filterComments('negative'));
    neutralBox.addEventListener('click', () => filterComments('neutral'));
    allCommentsBox.addEventListener('click', () => filterComments('all'));
  }

  function updateAnalysisTab() {
    if (!analysisData) return;

    // Update stats
    document.getElementById("avg_word_count").textContent = analysisData.avg_word_count;
    document.getElementById("avg_pos_word_count").textContent = analysisData.avg_pos_word_count;
    document.getElementById("avg_neg_word_count").textContent = analysisData.avg_neg_word_count;
    document.getElementById("avg_neu_word_count").textContent = analysisData.avg_neu_word_count;

    // Update charts (Base64 images)
    const chartIds = ['pie_chart', 'trend_chart', 'wordcloud_neg', 'wordcloud_neu', 'wordcloud_pos'];
    const chartKeys = ['pie_chart', 'trend_chart', 'wordcloud_neg', 'wordcloud_neu', 'wordcloud_pos'];
    const chartTitles = [
    "Sentiment Distribution (Pie Chart)",
    "Sentiment Trend Over Time",
    "Negative Words Wordcloud",
    "Neutral Words Wordcloud",
    "Positive Words Wordcloud"
  ];
    chartIds.forEach((id, idx) => {
      const el = document.getElementById(id);
      el.innerHTML = ""; // clear previous
      const title = document.createElement("div");
    title.classList.add("chart-title");
    title.textContent = chartTitles[idx];
    el.appendChild(title)

    const img = document.createElement("img");
      img.src = `data:image/png;base64,${analysisData[chartKeys[idx]]}`;
      img.style.width = "100%";
      img.style.height = "100%";
      img.style.objectFit = "contain";
      el.appendChild(img);
    });
  }

  main();
}

// Tab switching
window.onload = () => {
  const tabButtons = document.querySelectorAll('.tab-button');
  const tabContents = document.querySelectorAll('.tab-content');

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;

      tabContents.forEach(tc => tc.style.display = 'none');
      tabButtons.forEach(b => b.classList.remove('active'));

      document.getElementById(target).style.display = 'block';
      btn.classList.add('active');
    });
  });

  checkYouTubeUrl();
};
