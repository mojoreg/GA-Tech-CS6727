document.addEventListener("DOMContentLoaded", () => {
  const statusBox = document.getElementById("status-box");
  const resultText = document.getElementById("result-text");
  const probabilityText = document.getElementById("probability-text");

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const currentUrl = tabs[0].url;

    // Send to Flask backend
    fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: currentUrl }),
    })
      .then((response) => response.json())
      .then((data) => {
        let pred = data.prediction;
        let prob = Math.round(data.probability * 100);
        probabilityText.textContent = `Phishing Probability: ${prob}%`;

        if (pred === 0 && prob < 60) {
          statusBox.className = "safe";
          resultText.textContent = "✅ SAFE";
        } else if (prob >= 60 && prob < 80) {
          statusBox.className = "suspicious";
          resultText.textContent = "⚠️ SUSPICIOUS";
        } else {
          statusBox.className = "phishing";
          resultText.textContent = "🚨 PHISHING";
        }
      })
      .catch((error) => {
        console.error("Error:", error);
        statusBox.className = "neutral";
        resultText.textContent = "Error connecting to detection server";
      });
  });
});
