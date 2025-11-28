// Example: redirect if page flagged as phishing
if (window.location.href.includes("suspicious-example.com")) {
    alert("⚠️ This site may be a phishing attempt!");
    window.stop();
  }
  