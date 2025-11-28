chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "analyze") {
      let url = message.url;
      // TODO: Call ML model service
      sendResponse({result: "Safe ✅ (placeholder)"});
    }
  });
  