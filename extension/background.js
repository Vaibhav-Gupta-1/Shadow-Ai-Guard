chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "analyze_text") {
    // Forward the prompt to our running Python FastAPI server
    fetch("http://127.0.0.1:8000/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text: request.text })
    })
    .then(response => response.json())
    .then(data => sendResponse(data))
    .catch(error => {
      console.error("ShadowAI Guard Backend Error:", error);
      // If the server crashes, we let it pass for the demo so it doesn't break the browser
      sendResponse({ is_sensitive: false, reason: "Backend unreachable" });
    });

    return true; // Keep the message channel open for the async response
  }
});