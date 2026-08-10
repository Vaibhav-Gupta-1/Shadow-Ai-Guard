// A simple, isolated flag in memory that resets instantly after execution
let isAutomatedSubmission = false;

// Selector Matrix for Send Buttons across all target platforms
const SEND_BUTTON_SELECTORS = [
  'button[data-testid="send-button"]',
  'button[aria-label="Send message"]',
  'button[aria-label="Send prompt"]',
  'button[aria-label="Send Message"]',
  'mat-icon-button[aria-label="Send message"]',
  '.send-button-container button',
  'button[class*="send-button"]',
  'button[class*="SubmitButton"]',
  'button:has(svg)'
].join(', ');

// Safely extracts text from either a textarea or a contenteditable div
function getPromptText(element) {
  if (!element) return "";
  // If it's a contenteditable div, value doesn't exist; use textContent/innerText
  if (element.getAttribute('contenteditable') === 'true' || element.tagName !== 'TEXTAREA') {
    return element.textContent || element.innerText || "";
  }
  return element.value || "";
}

// Crucial function to force React to recognize updates across both Textareas and ContentEditable elements
function forceReactStateSync(element) {
  const text = getPromptText(element);
  
  if (element.tagName === 'TEXTAREA') {
    const valueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value')?.set;
    if (valueSetter) valueSetter.call(element, text);
  } else {
    // For contenteditable divs, force a text node update
    element.innerHTML = text;
  }
  
  // Dispatch events so React frameworks register the change state
  element.dispatchEvent(new Event('input', { bubbles: true }));
  element.dispatchEvent(new Event('change', { bubbles: true }));
}

// Helper function to handle the core security analysis and ultimate submission
function handleSecurityScan(activeEl, promptText) {
  if (!promptText.trim()) return;

  console.log("🔍 ShadowAI Guard: Intercepted prompt: ", promptText);

  chrome.runtime.sendMessage({ action: "analyze_text", text: promptText }, (response) => {
    // Fallback security if backend fails to respond or crashes
    if (!response) {
      console.error("❌ Backend server unreachable.");
      return;
    }

    if (response.is_sensitive) {
      showSecurityAlert(response.reason);
    } else {
      console.log("✅ ShadowAI Guard: Prompt passed corporate security filters. Submitting.");
      
      isAutomatedSubmission = true;
      forceReactStateSync(activeEl);

      const sendButton = document.querySelector(SEND_BUTTON_SELECTORS) || 
                         activeEl.closest('form, div.relative, [class*="input-area"], [class*="chat-input"]')?.querySelector('button');

      setTimeout(() => {
        if (sendButton) {
          sendButton.removeAttribute('disabled');
          sendButton.click();
        } else {
          const submitEvent = new KeyboardEvent('keydown', {
            key: 'Enter',
            code: 'Enter',
            keyCode: 13,
            which: 13,
            bubbles: true,
            cancelable: true
          });
          activeEl.dispatchEvent(submitEvent);
        }

        setTimeout(() => {
          isAutomatedSubmission = false;
        }, 50);
      }, 30);
    }
  });
}

// ─── TRIGGER A: INTERCEPT KEYBOARD (ENTER KEY) ───────────────────
document.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    if (isAutomatedSubmission) return;

    const activeEl = document.activeElement;
    if (activeEl && (activeEl.tagName === 'TEXTAREA' || activeEl.getAttribute('contenteditable') === 'true' || activeEl.closest('[contenteditable="true"]'))) {
      const targetEl = activeEl.getAttribute('contenteditable') === 'true' ? activeEl : (activeEl.closest('[contenteditable="true"]') || activeEl);
      const promptText = getPromptText(targetEl);
      
      if (!promptText.trim()) return;

      event.preventDefault();
      event.stopPropagation();
      
      handleSecurityScan(targetEl, promptText);
    }
  }
}, true);

// ─── TRIGGER B: INTERCEPT MOUSE CLICK (UNIVERSAL DETECTION) ──────
document.addEventListener('click', (event) => {
  if (isAutomatedSubmission) return;

  const activeEl = document.querySelector('textarea, [contenteditable="true"]');
  if (!activeEl) return;

  const clickedElement = event.target;
  const targetButton = clickedElement.closest(SEND_BUTTON_SELECTORS);
  const structuralButton = activeEl.closest('form, div.relative, [class*="input-area"], [class*="chat-input"]')?.querySelector('button');
  const isInsideStructuralButton = structuralButton && (clickedElement === structuralButton || structuralButton.contains(clickedElement));

  if (targetButton || isInsideStructuralButton) {
    const promptText = getPromptText(activeEl);
    if (!promptText.trim()) return;

    event.preventDefault();
    event.stopPropagation();

    handleSecurityScan(activeEl, promptText);
  }
}, true);

// Function to draw the security warning block directly on the screen
function showSecurityAlert(reason) {
  const alertDiv = document.createElement('div');
  alertDiv.style.position = 'fixed';
  alertDiv.style.top = '20px';
  alertDiv.style.right = '20px';
  alertDiv.style.backgroundColor = '#d32f2f';
  alertDiv.style.color = 'white';
  alertDiv.style.padding = '16px 24px';
  alertDiv.style.borderRadius = '8px';
  alertDiv.style.zIndex = '999999';
  alertDiv.style.fontFamily = 'Arial, sans-serif';
  alertDiv.style.boxShadow = '0 4px 12px rgba(0,0,0,0.3)';
  alertDiv.innerHTML = `<strong>⚠️ ShadowAI Guard Alert</strong><br>Submission Blocked: ${reason}`;
  
  document.body.appendChild(alertDiv);
  
  setTimeout(() => {
    alertDiv.remove();
  }, 6000);
}