# Shadow-Ai-Guard
Shadow AI Guard: the browser extension that intercepts prompts/paste/submit events in LLM chat UIs (ChatGPT, Claude, Gemini) via JS content scripts, then runs them through a Python backend with a regex engine (for credentials/API keys) and an NER engine (for PII/corporate secrets) before anything gets sent off.
