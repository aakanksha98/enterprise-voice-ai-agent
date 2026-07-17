const composer = document.querySelector("[data-composer]");
const messageInput = document.querySelector("[data-message-input]");
const sendButton = document.querySelector("[data-send]");
const characterCount = document.querySelector("[data-character-count]");
const conversationLog = document.querySelector("#conversation-log");
const sessionStatus = document.querySelector("[data-session-status]");
const sessionChannel = document.querySelector("[data-session-channel]");
const sessionBusiness = document.querySelector("[data-session-business]");
const sessionIntent = document.querySelector("[data-session-intent]");
const announcement = document.querySelector("[data-announcement]");
const resetButton = document.querySelector("[data-reset]");
const themeToggle = document.querySelector("[data-theme-toggle]");
const profileForm = document.querySelector("[data-profile-form]");
const businessTypeSelect = document.querySelector("[data-business-type]");
const businessNameInput = document.querySelector("[data-business-name]");
const quickActionButtons = document.querySelectorAll("[data-prompt]");
const voiceInputButton = document.querySelector("[data-voice-input]");
const speakLatestButton = document.querySelector("[data-speak-latest]");
const voiceStatus = document.querySelector("[data-voice-status]");
const contextBusiness = document.querySelector("[data-context-business]");
const contextService = document.querySelector("[data-context-service]");
const contextAppointment = document.querySelector("[data-context-appointment]");
const initialAssistantText = document.querySelector("[data-initial-assistant-text]");

const MAX_MESSAGE_LENGTH = Number(messageInput.maxLength);
const BUSINESS_PROFILES = {
  dental: {
    label: "Dental Clinic",
    defaultName: "BrightSmile Dental",
  },
  salon: {
    label: "Salon",
    defaultName: "Luxe Hair Studio",
  },
  auto_repair: {
    label: "Auto Repair Shop",
    defaultName: "TurboFix Garage",
  },
};
const THEME_STORAGE_KEY = "enterprise-voice-theme";
const SpeechRecognitionConstructor =
  window.SpeechRecognition || window.webkitSpeechRecognition;

let recognition = null;
let isListening = false;
let isRecognitionStarting = false;
let isRecognitionStopping = false;
let suppressRecognitionFeedback = false;
let recognitionBaseText = "";
let currentTranscript = "";
let recognitionErrorMessage = "";
let activeUtterance = null;
let isSubmitting = false;
let isProfileReady = false;
let requestGeneration = 0;
let sessionId = createSessionId();
let activeBusinessType = businessTypeSelect.value;
let activeBusinessName = BUSINESS_PROFILES[activeBusinessType].defaultName;

const supportsSpeechSynthesis =
  typeof window.speechSynthesis !== "undefined" &&
  typeof window.SpeechSynthesisUtterance === "function";

function applyTheme(theme) {
  const isDark = theme === "dark";
  document.body.classList.toggle("theme-dark", isDark);
  themeToggle.setAttribute("aria-pressed", String(isDark));
  themeToggle.textContent = isDark ? "Light mode" : "Dark mode";
}

function initializeTheme() {
  applyTheme(localStorage.getItem(THEME_STORAGE_KEY) === "dark" ? "dark" : "light");
}

function toggleTheme() {
  const nextTheme = document.body.classList.contains("theme-dark") ? "light" : "dark";
  localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
  applyTheme(nextTheme);
}

function resizeMessageInput() {
  messageInput.style.height = "auto";
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 144)}px`;
}

function updateComposerState() {
  const messageLength = messageInput.value.length;
  const hasMessage = messageInput.value.trim().length > 0;

  sendButton.disabled =
    !isProfileReady ||
    !hasMessage ||
    isListening ||
    isRecognitionStarting ||
    isSubmitting;
  characterCount.textContent = `${messageLength} / ${MAX_MESSAGE_LENGTH}`;
  characterCount.classList.toggle("is-near-limit", messageLength >= 450);
  resizeMessageInput();
}

function defaultVoiceStatus() {
  if (recognition && supportsSpeechSynthesis) {
    return "Voice ready";
  }

  if (recognition) {
    return "Voice input ready";
  }

  if (supportsSpeechSynthesis) {
    return "Read-aloud ready";
  }

  return "Voice unavailable";
}

function setVoiceStatus(message, isActive = false) {
  voiceStatus.textContent = message;
  voiceStatus.classList.toggle("is-active", isActive);
}

function updateVoiceControls() {
  voiceInputButton.disabled =
    !isProfileReady ||
    !recognition ||
    isRecognitionStarting ||
    isRecognitionStopping ||
    isSubmitting;
  speakLatestButton.disabled =
    !isProfileReady ||
    !supportsSpeechSynthesis ||
    isListening ||
    isRecognitionStarting ||
    isSubmitting;
}

function setListeningState(listening) {
  isListening = listening;
  voiceInputButton.classList.toggle("is-listening", listening);
  voiceInputButton.setAttribute("aria-pressed", String(listening));
  voiceInputButton.setAttribute(
    "aria-label",
    listening ? "Stop voice input" : "Start voice input",
  );
  voiceInputButton.title = listening ? "Stop voice input" : "Start voice input";
  updateComposerState();
  updateVoiceControls();
}

function formatCurrentTime() {
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date());
}

function selectedProfile() {
  return BUSINESS_PROFILES[businessTypeSelect.value] || BUSINESS_PROFILES.dental;
}

function cleanBusinessName(value, fallback) {
  const cleaned = value.replace(/[\u0000-\u001F\u007F]/g, " ").replace(/\s+/g, " ").trim();
  return (cleaned || fallback).slice(0, 80);
}

function profileSummary() {
  const profile = BUSINESS_PROFILES[activeBusinessType] || BUSINESS_PROFILES.dental;
  return `${activeBusinessName} (${profile.label})`;
}

function greetingForActiveProfile() {
  return `Hi! I'm Aster, the AI receptionist for ${activeBusinessName}. How can I help today?`;
}

function updateQuickActionState() {
  quickActionButtons.forEach((button) => {
    button.disabled = !isProfileReady || isSubmitting;
  });
}

function updateProfileDisplay() {
  if (!isProfileReady) {
    sessionBusiness.textContent = "Not selected";
    contextBusiness.textContent = "Not selected";
    initialAssistantText.textContent = "Choose a demo business profile to begin.";
    return;
  }

  const summary = profileSummary();
  sessionBusiness.textContent = activeBusinessName;
  contextBusiness.textContent = summary;
  initialAssistantText.textContent = greetingForActiveProfile();
}

function createUserMessage(message) {
  const article = document.createElement("article");
  const content = document.createElement("div");
  const metadata = document.createElement("div");
  const author = document.createElement("strong");
  const time = document.createElement("time");
  const bubble = document.createElement("div");
  const text = document.createElement("p");

  article.className = "message message--user";
  article.dataset.userMessage = "true";
  article.dataset.dynamicMessage = "true";
  content.className = "message__content";
  metadata.className = "message__meta";
  bubble.className = "message__bubble";
  author.textContent = "You";
  time.textContent = formatCurrentTime();
  text.textContent = message;

  metadata.append(author, time);
  bubble.append(text);
  content.append(metadata, bubble);
  article.append(content);

  return article;
}

function createAssistantMessage(message, isError = false) {
  const article = document.createElement("article");
  const avatar = document.createElement("span");
  const content = document.createElement("div");
  const metadata = document.createElement("div");
  const author = document.createElement("strong");
  const time = document.createElement("time");
  const bubble = document.createElement("div");
  const text = document.createElement("p");

  article.className = "message message--assistant";
  article.dataset.assistantMessage = "true";
  article.dataset.dynamicMessage = "true";
  article.classList.toggle("message--error", isError);
  avatar.className = "agent-avatar message__avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = "AI";
  content.className = "message__content";
  metadata.className = "message__meta";
  bubble.className = "message__bubble";
  author.textContent = "Aster";
  time.textContent = formatCurrentTime();
  text.textContent = message;

  metadata.append(author, time);
  bubble.append(text);
  content.append(metadata, bubble);
  article.append(avatar, content);

  return article;
}

function createSessionId() {
  if (typeof window.crypto?.randomUUID === "function") {
    return window.crypto.randomUUID();
  }

  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function formatIntent(intent) {
  if (!intent) {
    return "Not identified";
  }

  return intent
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function updateCollectedContext(response) {
  if (response.business_type && response.business_name) {
    activeBusinessType = response.business_type;
    activeBusinessName = response.business_name;
    updateProfileDisplay();
  }

  if (response.slots?.service) {
    contextService.textContent = response.slots.service;
  }

  if (response.active_appointment) {
    const appointment = response.active_appointment;
    contextService.textContent = appointment.service;
    contextAppointment.textContent = `${appointment.date} at ${appointment.time} (${formatIntent(appointment.status)})`;
  }
}

async function submitMessage() {
  const message = messageInput.value.trim();

  if (
    !isProfileReady ||
    !message ||
    isListening ||
    isRecognitionStarting ||
    isSubmitting
  ) {
    return;
  }

  const activeGeneration = ++requestGeneration;
  const activeSessionId = sessionId;
  conversationLog.append(createUserMessage(message));
  messageInput.value = "";
  isSubmitting = true;
  sessionStatus.textContent = "Processing";
  sessionIntent.textContent = "Awaiting classification";
  announcement.textContent = "Request sent to the assistant.";
  updateComposerState();
  updateQuickActionState();
  conversationLog.scrollTo({ top: conversationLog.scrollHeight, behavior: "smooth" });

  try {
    const response = await fetch("/api/v1/conversation", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        session_id: activeSessionId,
        business_type: activeBusinessType,
        business_name: activeBusinessName,
      }),
    });

    if (!response.ok) {
      throw new Error(`Conversation request failed with status ${response.status}`);
    }

    const payload = await response.json();
    if (activeGeneration !== requestGeneration || activeSessionId !== sessionId) {
      return;
    }

    conversationLog.append(createAssistantMessage(payload.response));
    sessionStatus.textContent = "Response ready";
    sessionIntent.textContent = formatIntent(payload.intent);
    updateCollectedContext(payload);
    announcement.textContent = "Assistant response received.";
  } catch {
    if (activeGeneration !== requestGeneration) {
      return;
    }

    conversationLog.append(
      createAssistantMessage(
        "I'm sorry, the assistant is temporarily unavailable. Please try again.",
        true,
      ),
    );
    sessionStatus.textContent = "Service unavailable";
    sessionIntent.textContent = "Not identified";
    announcement.textContent = "The assistant could not complete the request.";
  } finally {
    if (activeGeneration === requestGeneration) {
      isSubmitting = false;
      updateComposerState();
      updateVoiceControls();
      updateQuickActionState();
      conversationLog.scrollTo({
        top: conversationLog.scrollHeight,
        behavior: "smooth",
      });
      messageInput.focus();
    }
  }
}

function cancelSpeech() {
  if (!supportsSpeechSynthesis) {
    return;
  }

  activeUtterance = null;
  window.speechSynthesis.cancel();
}

function cancelVoiceActivity() {
  if (recognition && (isListening || isRecognitionStarting)) {
    suppressRecognitionFeedback = true;

    try {
      recognition.abort();
    } catch {
      suppressRecognitionFeedback = false;
    }
  }

  isRecognitionStarting = false;
  isRecognitionStopping = false;
  cancelSpeech();
  setListeningState(false);
  setVoiceStatus(defaultVoiceStatus());
}

function resetConversation({ announce = true } = {}) {
  cancelVoiceActivity();
  requestGeneration += 1;
  isSubmitting = false;
  sessionId = createSessionId();
  document
    .querySelectorAll("[data-dynamic-message]")
    .forEach((message) => message.remove());
  messageInput.value = "";
  sessionStatus.textContent = isProfileReady ? "Ready" : "Setup required";
  sessionChannel.textContent = "Text";
  sessionIntent.textContent = "Not identified";
  contextService.textContent = "Not selected";
  contextAppointment.textContent = "No details";
  updateProfileDisplay();
  announcement.textContent = announce
    ? "Conversation reset."
    : `Demo profile set to ${profileSummary()}.`;
  updateComposerState();
  updateVoiceControls();
  updateQuickActionState();
  conversationLog.scrollTo({ top: 0, behavior: "smooth" });
  if (isProfileReady) {
    messageInput.focus();
  } else {
    businessNameInput.focus();
  }
}

function updateBusinessNameDefault() {
  const profile = selectedProfile();
  const currentName = businessNameInput.value.trim();
  const isDefaultName = Object.values(BUSINESS_PROFILES).some(
    (businessProfile) => businessProfile.defaultName === currentName,
  );

  businessNameInput.placeholder = profile.defaultName;
  if (!currentName || isDefaultName) {
    businessNameInput.value = profile.defaultName;
  }
}

function applyBusinessProfile(event) {
  event.preventDefault();

  const profile = selectedProfile();
  activeBusinessType = businessTypeSelect.value;
  activeBusinessName = cleanBusinessName(businessNameInput.value, profile.defaultName);
  businessNameInput.value = activeBusinessName;
  isProfileReady = true;
  resetConversation({ announce: false });
}

function combineTranscript(baseText, transcript) {
  return [baseText, transcript]
    .filter(Boolean)
    .join(" ")
    .slice(0, MAX_MESSAGE_LENGTH);
}

function recognitionErrorText(error) {
  const messages = {
    "audio-capture": "Microphone unavailable",
    network: "Voice service unavailable",
    "no-speech": "No speech detected",
    "not-allowed": "Microphone access denied",
    "service-not-allowed": "Voice service blocked",
  };

  return messages[error] || "Voice input failed";
}

function configureSpeechRecognition() {
  if (!SpeechRecognitionConstructor) {
    voiceInputButton.title = "Voice input is not supported in this browser";
    voiceInputButton.setAttribute(
      "aria-label",
      "Voice input is not supported in this browser",
    );
    return;
  }

  try {
    recognition = new SpeechRecognitionConstructor();
  } catch {
    recognition = null;
    voiceInputButton.title = "Voice input could not be initialized";
    voiceInputButton.setAttribute(
      "aria-label",
      "Voice input could not be initialized",
    );
    return;
  }

  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = "en-US";
  recognition.maxAlternatives = 1;

  recognition.addEventListener("start", () => {
    isRecognitionStarting = false;
    isRecognitionStopping = false;
    recognitionErrorMessage = "";
    setListeningState(true);
    sessionChannel.textContent = "Voice";
    setVoiceStatus("Listening", true);
    announcement.textContent = "Voice input started.";
  });

  recognition.addEventListener("result", (event) => {
    const finalParts = [];
    const interimParts = [];

    for (let index = 0; index < event.results.length; index += 1) {
      const result = event.results[index];
      const transcript = result[0]?.transcript.trim();

      if (!transcript) {
        continue;
      }

      if (result.isFinal) {
        finalParts.push(transcript);
      } else {
        interimParts.push(transcript);
      }
    }

    currentTranscript = [...finalParts, ...interimParts].join(" ");
    messageInput.value = combineTranscript(recognitionBaseText, currentTranscript);
    updateComposerState();
    setVoiceStatus(finalParts.length > 0 ? "Transcript ready" : "Listening", true);
  });

  recognition.addEventListener("error", (event) => {
    isRecognitionStarting = false;
    recognitionErrorMessage = recognitionErrorText(event.error);

    if (suppressRecognitionFeedback && event.error === "aborted") {
      return;
    }

    setVoiceStatus(recognitionErrorMessage);
    announcement.textContent = `${recognitionErrorMessage}.`;
  });

  recognition.addEventListener("end", () => {
    const feedbackWasSuppressed = suppressRecognitionFeedback;

    suppressRecognitionFeedback = false;
    isRecognitionStarting = false;
    isRecognitionStopping = false;
    setListeningState(false);

    if (feedbackWasSuppressed) {
      setVoiceStatus(defaultVoiceStatus());
      return;
    }

    if (recognitionErrorMessage) {
      setVoiceStatus(recognitionErrorMessage);
      return;
    }

    if (currentTranscript) {
      sessionChannel.textContent = "Voice + text";
      setVoiceStatus("Transcript ready");
      announcement.textContent = "Voice transcript ready for review.";
      messageInput.focus();
      return;
    }

    setVoiceStatus("No speech detected");
  });
}

function toggleVoiceInput() {
  if (!recognition) {
    return;
  }

  if (isListening) {
    isRecognitionStopping = true;
    setVoiceStatus("Finishing transcript", true);
    updateVoiceControls();
    recognition.stop();
    return;
  }

  cancelSpeech();
  recognitionBaseText = messageInput.value.trim();
  currentTranscript = "";
  recognitionErrorMessage = "";
  suppressRecognitionFeedback = false;
  isRecognitionStarting = true;
  setVoiceStatus("Starting microphone", true);
  updateComposerState();
  updateVoiceControls();

  try {
    recognition.start();
  } catch {
    isRecognitionStarting = false;
    setVoiceStatus("Voice input is already active");
    updateComposerState();
    updateVoiceControls();
  }
}

function preferredVoice() {
  const voices = window.speechSynthesis.getVoices();

  return (
    voices.find((voice) => voice.lang.toLowerCase() === "en-us") ||
    voices.find((voice) => voice.lang.toLowerCase().startsWith("en")) ||
    null
  );
}

function speakLatestAssistantMessage() {
  if (!supportsSpeechSynthesis) {
    return;
  }

  const assistantMessages = document.querySelectorAll("[data-assistant-message]");
  const latestMessage = assistantMessages.item(assistantMessages.length - 1);
  const messageText = latestMessage
    ?.querySelector(".message__bubble")
    ?.textContent.trim();

  if (!messageText) {
    setVoiceStatus("No assistant response to read");
    return;
  }

  cancelSpeech();

  const utterance = new SpeechSynthesisUtterance(messageText);
  const voice = preferredVoice();

  utterance.lang = "en-US";
  utterance.rate = 1;
  utterance.pitch = 1;

  if (voice) {
    utterance.voice = voice;
  }

  activeUtterance = utterance;

  utterance.addEventListener("start", () => {
    if (activeUtterance === utterance) {
      setVoiceStatus("Speaking", true);
      announcement.textContent = "Reading the latest assistant response.";
    }
  });

  utterance.addEventListener("end", () => {
    if (activeUtterance === utterance) {
      activeUtterance = null;
      setVoiceStatus(defaultVoiceStatus());
    }
  });

  utterance.addEventListener("error", () => {
    if (activeUtterance === utterance) {
      activeUtterance = null;
      setVoiceStatus("Read-aloud failed");
    }
  });

  window.speechSynthesis.speak(utterance);
}

function initializeVoiceSupport() {
  configureSpeechRecognition();

  if (!supportsSpeechSynthesis) {
    speakLatestButton.title = "Read-aloud is not supported in this browser";
    speakLatestButton.setAttribute(
      "aria-label",
      "Read-aloud is not supported in this browser",
    );
  }

  setVoiceStatus(defaultVoiceStatus());
  updateVoiceControls();
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  submitMessage();
});

messageInput.addEventListener("input", updateComposerState);

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

quickActionButtons.forEach((button) => {
  button.addEventListener("click", () => {
    if (!isProfileReady) {
      return;
    }

    cancelVoiceActivity();
    messageInput.value = button.dataset.prompt;
    sessionChannel.textContent = "Text";
    updateComposerState();
    messageInput.focus();
  });
});

profileForm.addEventListener("submit", applyBusinessProfile);
businessTypeSelect.addEventListener("change", updateBusinessNameDefault);
voiceInputButton.addEventListener("click", toggleVoiceInput);
speakLatestButton.addEventListener("click", speakLatestAssistantMessage);
themeToggle.addEventListener("click", toggleTheme);
resetButton.addEventListener("click", () => resetConversation());

initializeTheme();
updateBusinessNameDefault();
updateProfileDisplay();
initializeVoiceSupport();
updateQuickActionState();
updateComposerState();
