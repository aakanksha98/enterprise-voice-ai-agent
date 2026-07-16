const composer = document.querySelector("[data-composer]");
const messageInput = document.querySelector("[data-message-input]");
const sendButton = document.querySelector("[data-send]");
const characterCount = document.querySelector("[data-character-count]");
const conversationLog = document.querySelector("#conversation-log");
const sessionStatus = document.querySelector("[data-session-status]");
const sessionIntent = document.querySelector("[data-session-intent]");
const announcement = document.querySelector("[data-announcement]");
const resetButton = document.querySelector("[data-reset]");
const quickActionButtons = document.querySelectorAll("[data-prompt]");

const MAX_MESSAGE_LENGTH = Number(messageInput.maxLength);

function resizeMessageInput() {
  messageInput.style.height = "auto";
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 144)}px`;
}

function updateComposerState() {
  const messageLength = messageInput.value.length;
  const hasMessage = messageInput.value.trim().length > 0;

  sendButton.disabled = !hasMessage;
  characterCount.textContent = `${messageLength} / ${MAX_MESSAGE_LENGTH}`;
  characterCount.classList.toggle("is-near-limit", messageLength >= 450);
  resizeMessageInput();
}

function formatCurrentTime() {
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date());
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

function submitMessage() {
  const message = messageInput.value.trim();

  if (!message) {
    return;
  }

  conversationLog.append(createUserMessage(message));
  messageInput.value = "";
  sessionStatus.textContent = "Message received";
  sessionIntent.textContent = "Awaiting classification";
  announcement.textContent = "Message added to the conversation.";
  updateComposerState();
  conversationLog.scrollTo({ top: conversationLog.scrollHeight, behavior: "smooth" });
  messageInput.focus();
}

function resetConversation() {
  document.querySelectorAll("[data-user-message]").forEach((message) => message.remove());
  messageInput.value = "";
  sessionStatus.textContent = "Ready";
  sessionIntent.textContent = "Not identified";
  announcement.textContent = "Conversation reset.";
  updateComposerState();
  conversationLog.scrollTo({ top: 0, behavior: "smooth" });
  messageInput.focus();
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
    messageInput.value = button.dataset.prompt;
    updateComposerState();
    messageInput.focus();
  });
});

resetButton.addEventListener("click", resetConversation);

updateComposerState();
