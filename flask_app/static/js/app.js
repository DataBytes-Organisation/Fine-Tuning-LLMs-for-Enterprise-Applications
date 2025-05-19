const chatWindow = document.getElementById("chat-window");
const input = document.getElementById("user-text");
const sendBtn = document.getElementById("send-btn");
const toggleBtn = document.getElementById("toggle-mode");
const searchBtn = document.getElementById("search-btn");
const fileBtn = document.getElementById("file-btn");

function appendMessage(role, text) {
  const msg = document.createElement("div");
  msg.className = `message ${role}`;

  const label = document.createElement("div");
  label.className = "label";
  label.innerText = role === "user" ? "User" : "AI";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerText = text;

  msg.appendChild(label);
  msg.appendChild(bubble);
  chatWindow.appendChild(msg);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return msg;
}

// Send button logic
sendBtn.onclick = async () => {
  const text = input.value.trim();
  if (!text) return;

  appendMessage("user", text);
  input.value = "";
  const loadingMsg = appendMessage("ai", "...");

  try {
    const resp = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (loadingMsg && chatWindow.contains(loadingMsg)) {
      chatWindow.removeChild(loadingMsg);
    }

    if (!resp.ok) {
      const errorText = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${errorText}`);
    }

    const data = await resp.json();
    if (data.error) {
      appendMessage("ai", `Error: ${data.error}`);
    } else if (data.sentiment) {
      try {
        const rawHtml = marked.parse(data.sentiment);
        const msg = document.createElement("div");
        msg.className = "message ai";
        const label = document.createElement("div");
        label.className = "label";
        label.innerText = "AI";
        const bubble = document.createElement("div");
        bubble.className = "bubble";
        bubble.innerHTML = rawHtml;
        msg.appendChild(label);
        msg.appendChild(bubble);
        chatWindow.appendChild(msg);
        chatWindow.scrollTop = chatWindow.scrollHeight;
      } catch (parseError) {
        console.error("Parsing error:", parseError);
        appendMessage("ai", `Raw response: ${data.sentiment}`);
      }
    } else {
      appendMessage("ai", "Unexpected server response.");
    }
  } catch (error) {
    console.error(error);
    if (loadingMsg && chatWindow.contains(loadingMsg)) {
      chatWindow.removeChild(loadingMsg);
    }
    appendMessage("ai", `Error: ${error.message}`);
  }
};

// Simple web search
searchBtn.addEventListener("click", async () => {
  const query = prompt("Enter your search query:");
  if (query) {
    appendMessage("user", `🔍 Searching: ${query}`);
    const loadingMsg = appendMessage("ai", "...");
    try {
      const resp = await fetch("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      if (loadingMsg && chatWindow.contains(loadingMsg)) {
        chatWindow.removeChild(loadingMsg);
      }
      if (!resp.ok) {
        const errorText = await resp.text();
        throw new Error(`HTTP ${resp.status}: ${errorText}`);
      }
      const data = await resp.json();
      if (data.error) {
        appendMessage("ai", `Error: ${data.error}`);
      } else if (data.answer) {
        const rawHtml = marked.parse(data.answer);
        const msg = document.createElement("div");
        msg.className = "message ai";
        const label = document.createElement("div");
        label.className = "label";
        label.innerText = "AI";
        const bubble = document.createElement("div");
        bubble.className = "bubble";
        bubble.innerHTML = rawHtml;
        msg.appendChild(label);
        msg.appendChild(bubble);
        chatWindow.appendChild(msg);
        chatWindow.scrollTop = chatWindow.scrollHeight;
      } else {
        appendMessage("ai", "Unexpected server response.");
      }
    } catch (error) {
      if (loadingMsg && chatWindow.contains(loadingMsg)) {
        chatWindow.removeChild(loadingMsg);
      }
      appendMessage("ai", `Error: ${error.message}`);
    }
  }
});

// File button logic
fileBtn.addEventListener("click", () => {
  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = ".txt";
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;
    if (!file.name.endsWith(".txt")) {
      appendMessage("ai", "❗ Only .txt files are supported.");
      return;
    }
    appendMessage("user", `📎 Attached file: ${file.name}`);
    const formData = new FormData();
    formData.append("file", file);
    const loadingMsg = appendMessage("ai", "Analyzing file...");
    fetch("/upload", { method: "POST", body: formData })
      .then((res) => res.json())
      .then((data) => {
        if (loadingMsg && chatWindow.contains(loadingMsg)) {
          chatWindow.removeChild(loadingMsg);
        }
        if (data.error) {
          appendMessage("ai", `Error: ${data.error}`);
        } else if (data.sentiment) {
          const rawHtml = marked.parse(data.sentiment);
          const msg = document.createElement("div");
          msg.className = "message ai";
          const label = document.createElement("div");
          label.className = "label";
          label.innerText = "AI";
          const bubble = document.createElement("div");
          bubble.className = "bubble";
          bubble.innerHTML = rawHtml;
          msg.appendChild(label);
          msg.appendChild(bubble);
          chatWindow.appendChild(msg);
          chatWindow.scrollTop = chatWindow.scrollHeight;
        } else {
          appendMessage("ai", "Unexpected server response.");
        }
      })
      .catch((err) => {
        if (loadingMsg && chatWindow.contains(loadingMsg)) {
          chatWindow.removeChild(loadingMsg);
        }
        appendMessage("ai", "❗ File upload failed.");
      });
  });
  fileInput.click();
});

// Dark/Light Mode toggle
function setMode(mode) {
  document.body.className = mode;
  toggleBtn.innerText =
    mode === "light-mode" ? "Switch to Dark Mode" : "Switch to Light Mode";
  localStorage.setItem("chatMode", mode);
}

toggleBtn.onclick = () => {
  const current = document.body.className;
  setMode(current === "light-mode" ? "dark-mode" : "light-mode");
};

const savedMode = localStorage.getItem("chatMode") || "light-mode";
setMode(savedMode);

// Enter key to send
input.addEventListener("keypress", function (e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendBtn.click();
  }
});
