// popup.js

const wordList = document.getElementById("wordList");
const searchInput = document.getElementById("searchInput");

function loadWords() {
  chrome.storage.local.get({ words: [] }, (data) => {
    displayWords(data.words);
  });
}

function displayWords(words) {
  wordList.innerHTML = "";
  const searchTerm = searchInput.value.toLowerCase();
  words.forEach((item, index) => {
    if (item.word.toLowerCase().includes(searchTerm)) {
      const li = document.createElement("li");
      li.className = "wordItem";
      li.innerHTML = `
        <div style="display: flex; align-items: center;">
          <strong>${item.word}</strong> ${item.phonetic ? `/${item.phonetic}/` : ""}
          ${
            item.audio
              ? `<button data-audio="${item.audio}" class="play-audio">🔊</button>`
              : ""
          }
          <button data-index="${index}" class="delete-word" style="margin-left: auto;">删除</button>
        </div>
        <div>中文意思：${item.translation.word}</div>
        <div><em>例句：</em> ${item.sentence}</div>
        <div><em>翻译：</em> ${item.translation.sentence}</div>
      `;
      wordList.appendChild(li);
    }
  });
}

wordList.addEventListener("click", (e) => {
  if (e.target.classList.contains("delete-word")) {
    const index = e.target.getAttribute("data-index");
    chrome.storage.local.get({ words: [] }, (data) => {
      const words = data.words;
      words.splice(index, 1);
      chrome.storage.local.set({ words: words }, () => {
        loadWords();
      });
    });
  } else if (e.target.classList.contains("play-audio")) {
    const audioUrl = e.target.getAttribute("data-audio");
    const audio = new Audio(audioUrl);
    audio.play();
  }
});

searchInput.addEventListener("input", () => {
  chrome.storage.local.get({ words: [] }, (data) => {
    displayWords(data.words);
  });
});

loadWords();
