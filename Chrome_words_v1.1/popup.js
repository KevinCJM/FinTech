// popup.js

const wordList = document.getElementById("wordList");
const searchInput = document.getElementById("searchInput");

function loadWords() {
  chrome.storage.local.get({ words: [] }, data => {
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
        <strong>${item.word}</strong> ${
        item.phonetic ? `/${item.phonetic}/` : ""
      } (${item.translation.word})
        ${
          item.audio
            ? `<button data-audio="${item.audio}" class="play-audio">🔊</button>`
            : ""
        }
        <br>
        例句：${item.sentence}<br>
        翻译：${item.translation.sentence}
        <button data-index="${index}" class="delete-word">删除</button>
      `;
      wordList.appendChild(li);
    }
  });
}

wordList.addEventListener("click", e => {
  if (e.target.classList.contains("delete-word")) {
    const index = e.target.getAttribute("data-index");
    chrome.storage.local.get({ words: [] }, data => {
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
  chrome.storage.local.get({ words: [] }, data => {
    displayWords(data.words);
  });
});

loadWords();
