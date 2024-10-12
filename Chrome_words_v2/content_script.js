// content_script.js

// 注入样式表
const styleElement = document.createElement("link");
styleElement.rel = "stylesheet";
styleElement.href = chrome.runtime.getURL("styles.css");
document.head.appendChild(styleElement);

// 处理选中文本的事件
let selectedText = "";
let tooltipIcon = null;

document.addEventListener("mouseup", (event) => {
  removeTooltip();

  selectedText = window.getSelection().toString().trim();

  if (selectedText && isEnglishWord(selectedText)) {
    const range = window.getSelection().getRangeAt(0);
    const rect = range.getBoundingClientRect();

    // 创建气泡图标
    tooltipIcon = document.createElement("img");
    tooltipIcon.src = chrome.runtime.getURL("icons/icon16.png");
    tooltipIcon.id = "tooltipIcon";
    tooltipIcon.style.position = "absolute";
    tooltipIcon.style.left = `${rect.right + window.scrollX + 5}px`;
    tooltipIcon.style.top = `${rect.top + window.scrollY - 10}px`;
    tooltipIcon.style.width = "16px";
    tooltipIcon.style.height = "16px";
    tooltipIcon.style.cursor = "pointer";
    tooltipIcon.style.zIndex = "1000000";

    document.body.appendChild(tooltipIcon);

    tooltipIcon.addEventListener("click", () => {
      showDialog(selectedText, rect);
    });
  }
});

// 点击页面其他地方时，移除气泡和对话框
document.addEventListener("mousedown", (event) => {
  if (
    !event.target.closest("#tooltipIcon") &&
    !event.target.closest("#wordDialog")
  ) {
    removeTooltip();
    removeDialog();
  }
});

function removeTooltip() {
  if (tooltipIcon) {
    tooltipIcon.remove();
    tooltipIcon = null;
  }
}

function removeDialog() {
  const existingDialog = document.getElementById("wordDialog");
  if (existingDialog) {
    existingDialog.remove();
  }
}

function isEnglishWord(word) {
  return /^[a-zA-Z]+$/.test(word);
}

function showDialog(word, rect) {
  removeDialog();

  // 创建对话框
  const dialog = document.createElement("div");
  dialog.id = "wordDialog";
  dialog.style.position = "absolute";
  dialog.style.left = `${rect.left + window.scrollX}px`;
  dialog.style.top = `${rect.bottom + window.scrollY + 5}px`;
  dialog.style.width = "350px";
  dialog.style.backgroundColor = "#fff";
  dialog.style.border = "1px solid #ccc";
  dialog.style.borderRadius = "5px";
  dialog.style.boxShadow = "0 2px 8px rgba(0,0,0,0.2)";
  dialog.style.padding = "10px";
  dialog.style.zIndex = "1000001";
  dialog.style.fontFamily = "Arial, sans-serif";
  dialog.style.color = "#000";

  // 加载内容
  dialog.innerHTML = `<p>加载中...</p>`;
  document.body.appendChild(dialog);

  // 获取单词信息
  getWordInfo(word)
    .then((wordInfo) => {
      dialog.innerHTML = `
        <div style="display: flex; align-items: center;">
          <strong style="font-size: 20px;">${word}</strong>
          ${
            wordInfo.phonetic
              ? `<span style="margin-left: 10px;">/${wordInfo.phonetic}/</span>`
              : ""
          }
          ${
            wordInfo.audio
              ? `<button id="dialogPlayAudio" style="margin-left: 10px; cursor: pointer;">🔊</button>`
              : ""
          }
          <button id="dialogAddWord" style="margin-left: auto;">加入单词本</button>
        </div>
        <div>中文意思：${wordInfo.translation.word}</div>
        <hr>
        <div><em>例句：</em> ${wordInfo.sentence}</div>
        <div><em>翻译：</em> ${wordInfo.translation.sentence}</div>
      `;

      if (wordInfo.audio) {
        const playButton = document.getElementById("dialogPlayAudio");
        playButton.addEventListener("click", () => {
          const audio = new Audio(wordInfo.audio);
          audio.play();
        });
      }

      const addButton = document.getElementById("dialogAddWord");
      addButton.addEventListener("click", () => {
        saveWord(wordInfo);
        addButton.disabled = true;
        addButton.textContent = "已加入";
      });
    })
    .catch((error) => {
      console.error(error);
      dialog.innerHTML = `<p>获取单词信息失败。</p>`;
    });
}

function getWordInfo(word) {
  return new Promise((resolve, reject) => {
    // 获取上下文句子
    const sentence = getSentenceContainingWord(word);

    // 调用新的单词翻译API
    const wordApiUrl = `https://dict-mobile.iciba.com/interface/index.php?c=word&m=getsuggest&nums=10&is_need_mean=1&word=${encodeURIComponent(
      word
    )}`;

    fetch(wordApiUrl)
      .then((res) => res.json())
      .then((wordData) => {
        let wordTranslation = "翻译错误";

        if (
          wordData.status === 1 &&
          wordData.message &&
          wordData.message.length > 0
        ) {
          wordTranslation = wordData.message[0].paraphrase || "翻译错误";
        } else {
          console.error("Word Translation API error:", wordData);
        }

        // 调用新的句子翻译API
        const sentenceApiUrl = `https://dict.youdao.com/jsonapi?q=${encodeURIComponent(
          sentence
        )}`;

        fetch(sentenceApiUrl)
          .then((res) => res.json())
          .then((sentenceData) => {
            let sentenceTranslation = "翻译错误";

            if (sentenceData.fanyi && sentenceData.fanyi.tran) {
              sentenceTranslation = sentenceData.fanyi.tran;
            } else {
              console.error("Sentence Translation API error:", sentenceData);
            }

            // 调用字典API获取音标和音频
            const dictApiUrl = `https://api.dictionaryapi.dev/api/v2/entries/en/${encodeURIComponent(
              word
            )}`;

            fetch(dictApiUrl)
              .then((res) => res.json())
              .then((dictData) => {
                let phonetic = "";
                let audio = "";

                if (Array.isArray(dictData) && dictData.length > 0) {
                  const entry = dictData[0];
                  if (entry.phonetics && entry.phonetics.length > 0) {
                    // 获取第一个有音标的项
                    const phoneticEntry = entry.phonetics.find(
                      (p) => p.text && p.text.length > 0
                    );
                    if (phoneticEntry) {
                      phonetic = phoneticEntry.text;
                    }

                    // 获取第一个有音频的项
                    const audioEntry = entry.phonetics.find(
                      (p) => p.audio && p.audio.length > 0
                    );
                    if (audioEntry) {
                      audio = audioEntry.audio;
                    }
                  }
                } else {
                  console.error("Dictionary API error:", dictData);
                }

                resolve({
                  word: word,
                  phonetic: phonetic,
                  audio: audio,
                  sentence: sentence,
                  translation: {
                    word: wordTranslation,
                    sentence: sentenceTranslation,
                  },
                });
              })
              .catch((error) => {
                console.error("Error calling dictionary API:", error);
                resolve({
                  word: word,
                  phonetic: "",
                  audio: "",
                  sentence: sentence,
                  translation: {
                    word: wordTranslation,
                    sentence: sentenceTranslation,
                  },
                });
              });
          })
          .catch((error) => {
            console.error("Error calling sentence translation API:", error);
            reject(error);
          });
      })
      .catch((error) => {
        console.error("Error calling word translation API:", error);
        reject(error);
      });
  });
}

function saveWord(wordInfo) {
  chrome.storage.local.get({ words: [] }, (data) => {
    const words = data.words;
    // 检查是否已经存在
    if (!words.find((w) => w.word === wordInfo.word)) {
      words.push({
        word: wordInfo.word,
        phonetic: wordInfo.phonetic,
        audio: wordInfo.audio,
        sentence: wordInfo.sentence,
        translation: wordInfo.translation,
        nextReviewTime: Date.now(),
      });
      chrome.storage.local.set({ words: words });
    }
  });
}

function getSentenceContainingWord(word) {
  const selection = window.getSelection();
  if (selection.rangeCount > 0) {
    const range = selection.getRangeAt(0);
    const container = range.startContainer;
    let textContent =
      container.textContent || container.parentNode.textContent;

    // 简单地返回包含单词的整句话
    const sentences = textContent.match(/[^\.!\?]+[\.!\?]+/g);
    if (sentences) {
      for (let s of sentences) {
        if (s.includes(word)) {
          return s.trim();
        }
      }
    }
    return textContent.trim();
  }
  return "";
}
