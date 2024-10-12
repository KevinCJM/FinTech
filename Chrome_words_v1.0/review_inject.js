// review_inject.js

(function() {
  // 创建复习提示窗口的容器
  const reviewDiv = document.createElement("div");
  reviewDiv.id = "reviewPrompt";
  reviewDiv.style.position = "fixed";
  reviewDiv.style.bottom = "20px";
  reviewDiv.style.right = "20px";
  reviewDiv.style.width = "300px";
  reviewDiv.style.backgroundColor = "#fff";
  reviewDiv.style.color = "#000"; // 设置文本颜色为黑色
  reviewDiv.style.border = "1px solid #ccc";
  reviewDiv.style.borderRadius = "5px";
  reviewDiv.style.boxShadow = "0 2px 8px rgba(0,0,0,0.2)";
  reviewDiv.style.padding = "10px";
  reviewDiv.style.zIndex = "1000000";
  reviewDiv.style.fontFamily = "Arial, sans-serif";

  // 关闭按钮
  const closeButton = document.createElement("button");
  closeButton.textContent = "×";
  closeButton.style.position = "absolute";
  closeButton.style.top = "5px";
  closeButton.style.right = "5px";
  closeButton.style.border = "none";
  closeButton.style.background = "transparent";
  closeButton.style.fontSize = "16px";
  closeButton.style.cursor = "pointer";
  closeButton.style.color = "#000"; // 设置关闭按钮文本颜色为黑色
  closeButton.addEventListener("click", () => {
    document.body.removeChild(reviewDiv);
  });
  reviewDiv.appendChild(closeButton);

  // 内容区域
  const contentDiv = document.createElement("div");
  contentDiv.style.marginTop = "20px";
  contentDiv.style.color = "#000"; // 设置内容区域文本颜色为黑色
  reviewDiv.appendChild(contentDiv);

  // 按钮
  const masteredButton = document.createElement("button");
  masteredButton.textContent = "已掌握";
  masteredButton.style.marginRight = "10px";
  masteredButton.style.color = "#000"; // 设置按钮文本颜色为黑色
  masteredButton.style.backgroundColor = "#e0e0e0"; // 设置按钮背景色
  masteredButton.style.border = "1px solid #ccc";
  masteredButton.style.padding = "5px 10px";
  masteredButton.style.cursor = "pointer";

  const skipButton = document.createElement("button");
  skipButton.textContent = "跳过";
  skipButton.style.color = "#000"; // 设置按钮文本颜色为黑色
  skipButton.style.backgroundColor = "#e0e0e0"; // 设置按钮背景色
  skipButton.style.border = "1px solid #ccc";
  skipButton.style.padding = "5px 10px";
  skipButton.style.cursor = "pointer";

  const buttonsDiv = document.createElement("div");
  buttonsDiv.style.marginTop = "10px";
  buttonsDiv.appendChild(masteredButton);
  buttonsDiv.appendChild(skipButton);
  reviewDiv.appendChild(buttonsDiv);

  document.body.appendChild(reviewDiv);

  // 获取需要复习的单词
  chrome.runtime.sendMessage({ action: "getWordsToReview" }, response => {
    if (response && response.words && response.words.length > 0) {
      let currentWordIndex = 0;
      let currentWord = response.words[currentWordIndex];
      displayWord(currentWord);

      masteredButton.addEventListener("click", () => {
        // 移除已掌握的单词
        chrome.runtime.sendMessage(
          { action: "removeWord", word: currentWord },
          response => {
            if (response.success) {
              nextWord();
            }
          }
        );
      });

      skipButton.addEventListener("click", () => {
        // 更新下次复习时间
        currentWord.nextReviewTime = Date.now() + getNextInterval(currentWord);
        chrome.runtime.sendMessage(
          { action: "updateWord", word: currentWord },
          response => {
            if (response.success) {
              nextWord();
            }
          }
        );
      });

      function nextWord() {
        currentWordIndex++;
        if (currentWordIndex < response.words.length) {
          currentWord = response.words[currentWordIndex];
          displayWord(currentWord);
        } else {
          // 没有更多需要复习的单词，关闭窗口
          document.body.removeChild(reviewDiv);
        }
      }

      function displayWord(word) {
        contentDiv.innerHTML = `
          <strong>${word.word}</strong> (${word.translation.word})<br>
          例句：${word.sentence}<br>
          翻译：${word.translation.sentence}
        `;
      }

      function getNextInterval(word) {
        // 返回下次复习间隔（毫秒）
        return 24 * 60 * 60 * 1000; // 1天
      }
    } else {
      // 没有需要复习的单词，关闭窗口
      document.body.removeChild(reviewDiv);
    }
  });
})();
