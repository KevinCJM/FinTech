// 创建右键菜单
chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
      id: "addToWordBook",
      title: "添加到单词本",
      contexts: ["selection"]
    });
  });
  
  // 监听右键菜单点击事件
  chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "addToWordBook") {
      const selectedText = info.selectionText.trim();
  
      // 首先在目标标签页中注入内容脚本
      chrome.scripting.executeScript(
        {
          target: { tabId: tab.id },
          files: ["content_script.js"]
        },
        () => {
          // 注入完成后，发送消息获取句子
          chrome.tabs.sendMessage(
            tab.id,
            { action: "getSentence", word: selectedText },
            response => {
              if (chrome.runtime.lastError) {
                console.error(
                  "Error sending message to content script:",
                  chrome.runtime.lastError.message
                );
                return;
              }
              if (response && response.sentence) {
                const sentence = response.sentence;
                const textToTranslate = selectedText + "\n" + sentence;
  
                // 调用翻译API
                const apiUrl =
                  "https://api.niutrans.com/NiuTransServer/translation";
                const params = new URLSearchParams();
                params.append("from", "en");
                params.append("to", "zh");
                params.append("apikey", "ee47d395cb90b838c2db7293e173bac5");
                params.append("src_text", textToTranslate);
  
                fetch(apiUrl + "?" + params.toString())
                  .then(res => res.json())
                  .then(data => {
                    if (data.tgt_text) {
                      const translations = data.tgt_text.split("\n");
                      const wordTranslation = translations[0];
                      const sentenceTranslation = translations
                        .slice(1)
                        .join("\n");
  
                      // 保存到本地存储
                      chrome.storage.local.get({ words: [] }, data => {
                        const words = data.words;
                        words.push({
                          word: selectedText,
                          sentence: sentence,
                          translation: {
                            word: wordTranslation,
                            sentence: sentenceTranslation
                          },
                          nextReviewTime: Date.now()
                        });
                        chrome.storage.local.set({ words: words });
                      });
                    } else {
                      console.error("Translation API error:", data);
                      // 处理翻译错误
                      chrome.storage.local.get({ words: [] }, data => {
                        const words = data.words;
                        words.push({
                          word: selectedText,
                          sentence: sentence,
                          translation: {
                            word: "翻译错误",
                            sentence: "翻译错误"
                          },
                          nextReviewTime: Date.now()
                        });
                        chrome.storage.local.set({ words: words });
                      });
                    }
                  })
                  .catch(error => {
                    console.error("Error calling translation API:", error);
                    // 处理请求错误
                    chrome.storage.local.get({ words: [] }, data => {
                      const words = data.words;
                      words.push({
                        word: selectedText,
                        sentence: sentence,
                        translation: {
                          word: "翻译错误",
                          sentence: "翻译错误"
                        },
                        nextReviewTime: Date.now()
                      });
                      chrome.storage.local.set({ words: words });
                    });
                  });
              }
            }
          );
        }
      );
    }
  });
  
  // 定时检查复习时间
  chrome.alarms.create("reviewAlarm", { periodInMinutes: 1 });
  
  chrome.alarms.onAlarm.addListener(alarm => {
    if (alarm.name === "reviewAlarm") {
      chrome.storage.local.get({ words: [] }, data => {
        const wordsToReview = data.words.filter(
          word => word.nextReviewTime <= Date.now()
        );
        if (wordsToReview.length > 0) {
          // 获取当前活动的标签页
          chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
            if (tabs.length > 0) {
              const activeTab = tabs[0];
              // 确保标签页是普通的网页
              if (activeTab.url && activeTab.url.startsWith("http")) {
                // 注入复习内容脚本
                chrome.scripting.executeScript({
                  target: { tabId: activeTab.id },
                  files: ["review_inject.js"]
                });
              } else {
                // 如果当前标签页不是普通网页，尝试找到其他标签页
                chrome.tabs.query({ url: "<all_urls>" }, tabs => {
                  for (let tab of tabs) {
                    if (tab.url && tab.url.startsWith("http")) {
                      chrome.scripting.executeScript({
                        target: { tabId: tab.id },
                        files: ["review_inject.js"]
                      });
                      break;
                    }
                  }
                });
              }
            } else {
              // 如果没有打开的标签页，创建一个新标签页
              chrome.tabs.create({ url: "https://www.google.com" }, tab => {
                chrome.scripting.executeScript({
                  target: { tabId: tab.id },
                  files: ["review_inject.js"]
                });
              });
            }
          });
        }
      });
    }
  });
  
  // 处理来自内容脚本的消息
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "getWordsToReview") {
      chrome.storage.local.get({ words: [] }, data => {
        const wordsToReview = data.words.filter(
          word => word.nextReviewTime <= Date.now()
        );
        sendResponse({ words: wordsToReview });
      });
      return true;
    } else if (request.action === "updateWord") {
      const updatedWord = request.word;
      chrome.storage.local.get({ words: [] }, data => {
        const words = data.words;
        const index = words.findIndex(w => w.word === updatedWord.word);
        if (index !== -1) {
          words[index] = updatedWord;
          chrome.storage.local.set({ words: words }, () => {
            sendResponse({ success: true });
          });
        } else {
          sendResponse({ success: false });
        }
      });
      return true;
    } else if (request.action === "removeWord") {
      const wordToRemove = request.word;
      chrome.storage.local.get({ words: [] }, data => {
        const words = data.words;
        const index = words.findIndex(w => w.word === wordToRemove.word);
        if (index !== -1) {
          words.splice(index, 1);
          chrome.storage.local.set({ words: words }, () => {
            sendResponse({ success: true });
          });
        } else {
          sendResponse({ success: false });
        }
      });
      return true;
    }
  });
  