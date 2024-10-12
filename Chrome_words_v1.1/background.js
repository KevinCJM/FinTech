// background.js

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
              const translateApiUrl =
                "https://api.niutrans.com/NiuTransServer/translation";
              const translateParams = new URLSearchParams();
              translateParams.append("from", "en");
              translateParams.append("to", "zh");
              translateParams.append(
                "apikey",
                "ee47d395cb90b838c2db7293e173bac5"
              );
              translateParams.append("src_text", textToTranslate);

              fetch(translateApiUrl + "?" + translateParams.toString())
                .then(res => res.json())
                .then(translateData => {
                  let wordTranslation = "翻译错误";
                  let sentenceTranslation = "翻译错误";

                  if (translateData.tgt_text) {
                    const translations = translateData.tgt_text.split("\n");
                    wordTranslation = translations[0];
                    sentenceTranslation = translations.slice(1).join("\n");
                  } else {
                    console.error("Translation API error:", translateData);
                  }

                  // 调用字典API获取音标和音频
                  const dictApiUrl = `https://api.dictionaryapi.dev/api/v2/entries/en/${encodeURIComponent(
                    selectedText
                  )}`;

                  fetch(dictApiUrl)
                    .then(res => res.json())
                    .then(dictData => {
                      let phonetic = "";
                      let audio = "";

                      if (Array.isArray(dictData) && dictData.length > 0) {
                        const entry = dictData[0];
                        if (entry.phonetics && entry.phonetics.length > 0) {
                          // 获取第一个有音标的项
                          const phoneticEntry = entry.phonetics.find(
                            p => p.text && p.text.length > 0
                          );
                          if (phoneticEntry) {
                            phonetic = phoneticEntry.text;
                          }

                          // 获取第一个有音频的项
                          const audioEntry = entry.phonetics.find(
                            p => p.audio && p.audio.length > 0
                          );
                          if (audioEntry) {
                            audio = audioEntry.audio;
                          }
                        }
                      } else {
                        console.error("Dictionary API error:", dictData);
                      }

                      // 保存到本地存储
                      chrome.storage.local.get({ words: [] }, data => {
                        const words = data.words;
                        words.push({
                          word: selectedText,
                          phonetic: phonetic,
                          audio: audio,
                          sentence: sentence,
                          translation: {
                            word: wordTranslation,
                            sentence: sentenceTranslation
                          },
                          nextReviewTime: Date.now()
                        });
                        chrome.storage.local.set({ words: words });
                      });
                    })
                    .catch(error => {
                      console.error("Error calling dictionary API:", error);
                      // 即使字典API出错，也保存单词，但音标和音频为空
                      chrome.storage.local.get({ words: [] }, data => {
                        const words = data.words;
                        words.push({
                          word: selectedText,
                          phonetic: "",
                          audio: "",
                          sentence: sentence,
                          translation: {
                            word: wordTranslation,
                            sentence: sentenceTranslation
                          },
                          nextReviewTime: Date.now()
                        });
                        chrome.storage.local.set({ words: words });
                      });
                    });
                })
                .catch(error => {
                  console.error("Error calling translation API:", error);
                  // 处理翻译API请求错误
                  chrome.storage.local.get({ words: [] }, data => {
                    const words = data.words;
                    words.push({
                      word: selectedText,
                      phonetic: "",
                      audio: "",
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
  