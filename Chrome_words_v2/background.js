// background.js

// 创建右键菜单（可选，如果不需要右键菜单，可以移除）
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "addToWordBook",
    title: "添加到单词本",
    contexts: ["selection"]
  });
});

// 监听右键菜单点击事件（可选）
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "addToWordBook") {
    // 可以考虑调用 content_script.js 中的逻辑
    // 但由于我们现在通过气泡图标添加单词，本部分可以省略或保留作为备用
  }
});

// 定时检查复习时间
chrome.alarms.create("reviewAlarm", { periodInMinutes: 1 });

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === "reviewAlarm") {
    chrome.storage.local.get({ words: [] }, (data) => {
      const wordsToReview = data.words.filter(
        (word) => word.nextReviewTime <= Date.now()
      );
      if (wordsToReview.length > 0) {
        // 获取当前活动的标签页
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
          if (tabs.length > 0) {
            const activeTab = tabs[0];
            // 确保标签页是普通的网页
            if (activeTab.url && activeTab.url.startsWith("http")) {
              // 注入复习内容脚本
              chrome.scripting.executeScript({
                target: { tabId: activeTab.id },
                files: ["review_inject.js"],
              });
            } else {
              // 如果当前标签页不是普通网页，尝试找到其他标签页
              chrome.tabs.query({ url: "<all_urls>" }, (tabs) => {
                for (let tab of tabs) {
                  if (tab.url && tab.url.startsWith("http")) {
                    chrome.scripting.executeScript({
                      target: { tabId: tab.id },
                      files: ["review_inject.js"],
                    });
                    break;
                  }
                }
              });
            }
          } else {
            // 如果没有打开的标签页，创建一个新标签页
            chrome.tabs.create({ url: "https://www.google.com" }, (tab) => {
              chrome.scripting.executeScript({
                target: { tabId: tab.id },
                files: ["review_inject.js"],
              });
            });
          }
        });
      }
    });
  }
});

// 处理来自内容脚本的消息（如有需要）
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
