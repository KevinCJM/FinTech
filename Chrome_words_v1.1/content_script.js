// content_script.js

console.log("Content script loaded on", window.location.href);

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "getSentence") {
    const word = request.word;
    const sentence = getSentenceContainingWord(word);
    sendResponse({ sentence: sentence });
  }
  return true;
});

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
