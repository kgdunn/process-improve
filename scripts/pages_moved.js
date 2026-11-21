// Factorial has moved to https://factori.al/ (#1237). This runs in each page of the copy on GitHub Pages only:
// scripts/move_pages_copy.py puts it there, and factori.al's own build never has it. A browser that keeps no study at
// this address goes on to the same page at factori.al, its query and its # part with it. One that keeps studies stays:
// they live in this address's storage and do not move by themselves, so the page says how to take them along.
(() => {
  const HOME = "https://factori.al/";
  const APP = location.pathname.replace(/(\/app\/).*$/, "$1"); // the copy's own root, where Your studies is
  const there = new URL(location.pathname.replace(/^.*?\/app\//, "") + location.search + location.hash, HOME).href;

  /** The studies kept here: their number, 0 when there are none, or null when the browser does not say. */
  const kept = () =>
    new Promise((resolve) => {
      let fresh = false;
      let request;
      try {
        request = indexedDB.open("factorial");
      } catch {
        resolve(0); // no storage at all, so nothing was saved
        return;
      }
      request.onupgradeneeded = () => {
        fresh = true; // no database here: none is made, the open is undone
        request.transaction.abort();
      };
      request.onerror = () => resolve(fresh ? 0 : null);
      request.onblocked = () => resolve(null);
      request.onsuccess = () => {
        const db = request.result;
        try {
          const count = db.transaction("campaigns").objectStore("campaigns").count();
          count.onsuccess = () => resolve(count.result);
          count.onerror = () => resolve(null);
        } catch {
          resolve(db.objectStoreNames.contains("campaigns") ? null : 0);
        } finally {
          db.close();
        }
      };
    });

  const link = (href, text) => Object.assign(document.createElement("a"), { href, textContent: text });

  function say(count) {
    const box = document.createElement("div");
    box.id = "factorial-moved";
    box.setAttribute("role", "region");
    box.setAttribute("aria-label", "Factorial has moved");
    box.style.cssText =
      "background:#fff4c2;color:#1d1d1f;padding:12px 16px;font:16px/1.5 system-ui,sans-serif;border-bottom:1px solid #c9a227";
    const studies = count === 1 ? "1 study" : `${count} studies`;
    const head = document.createElement("strong");
    head.append("Factorial has moved to ", link(there, "factori.al"), ".");
    const how = document.createElement("p");
    how.style.margin = "4px 0 0";
    how.append(
      count === null
        ? "Studies saved at this address stay in this browser, here. "
        : `This browser keeps ${studies} at this address, and they do not move by themselves. `,
      "To take them along, open ",
      link(APP, "Your studies"),
      " (under Studies), press Download all, then drop that file into Your studies at factori.al.",
    );
    box.append(head, how);
    document.body.prepend(box);
  }

  kept().then((count) => {
    if (count === 0) {
      location.replace(there);
      return;
    }
    if (document.body) say(count);
    else document.addEventListener("DOMContentLoaded", () => say(count));
  });
})();
