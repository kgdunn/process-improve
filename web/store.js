// Where the page keeps the reader's designs: IndexedDB, in this browser, on this device only.
// Nothing leaves the machine. A browser may clear it (Safari does after seven days without a
// visit, unless the app is installed), so the page also offers the workbook and a link.
//
// Every call resolves, and to nothing rather than an error when the browser refuses storage
// (a private window in some browsers): the page then works as it did before it could save.

const DATABASE = "factorial";
const CAMPAIGNS = "campaigns";

/** How many designs the page keeps; saving one more drops the one opened longest ago. */
export const KEEP = 30;

let database = null;

function open() {
  database ??= new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(CAMPAIGNS, { keyPath: "id" });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(new Error("the design store is open in an older tab"));
  });
  return database;
}

/** Run one request in a transaction; resolve to its result once the transaction is done, or to null if it fails. */
async function transaction(mode, make) {
  try {
    const db = await open();
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(CAMPAIGNS, mode);
      const request = make(tx.objectStore(CAMPAIGNS));
      tx.oncomplete = () => resolve(request.result);
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
  } catch {
    database = null; // the next call tries again
    return null;
  }
}

/** Every saved design, the one saved most recently first; [] if the browser keeps none. */
export async function savedCampaigns() {
  const all = (await transaction("readonly", (store) => store.getAll())) ?? [];
  return all.sort((a, b) => b.updated - a.updated);
}

/** Save a design (a new one, or a newer copy of one saved before), keeping only the KEEP most recent. */
export async function saveCampaign(record) {
  const saved = await transaction("readwrite", (store) => store.put(record));
  if (saved === null) return false;
  const all = await savedCampaigns();
  for (const old of all.slice(KEEP)) await forgetCampaign(old.id);
  return true;
}

/** Remove a saved design. */
export const forgetCampaign = (id) => transaction("readwrite", (store) => store.delete(id));
