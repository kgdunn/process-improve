// The service worker a build puts in sw.js's place to retire it (scripts/build_web_app.py --no-service-worker).
//
// A browser that has Factorial's worker asks for sw.js again at the reader's next visit, whatever the page's own
// code does, and finds this one: it takes over at once, deletes every copy the worker kept, and unregisters. The
// page then loads from the network, as it did before there was a worker. The same build's manifest tells the page
// not to register one again (offline.js).

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      for (const name of await caches.keys()) if (name.startsWith("factorial-")) await caches.delete(name);
      await self.registration.unregister();
    })(),
  );
});
