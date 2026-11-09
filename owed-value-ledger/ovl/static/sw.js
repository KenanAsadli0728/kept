/* Service worker: makes the app installable and gives it an offline shell.
   Safety rule: LIVE DATA IS NEVER CACHED. Every API path goes straight to the network, so the app can never show stale
   ledger numbers. Only the static shell (page, icons, manifest) is cached, network-first, as an offline fallback. */
const CACHE = 'kept-shell-v2';
const SHELL = ['/', '/manifest.webmanifest', '/icons/icon-192.png', '/icons/icon-512.png', '/icons/icon.svg'];
const LIVE = /^\/(customers|ops|contracts|run|simulate|health|docs|redoc|openapi\.json)(\/|$|\?)/;

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;                       // never touch writes (approve / deny / simulate / run)
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  if (LIVE.test(url.pathname)) return;                    // live data: browser default, no caching, no fallback
  e.respondWith(
    fetch(req).then(res => {
      if (res.ok && (SHELL.includes(url.pathname) || url.pathname.startsWith('/icons/'))) {
        const copy = res.clone();
        caches.open(CACHE).then(c => c.put(req, copy));
      }
      return res;
    }).catch(() => caches.match(req).then(hit => hit || (req.mode === 'navigate' ? caches.match('/') : Response.error())))
  );
});
