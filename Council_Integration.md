# Wiring the council page into server.js

`web/council.js` is an Express router module. It renders the chat page with
your existing `generateHeader` / `generateFooter`, and proxies two endpoints to
the Python API.

## 1. Copy the file

Put `council.js` next to `server.js` (or in a subfolder and adjust the require
path).

## 2. Add the require

With the other requires at the top of `server.js`:

```js
const council = require("./council");
```

## 3. Install the routes

At the bottom of `server.js`, next to `auth.install(...)` — **after**
`app.use(auth.middleware)`, so the page and its proxy are behind the login:

```js
council.install(app, {
  header: generateHeader,
  footer: generateFooter,
  escapeHtml: auth.escapeHtml
});
```

## 4. Add the nav link

In `generateHeader`, in the `nav-group` div:

```js
<a href="/council">Council</a>
```

Or admin-only, matching the Users link:

```js
${user && user.role === 'admin' ? '<a href="/council">Council</a>' : ''}
```

## 5. Start the Python API

**The web server starts it.** `services.js` supervises the API as a child of
the Node process: it starts with the server, is restarted with a growing
backoff if it dies (and marked failed rather than hammered after five exits in
ten minutes), stops with the server, and is visible on the admin **Jobs** page
with its state, its log and start/stop/restart buttons. Set
`COUNCIL_API_AUTOSTART=off` in `.env` to go back to starting it by hand. If
something already answers on the port - the other checkout, or an instance you
started yourself - the server adopts it and says so instead of starting a
second one.

By hand it is:

```bash
python api.py --config config.json --port 8099
```

Leave `--check` **off** here. It probes the model endpoints once and refuses to
serve when any of them is down, which is useful as a manual pre-flight but
wrong for the API the page talks to: the llama.cpp boxes are not powered
around the clock, and with `--check` the API would simply not start while they
are off.

The API is therefore always up, and reports the state of the models instead.
`GET /health` and `GET /endpoints` both carry a `status` object - every
configured endpoint probed in parallel with a 2 s timeout, cached for 15 s, so
a dark model box costs one short timeout per quarter minute rather than one
per page load:

```json
{"checked": "2026-09-20T16:27:06+00:00", "ready": false, "reachable": 0, "total": 3,
 "members": [{"name": "qwen2.5-1.5b", "ok": false, "detail": "...unreachable..."}],
 "head": {"name": "head-llama-3.2-3b", "ok": false, "detail": "..."}}
```

`ready` is what the page gates on: at least one member answers and, when a head
is configured, the head answers too - without the head there is nobody to
synthesise the members' answers. The page then says "The members of the council
have to be summoned" and keeps the Ask button disabled, naming the endpoints
that did not answer. It shows the same message when the API itself is not
running, and when only some members are down it serves normally with a
"n member(s) not answering" note.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `COUNCIL_API_HOST` | `127.0.0.1` | Where api.py listens |
| `COUNCIL_API_PORT` | `8099` | |
| `COUNCIL_ACCESS` | `user` | `admin` restricts the page to admins |

## Two things to check on your side

**CSRF.** The page sends the token from `req.user.csrf` as an `X-CSRF-Token`
header, because a JSON `fetch` cannot use the hidden form field your other
POSTs use. Whether `auth.middleware` validates that header is something only
your `auth.js` can answer — if it only checks `req.body._csrf` on urlencoded
forms, these JSON POSTs are currently unprotected against CSRF. Either extend
the middleware to accept the header, or have the proxy read `_csrf` from the
JSON body and validate it the same way the forms do.

**Access.** A council run costs minutes of CPU on the inference box and there
is no rate limiting beyond "one job at a time". Any logged-in user can start
one with the default setting. `COUNCIL_ACCESS=admin` narrows that.

## Why proxy rather than call the API from the browser

The Python API has no authentication. Proxying keeps it bound to localhost,
unreachable from the network, and leaves login in one place — your existing
`auth.middleware` — instead of being reimplemented in Python. It also avoids
CORS entirely, since the browser only ever talks to the Express origin.