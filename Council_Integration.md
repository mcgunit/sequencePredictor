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

On the machine running the orchestrator:

```bash
python api.py --config config.json --port 8099 --check
```

When it is not running the page says "The members of the council have to be
summoned", rather than failing silently.

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