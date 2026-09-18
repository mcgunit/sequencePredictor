const path = require('path');

module.exports = {
    INTERFACE: "127.0.0.1",
    PORT: 3001,

    // Login (README "Run server"). Both must be set to require a login; with
    // either unset the pages are open (local development). The administrator
    // exists only in the environment and never on disk.
    ADMIN_USER: process.env.WEB_USER || "",
    ADMIN_PASSWORD: process.env.WEB_PASSWORD || "",
    // Optional fixed secret for the session cookie signature - at least 32
    // bytes (64 hex characters, or a long passphrase); anything shorter is
    // ignored with a warning. When unset one is generated once into
    // CONFIG_DIR/session.secret.
    SESSION_SECRET: process.env.WEB_SESSION_SECRET || "",
    // Express "trust proxy" setting: which proxies' X-Forwarded-* headers to
    // believe for the client address (login lockout) and the HTTPS flag.
    // "loopback" = a reverse proxy on this machine; set WEB_TRUST_PROXY to a
    // proxy address/CIDR, "true" for any proxy, or "false" for none.
    TRUST_PROXY: (process.env.WEB_TRUST_PROXY === undefined || process.env.WEB_TRUST_PROXY === "")
        ? "loopback"
        : (process.env.WEB_TRUST_PROXY === "true" ? true : (process.env.WEB_TRUST_PROXY === "false" ? false : process.env.WEB_TRUST_PROXY)),
    // Gitignored folder for the user accounts (users.json) and the session
    // secret - separate from data/, which the daily predictor commits.
    CONFIG_DIR: path.join(__dirname, "config"),
}
