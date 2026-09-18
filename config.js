const path = require('path');
const fs = require('fs');

// Load the repo-root .env (gitignored; copy .env.example) before reading the
// settings below. Variables already present in the real environment keep
// precedence, like Node's own --env-file. Node >= 20.12 has
// process.loadEnvFile; the fallback parser covers older runtimes and the
// same KEY=value / KEY="value" / # comment syntax. A change to .env needs a
// server restart to be picked up.
(function loadDotEnv() {
    const envFile = path.join(__dirname, ".env");
    if (!fs.existsSync(envFile)) return;
    try {
        if (typeof process.loadEnvFile === "function") { process.loadEnvFile(envFile); return; }
        for (const rawLine of fs.readFileSync(envFile, "utf-8").split(/\r?\n/)) {
            const line = rawLine.trim();
            if (!line || line.startsWith("#")) continue;
            const eq = line.indexOf("=");
            if (eq <= 0) continue;
            const key = line.slice(0, eq).trim().replace(/^export\s+/, "");
            let value = line.slice(eq + 1).trim();
            if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) value = value.slice(1, -1);
            if (process.env[key] === undefined) process.env[key] = value;
        }
    } catch (e) {
        console.log(`Could not read ${envFile}: ${e.message}`);
    }
})();

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
    // ignored with a warning. WEB_SESSION_SECRET or SESSION_SECRET (either
    // name). When unset one is generated once into CONFIG_DIR/session.secret.
    SESSION_SECRET: process.env.WEB_SESSION_SECRET || process.env.SESSION_SECRET || "",
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
