// The README's disclaimer at the foot of every page, sticky so it is always in view (5 Nov 2026, owner): one compact
// line, the full text folded behind "Read more" (every collapsible starts folded). Keep it in step with the README's.
export default function Disclaimer() {
  return (
    <footer className="disclaimer">
      <details className="help">
        <summary>
          <strong>Research project, not financial or gambling advice.</strong> Every euro shown is paper money; you use
          what it shows at your own risk. <span className="more">Read more</span>
        </summary>
        <ul>
          <li>
            Nobody involved is a licensed financial advisor. No ticket, buy or sell, profit figure or model ranking here is a
            recommendation to buy, sell, bet or play anything.
          </li>
          <li>
            Lottery draws are designed to be random. No model here has shown a reproducible edge over chance; the
            project's own controls exist to check exactly that, and so far they find none.
          </li>
          <li>
            Crypto and share prices can fall as well as rise. Past results, paper profits and backtests say nothing
            reliable about the future.
          </li>
          <li>
            The authors and contributors accept no responsibility or liability for any loss of money, or any other damage,
            that results from using it, whether you play, trade or invest on the strength of what it shows or not.
          </li>
          <li>
            If gambling or trading is causing you problems, look for help, in Belgium for example through the National
            Lottery's responsible-gaming information or your doctor.
          </li>
        </ul>
      </details>
    </footer>
  );
}
