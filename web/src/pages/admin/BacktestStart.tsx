import { useState } from 'react';
import { GAMES } from '../../api';
import { short } from '../../format';
import type { Sender } from '../../hooks/useSend';
import Confirm from '../../components/admin/Confirm';

// Starting a backtest (POST /api/admin/backtests/start, backtests.js): one row on one game over the newest N draws,
// with the served values (the tuned file and the manual values, as the daily run) or the tuned ones alone. Opened
// from a Parameters card (game and row fixed, `rows` the card's backtestable rows) or from the Backtests page (both
// chosen from `table`, {row: [games]}).
export default function BacktestStart({
  sender,
  game: fixedGame,
  rows: cardRows,
  table,
  daysDefault = 90,
  daysMax = 1000,
  onStarted,
  onCancel,
}: {
  sender: Sender;
  game?: string;
  rows?: string[];
  table?: Record<string, string[]>;
  daysDefault?: number;
  daysMax?: number;
  onStarted: (id: string | null) => void;
  onCancel: () => void;
}) {
  const allRows = cardRows ?? Object.keys(table ?? {});
  const [row, setRow] = useState(allRows[0] ?? '');
  const gamesOf = (r: string) => (fixedGame ? [fixedGame] : (table?.[r] ?? GAMES));
  const [game, setGame] = useState(fixedGame ?? gamesOf(allRows[0] ?? '')[0] ?? 'lotto');
  const [days, setDays] = useState(String(daysDefault));
  const [values, setValues] = useState<'served' | 'tuned'>('served');
  const n = Number(days);
  const bad = !Number.isInteger(n) || n < 1 || n > daysMax ? `a whole number of draws from 1 to ${daysMax}` : null;
  const start = async () => {
    if (bad) return;
    const done = await sender.send<{ ok: true; message: string; id?: string }>('/api/admin/backtests/start', { game, row, days: n, values });
    if (done) onStarted(done.id ?? null);
  };
  return (
    <Confirm
      title={fixedGame && allRows.length === 1 ? `Backtest ${short(row)} on ${game}` : 'Start a backtest'}
      action="Start backtest"
      busy={sender.busy}
      error={sender.error ?? (days !== '' ? bad : null)}
      disabled={Boolean(bad) || !row}
      onConfirm={() => void start()}
      onCancel={onCancel}
      lines={[
        'The weekend tuner’s own score: each draw predicted from the history before it only. The lockbox draws are backtested but never scored or shown.',
        'It waits behind every scheduled job and only starts when it can finish before the next scheduled run (crypto 00:05 UTC, lottery 06:00, stocks 18:05); boosting rows can take an hour or more.',
      ]}
    >
      {!fixedGame || allRows.length > 1 ? (
        <label className="field">
          <span>Model</span>
          <select
            value={row}
            onChange={(e) => {
              setRow(e.target.value);
              if (!fixedGame && !gamesOf(e.target.value).includes(game)) setGame(gamesOf(e.target.value)[0] ?? game);
            }}
          >
            {allRows.map((r) => (
              <option key={r} value={r}>
                {short(r)}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      {!fixedGame ? (
        <label className="field">
          <span>Game</span>
          <select value={game} onChange={(e) => setGame(e.target.value)}>
            {gamesOf(row).map((g) => (
              <option key={g} value={g}>
                {g}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      <label className="field">
        <span>Draws</span>
        <input value={days} onChange={(e) => setDays(e.target.value)} inputMode="numeric" autoComplete="off" />
        <small>The newest draws to walk through ({daysDefault} is the weekend tuners’ window, so the score compares with the gate’s).</small>
      </label>
      <label className="field">
        <span>Values</span>
        <select value={values} onChange={(e) => setValues(e.target.value === 'tuned' ? 'tuned' : 'served')}>
          <option value="served">Served: the tuned values with the manual ones (as the daily run)</option>
          <option value="tuned">Tuned: the tuners’ values only, no manual ones</option>
        </select>
      </label>
    </Confirm>
  );
}
