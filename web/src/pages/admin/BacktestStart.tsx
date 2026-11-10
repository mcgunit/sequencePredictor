import { useState } from 'react';
import { GAMES, type BacktestLimits } from '../../api';
import { short } from '../../format';
import type { Sender } from '../../hooks/useSend';
import Confirm from '../../components/admin/Confirm';

// Starting a backtest (POST /api/admin/backtests/start, backtests.js): one row on one game over the newest N draws,
// with the served values (the tuned file and the manual values, as the daily run) or the tuned ones alone. Opened
// from a Parameters card (game and row fixed, `rows` the card's backtestable rows) or from the Backtests page (both
// chosen from `table`, {row: [games]}). `limits` gives each row's draws: the deep-learning rows run on the GPU worker,
// 3 draws by default and 30 at most (each draw is a retrain); the others take the props' 90 / 1000.
export default function BacktestStart({
  sender,
  game: fixedGame,
  rows: cardRows,
  table,
  daysDefault = 90,
  daysMax = 1000,
  limits,
  onStarted,
  onCancel,
}: {
  sender: Sender;
  game?: string;
  rows?: string[];
  table?: Record<string, string[]>;
  daysDefault?: number;
  daysMax?: number;
  limits?: Record<string, BacktestLimits>;
  onStarted: (id: string | null) => void;
  onCancel: () => void;
}) {
  const allRows = cardRows ?? Object.keys(table ?? {});
  const [row, setRow] = useState(allRows[0] ?? '');
  const gamesOf = (r: string) => (fixedGame ? [fixedGame] : (table?.[r] ?? GAMES));
  const [game, setGame] = useState(fixedGame ?? gamesOf(allRows[0] ?? '')[0] ?? 'lotto');
  const limitOf = (r: string): BacktestLimits => limits?.[r] ?? { daysDefault, daysMax, gpu: false, where: 'server' };
  const lim = limitOf(row);
  const [days, setDays] = useState(String(limitOf(allRows[0] ?? '').daysDefault));
  const [values, setValues] = useState<'served' | 'tuned'>('served');
  const n = Number(days);
  const bad = !Number.isInteger(n) || n < 1 || n > lim.daysMax ? `a whole number of draws from 1 to ${lim.daysMax}` : null;
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
      lines={
        lim.gpu
          ? [
              'The deep-learning tuner’s own score: each draw is a retrain on the history before it (240 s of training, as the daily run), scored by how much better than the counts it predicts (0 = no better). The lockbox draws are left out of the score.',
              'It runs on the GPU worker, one at a time, beside the schedule: it does not wait for the scheduled jobs. A few minutes a draw (2 for the LSTM on lotto; the training stops at 240 s).',
            ]
          : lim.where === 'cpu'
            ? [
                'The weekend tuner’s own score: each draw predicted from the history before it only. The lockbox draws are backtested but never scored or shown.',
                'It runs on the DL360, one at a time, beside the schedule (12 hours at most). The DL360 is switched on if it is off (5 to 10 minutes) and switches itself off after its idle minutes. It is about half this server’s speed: boosting rows can take two hours or more.',
              ]
            : [
                'The weekend tuner’s own score: each draw predicted from the history before it only. The lockbox draws are backtested but never scored or shown.',
                'It waits behind every scheduled job and only starts when it can finish before the next scheduled run (crypto 00:05 UTC, lottery 06:00, stocks 18:05); boosting rows can take an hour or more.',
              ]
      }
    >
      {!fixedGame || allRows.length > 1 ? (
        <label className="field">
          <span>Model</span>
          <select
            value={row}
            onChange={(e) => {
              setRow(e.target.value);
              if (limitOf(e.target.value).gpu !== lim.gpu || Number(days) > limitOf(e.target.value).daysMax) setDays(String(limitOf(e.target.value).daysDefault));
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
        <small>
          {lim.gpu
            ? `The newest draws to retrain, at most ${lim.daysMax} (${lim.daysDefault} is the deep-learning gate’s window, so the score compares with the gate’s).`
            : `The newest draws to walk through (${lim.daysDefault} is the weekend tuners’ window, so the score compares with the gate’s).`}
        </small>
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
