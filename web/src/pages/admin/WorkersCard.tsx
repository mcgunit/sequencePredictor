import { getAdminWorkers, type WorkersView } from '../../api';
import { DASH, stamp, time } from '../../format';
import { usePoll } from '../../hooks/usePoll';
import { Card } from '../../components/admin/Card';
import Chip from '../../components/admin/Chip';
import type { Ask } from './jobs';

// Jobs & services › Workers (workers.js, 10 Nov 2026): the GPU worker (the 2080 Ti LXC, always on) and the DL360 -
// on or off, whether this server woke it (then it switches itself off after its idle minutes without a backtest, a
// council request or a login), and Switch on / Switch off behind a confirmation. Read every 15 s; the DL360's state
// itself is read at most every 30 s (an IPMI call and an ssh call).
const POLL_MS = 15000;
const minutesAgo = (ms: number | null) => (ms === null ? null : Math.max(0, Math.round((Date.now() - ms) / 60000)));

export default function WorkersCard({ onAsk }: { onAsk: (a: Ask) => void }) {
  const poll = usePoll(getAdminWorkers, POLL_MS);
  const d = poll.data;
  return (
    <Card id="workers" title="Workers" help="The machines that run the backtests beside this server. They get the code and the training data, and send back the backtest's record only." meta={['Brussels time']}>
      {!d ? (
        <p className="note">{poll.error ? `Not read: ${poll.error.message}` : 'Reading…'}</p>
      ) : (
        <dl className="kv">
          <dt>GPU worker</dt>
          <dd>
            {d.gpu.configured ? (
              <>
                <span className="ln">
                  <Chip kind="flat">set up</Chip> {d.gpu.host ?? DASH} · always on
                </span>
                <span className="sub">the deep-learning backtests{d.gpu.running ? ' · one running' : ''}{d.gpu.queued ? ` · ${d.gpu.queued} queued` : ''}</span>
              </>
            ) : (
              <span className="mut">not set up (config/workers.json): the deep-learning rows cannot be backtested</span>
            )}
          </dd>
          <dt>DL360</dt>
          <dd>{d.cpu ? <Dl360 cpu={d.cpu} onAsk={onAsk} /> : <span className="mut">not set up (config/workers.json): the other backtests run on this server</span>}</dd>
        </dl>
      )}
    </Card>
  );
}

function Dl360({ cpu, onAsk }: { cpu: NonNullable<WorkersView['cpu']>; onAsk: (a: Ask) => void }) {
  if (cpu.note) return <span className="mut">{cpu.host ?? DASH}: {cpu.note}</span>;
  const idleFor = minutesAgo(cpu.lastBusy);
  const busy = Boolean(cpu.running || cpu.queued);
  const on = cpu.power === 'on';
  const state = cpu.switching
    ? cpu.switching.action === 'on' ? `switching on (asked by ${cpu.switching.by} at ${time(cpu.switching.at)}; 5 to 10 minutes)` : `shutting down (asked by ${cpu.switching.by})`
    : on ? (cpu.reachable ? 'on and answering' : 'on, not answering yet (starting?)') : cpu.power === 'off' ? 'off' : 'not known (the iLO did not answer)';
  return (
    <>
      <span className="ln">
        <Chip kind={on && cpu.reachable ? 'flat' : cpu.power === 'unknown' ? 'down' : 'dim'}>{on ? 'on' : cpu.power}</Chip> {cpu.host ?? DASH} · {state}
      </span>
      <span className="sub">
        {on && cpu.woken
          ? `switched on by this server${cpu.wokenAt ? ` at ${stamp(cpu.wokenAt)}` : ''}: it switches itself off after ${cpu.idleMinutes} minutes without a backtest, a council request or a login${idleFor !== null ? ` (idle ${idleFor} min)` : ''}`
          : on
            ? 'switched on by hand: it stays on until someone switches it off'
            : 'switched on when a backtest needs it'}
        {cpu.running ? ' · a backtest is running' : ''}
        {cpu.queued ? ` · ${cpu.queued} queued` : ''}
      </span>
      {cpu.error ? <span className="sub">last read failed: {cpu.error}</span> : null}
      <span className="sub">{cpu.readAt ? `read ${time(cpu.readAt)}` : 'not read yet'}</span>
      <div className="acts">
        <button
          type="button"
          className="btn"
          disabled={Boolean(cpu.switching) || (on && cpu.reachable === true)}
          onClick={() =>
            onAsk({
              title: 'Switch the DL360 on?',
              lines: [
                'Through its iLO. It takes 5 to 10 minutes, and the council comes up with it.',
                `It switches itself off again after ${cpu.idleMinutes} minutes without a backtest, a council request or a login. It draws about 185 W idle.`,
                'With a USB disk plugged in, its start-up stops at a red screen.',
              ],
              action: 'Switch on',
              url: '/api/admin/workers/cpu/on',
              body: {},
            })
          }
        >
          Switch on
        </button>
        <button
          type="button"
          className="btn danger"
          disabled={Boolean(cpu.switching) || !on || busy}
          onClick={() =>
            onAsk({
              title: 'Shut the DL360 down?',
              lines: ['A clean shutdown (systemctl poweroff). The council goes down with it.', 'A backtest needing it later switches it on again.'],
              action: 'Shut down',
              danger: true,
              url: '/api/admin/workers/cpu/off',
              body: {},
            })
          }
        >
          Shut down
        </button>
      </div>
      {busy ? <span className="fnote">Shut down waits until no backtest runs or waits for it.</span> : null}
    </>
  );
}
