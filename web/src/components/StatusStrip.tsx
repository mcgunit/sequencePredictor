import type { Status } from '../api';
import { DASH, day, int, isNum, money, plural, share, signClass, unsigned } from '../format';

export default function StatusStrip({ status: s, unit }: { status: Status; unit: string }) {
  const planMeta = [unit, s.since ? `since ${day(s.since)}` : '', plural(s.days, 'day'), 'day game'].filter(Boolean).join(' · ');
  const pending = s.days > 0 && s.positions === 0; // the record has days but no positions yet: they are written by the next run
  const wk = s.week;
  const wkMeta = wk ? [unit, plural(wk.days, 'week'), isNum(wk.liveMarket) ? `market ${money(wk.liveMarket)}` : ''].filter(Boolean).join(' · ') : '';
  const behind = isNum(s.ahead) && s.ahead < 0;
  const positionsMeta = [`${int(s.won)} won`, `${int(s.lost)} lost`, isNum(s.winRate) ? share(s.winRate) : ''].filter(Boolean).join(' · ');
  const daysMeta = ['won', s.bestDay ? `best ${money(s.bestDay.pnl)}` : '', s.worstDay ? `worst ${money(s.worstDay.pnl)}` : '']
    .filter(Boolean)
    .join(' · ');
  const daysTitle = [s.bestDay ? `best day ${day(s.bestDay.date)}` : '', s.worstDay ? `worst day ${day(s.worstDay.date)}` : '']
    .filter(Boolean)
    .join(', ');

  return (
    <div className="status">
      <div>
        <span className="k">Following the plan</span>
        <span className={`v ${signClass(s.liveTotal)}`}>
          {money(s.liveTotal)} <small>{planMeta}</small>
        </span>
      </div>
      <div>
        <span className="k">vs buying everything</span>
        <span className="v">
          <span className={signClass(s.liveMarket)}>{money(s.liveMarket)}</span>
          {isNum(s.ahead) ? (
            <>
              {' '}
              <small>plan is</small> <span className={signClass(s.ahead)}>{unsigned(s.ahead)}</span> <small>{behind ? 'behind' : 'ahead'}</small>
            </>
          ) : null}
        </span>
      </div>
      {wk ? (
        <div>
          <span className="k">Week game</span>
          <span className={`v ${signClass(wk.liveTotal)}`}>
            {money(wk.liveTotal)} <small>{wkMeta}</small>
          </span>
        </div>
      ) : null}
      <div>
        <span className="k">Positions</span>
        <span className="v">
          {pending ? DASH : int(s.positions)} <small>{pending ? 'positions appear after the next run' : positionsMeta}</small>
        </span>
      </div>
      <div>
        <span className="k">Days</span>
        <span className="v" title={daysTitle || undefined}>
          {int(s.daysWon)} / {int(s.days)} <small>{daysMeta}</small>
        </span>
      </div>
      <div>
        <span className="k">Open now</span>
        <span className="v">
          {int(s.openToBuy)} <small>to buy</small> · {int(s.openHeld)} <small>held</small>
        </span>
      </div>
      {s.paperNote ? (
        <button type="button" className="q" title={s.paperNote} aria-label={s.paperNote}>
          ?
        </button>
      ) : null}
    </div>
  );
}
