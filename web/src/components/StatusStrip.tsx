import type { Status } from '../api';
import { DASH, day, int, isNum, money, plural, share, signClass, unsigned } from '../format';
import Tile from './Tile';

// A row of six equal tiles: label, one figure, one line of context. What following the whole table made
// (daily and weekly positions together) first, then how it compares with buying everything, the
// positions, the days, the coins, and what is open now.
export default function StatusStrip({ status: s, unit, noun }: { status: Status; unit: string; noun: string }) {
  const pending = s.days > 0 && s.positions === 0; // the record has days but no positions yet: they are written by the next run
  const arrow = (v: number | null) => (isNum(v) ? (v > 0 ? '▲ ' : v < 0 ? '▼ ' : '') : '');
  const split = [isNum(s.dailyTotal) ? `daily ${money(s.dailyTotal)}` : '', isNum(s.weeklyTotal) ? `weekly ${money(s.weeklyTotal)}` : ''].filter(Boolean);
  // the weekly part counts the plain week before the guard was on the page and the guard after it (api.js guardFrom)
  const weekText = s.week && s.week.guardFrom ? ` The weekly part counts the plain week (held to the week's end) for weeks before ${day(s.week.guardFrom)} and the guard from then on: what the page said each week.` : '';
  return (
    <div className="kpis">
      <Tile
        label="Following the plan"
        value={money(s.liveTotal)}
        cls={signClass(s.liveTotal)}
        unit={unit}
        foot={split.length > 1 ? split.join(' · ') : s.since ? `since ${day(s.since)} · ${plural(s.days, 'day')}` : 'no live day yet'}
        title={`What doing every row of the table made, daily and weekly positions together${s.since ? `, since ${day(s.since)}` : ''}; paper money, live days only.${weekText}${s.entryNote ? ` ${s.entryNote}` : ''}`}
      />
      <Tile
        label="vs buying everything"
        value={isNum(s.ahead) ? `${arrow(s.ahead)}${unsigned(s.ahead)}` : DASH}
        cls={signClass(s.ahead)}
        unit={isNum(s.ahead) ? (s.ahead < 0 ? 'worse' : 'better') : undefined}
        foot={`plan ${money(s.liveTotal)} · market ${money(s.liveMarket)}`}
        title={`The plan's result minus what buying every ${noun || 'instrument'} made under the same rules, over the same days with the same stake and fees.`}
      />
      <Tile
        label="Win rate"
        value={pending ? DASH : share(s.winRate)}
        foot={pending ? 'after the next run' : `${int(s.won)} won · ${int(s.lost)} lost`}
        title="Positions that made money, daily and weekly."
      />
      <Tile
        label="Days won"
        value={`${int(s.daysWon)} / ${int(s.daysPlayed)}`}
        foot={[s.bestDay ? `best ${money(s.bestDay.pnl)}` : '', s.worstDay ? `worst ${money(s.worstDay.pnl)}` : ''].filter(Boolean).join(' · ') || DASH}
        title={[
          `Days on which the positions booked that day made money, out of the days with at least one position. A week's positions are booked on the week's last day, also those the guard sold earlier in the week`,
          s.daysIdle ? `${plural(s.daysIdle, 'day')} without a position, not counted` : '',
          s.bestDay ? `best day ${day(s.bestDay.date)}` : '',
          s.worstDay ? `worst day ${day(s.worstDay.date)}` : '',
        ].filter(Boolean).join('; ') + '.'}
      />
      <Tile
        label={`Best ${noun || 'coin'}`}
        value={s.bestCoin ? s.bestCoin.symbol : DASH}
        cls={s.bestCoin ? signClass(s.bestCoin.pnl) : 'mut'}
        unit={s.bestCoin ? money(s.bestCoin.pnl) : undefined}
        foot={pending ? 'after the next run' : s.worstCoin ? `worst ${s.worstCoin.symbol} ${money(s.worstCoin.pnl)}` : DASH}
        title={s.bestCoin ? `${s.bestCoin.symbol}: ${money(s.bestCoin.pnl)} ${unit} over ${plural(s.bestCoin.positions, 'position')}` : undefined}
      />
      <Tile
        label="Open now"
        value={int(s.openToBuy + s.openHeld)}
        foot={`${int(s.openToBuy)} to buy · ${int(s.openHeld)} held`}
        title="Positions in the table below: to buy now, or held."
      />
    </div>
  );
}
