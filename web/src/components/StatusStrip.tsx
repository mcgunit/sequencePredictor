import type { TraderStatus } from '../api';
import { DASH, day, int, isNum, money, plural, share, signClass, unsigned } from '../format';
import Tile from './Tile';

// A row of six equal tiles on the paper trader's long positions (18 Nov 2026; the day and week games' figures
// before it): what the record made, how that compares with holding, the closed positions won, how it stands
// against the same positions on random days, the instruments, and what is open now.
export default function StatusStrip({ status: s, unit, noun }: { status: TraderStatus; unit: string; noun: string }) {
  const none = s.positions === 0;
  const arrow = (v: number | null) => (isNum(v) ? (v > 0 ? '▲ ' : v < 0 ? '▼ ' : '') : '');
  const since = s.since ? `since ${day(s.since)}` : '';
  return (
    <div className="kpis">
      <Tile
        label="The trader's record"
        value={none ? DASH : money(s.total)}
        cls={none ? 'mut' : signClass(s.total)}
        unit={none ? undefined : unit}
        foot={none ? `no position yet${since ? ` · ${since}` : ''}` : [since, plural(s.days, 'day')].filter(Boolean).join(' · ')}
        title={`What the trader's long positions made on paper${since ? `, ${since}` : ''}: the sold ones, and the open ones as if sold at the last close; costs included. It is the rules' record, not what a reader made.`}
      />
      <Tile
        label="vs holding"
        value={isNum(s.ahead) ? `${arrow(s.ahead)}${unsigned(s.ahead)}` : DASH}
        cls={signClass(s.ahead)}
        unit={isNum(s.ahead) ? (s.ahead < 0 ? 'worse' : 'better') : undefined}
        foot={isNum(s.ahead) ? `trader ${money(s.total)} · holding ${money(s.hold)}` : 'after the first position'}
        title={`The trader's result minus what buying every ${noun || 'instrument'} on the trader's first day and holding it made, with the same stake and costs.`}
      />
      <Tile
        label="Win rate"
        value={s.closed ? share(s.winRate) : DASH}
        foot={s.closed ? `${int(s.won)} won · ${int(s.lost)} lost` : 'no sold position yet'}
        title="Sold positions that made money, costs included. An open position is not counted here."
      />
      <Tile
        label="Against chance"
        value={isNum(s.randomBeaten) ? share(s.randomBeaten) : DASH}
        foot={isNum(s.randomMean) ? `random days ${money(s.randomMean)}` : 'after the first position'}
        title={`The trader's own positions, as many and as long, laid on random days ${int(s.randomRuns)} times: the share of those draws the trader did better than, and what they made on average. Around 50% is what chance gives.`}
      />
      <Tile
        label={`Best ${noun || 'coin'}`}
        value={s.bestCoin ? s.bestCoin.symbol : DASH}
        cls={s.bestCoin ? signClass(s.bestCoin.pnl) : 'mut'}
        unit={s.bestCoin ? money(s.bestCoin.pnl) : undefined}
        foot={s.worstCoin ? `worst ${s.worstCoin.symbol} ${money(s.worstCoin.pnl)}` : DASH}
        title={s.bestCoin ? `${s.bestCoin.symbol}: ${money(s.bestCoin.pnl)} ${unit} over ${plural(s.bestCoin.positions, 'position')}` : undefined}
      />
      <Tile
        label="Open now"
        value={int(s.open)}
        foot={`${int(s.toBuy)} to buy · ${int(s.toSell)} to sell today`}
        title="Positions the trader holds, and what the card above says to do today."
      />
    </div>
  );
}
