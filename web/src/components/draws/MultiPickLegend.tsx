import { DASH, isNum, share } from '../../format';
import type { MultiPickLegendData } from '../../workbench';
import { Scroll } from '../admin/Card';
import type { Mode } from './text';

// Lotto's multi-pick, once per draw (B7): a row's three shaded numbers are its next most probable ones, for a system
// play. Per size: the grids it is, its price, and the chance of 3 or more of the six drawn among its numbers - more
// numbers win more by arithmetic alone. The admin's line points to the History tracking of the extras.
export default function MultiPickLegend({ data, mode }: { data: MultiPickLegendData; mode: Mode }) {
  const price = (v: number | null) => (isNum(v) ? `${v.toFixed(2)} EUR` : DASH);
  return (
    <>
      <p className="note">
        Shaded numbers: a row&apos;s {data.extra} next most probable numbers, for a system play at {price(data.gridPrice)} a grid. Chance of {data.minHits ?? 3} or more of the 6
        drawn among the numbers played (by arithmetic alone; any extra number hits {share(data.perExtra)} of draws by luck):
      </p>
      <Scroll>
        <table className="tbl">
          <thead>
            <tr>
              <th>Numbers</th>
              <th className="num">Grids</th>
              <th className="num">Price</th>
              <th className="num">Chance of {data.minHits ?? 3}+</th>
            </tr>
          </thead>
          <tbody>
            {data.sizes.map((s) => (
              <tr key={s.size}>
                <td>{s.size}</td>
                <td className="num">{s.grids ?? DASH}</td>
                <td className="num">{price(s.price)}</td>
                <td className="num">{share(s.chance)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Scroll>
      {mode === 'admin' ? <p className="note">Whether the extras do better than that is tracked per row on Best model &amp; rankings (lotto&apos;s drawer).</p> : null}
    </>
  );
}
