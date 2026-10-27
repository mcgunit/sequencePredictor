import { useEffect } from 'react';
import { getAdminRandomness, getAdminRandomnessGame, type RandomnessLine, type RandomnessView } from '../../api';
import { DASH, day, fixed, short, signedFixed, stamp } from '../../format';
import { useLoad } from '../../hooks/useLoad';
import { useDrawerParam, useScope } from '../../hooks/useUrlState';
import { Card, Scroll } from '../../components/admin/Card';
import Chip from '../../components/admin/Chip';
import Drawer, { Section } from '../../components/admin/Drawer';
import ScopeSwitch from '../../components/admin/ScopeSwitch';
import { Failed, Loading, PageHead } from '../../components/admin/States';
import { GameButton, GroupRow, units } from './bits';

// /admin/randomness: the old Randomness watch. Per game the daily tripwires over the newest draws (entropy, KL
// against the game's own history, the autoencoder's lowest z) and the weekly Q2 verdict next to them; the
// status says which tripwire is up. A game's drawer (?game=lotto) holds the trend and the per-model KL rows,
// which have no reference value and are information only.
const REASON: Record<string, string> = { entropy: 'entropy', kl: 'KL', ae: 'autoencoder', q2: 'Q2' };

export default function RandomnessPage() {
  const load = useLoad(getAdminRandomness, []);
  const drawer = useDrawerParam();

  useEffect(() => {
    document.title = 'Randomness watch · Predictor';
  }, []);

  if (load.kind !== 'ok') {
    return (
      <>
        <PageHead title="Randomness watch" />
        {load.kind === 'loading' ? <Loading /> : <Failed error={load.error} retry={load.retry} />}
      </>
    );
  }
  const d = load.data;
  return (
    <div className="admin">
      <PageHead title="Randomness watch" sub="Daily tripwires · the weekly Q2 test next to them" meta={[`ranking ${stamp(d.generatedAt)}`, 'Brussels time']} />
      <div className="stack">
        <WatchCard d={d} onOpen={drawer.open} />
      </div>
      {drawer.game ? <RandomnessDrawer key={drawer.game} game={drawer.game} meta={d.meta} onClose={drawer.close} /> : null}
    </div>
  );
}

function tripwireText(m: RandomnessView['meta']) {
  const t = m.tripwires;
  return `entropy below ${fixed(t.entropyBelow, 2)} · KL vs history above ${fixed(t.klHistoryAbove, 2)} · autoencoder z below ${signedFixed(t.aeBelow, 0)}`;
}

function StatusCell({ g }: { g: RandomnessLine }) {
  if (g.status === 'normal') return <Chip>normal</Chip>;
  return (
    <span className="chips">
      <Chip kind="hi">watch</Chip>
      {g.reasons.map((r) => (
        <Chip key={r}>{REASON[r] ?? r}</Chip>
      ))}
    </span>
  );
}

function WatchCard({ d, onOpen }: { d: RandomnessView; onOpen: (g: string) => void }) {
  const groups = (['Lottery', 'Markets'] as const).map((name) => ({ name, games: d.games.filter((g) => g.group === name) })).filter((g) => g.games.length);
  return (
    <Card
      title="Randomness watch"
      flush
      help={
        <>
          Tripwires, deliberately loose: an alarm, not a verdict. Entropy: how evenly the newest draws spread over the numbers (1 = perfectly even), over{' '}
          {d.meta.window}. KL vs history: how far the newest draws&apos; frequencies sit from the game&apos;s own history; KL vs uniform is shown with it, without a
          tripwire. Autoencoder: {d.meta.ae}. Q2: {d.meta.q2}. Watch names the tripwires that are up. The market games and
          vikinglotto carry the autoencoder only. Rule changes, data artefacts and small windows all move these figures: a tripwire that is up is a reason to
          look, not proof of manipulation, and even a Q2 above the band has the boring causes in the README to rule out first.
        </>
      }
      meta={[`tripwires: ${tripwireText(d.meta)}`]}
    >
      <Scroll>
        <table className="tbl">
          <thead>
            <tr>
              <th>Game</th>
              <th>Status</th>
              <th className="num hide-sm">Entropy</th>
              <th className="num hide-sm">KL vs history</th>
              <th className="num hide-sm">Autoencoder</th>
              <th className="num">Q2</th>
            </tr>
          </thead>
          <tbody>
            {groups.map((grp) => (
              <Lines key={grp.name} name={grp.name} games={grp.games} onOpen={onOpen} />
            ))}
          </tbody>
        </table>
      </Scroll>
    </Card>
  );
}

function Lines({ name, games, onOpen }: { name: string; games: RandomnessLine[]; onOpen: (g: string) => void }) {
  return (
    <>
      <GroupRow cols={6}>{name}</GroupRow>
      {games.map((g) => (
        <tr key={g.game}>
          <td className="sym">
            <GameButton game={g.game} onOpen={onOpen} />
            {/* the phone line keeps each figure's window, as the hidden columns do */}
            <span className="sub show-sm">
              {g.entropy === null ? 'autoencoder only' : `entropy ${fixed(g.entropy)} · KL ${fixed(g.klHistory)} over the newest ${g.window ?? DASH} ${units(g.unit)}`}
              {g.ae ? ` · AE ${signedFixed(g.ae.minZ, 2)}, last ${g.ae.window ?? DASH} ${units(g.unit)}` : ''}
            </span>
          </td>
          <td>
            <StatusCell g={g} />
          </td>
          <td className="num hide-sm">
            {g.entropy === null ? <span className="fnt">{DASH}</span> : fixed(g.entropy)}
            <span className="sub">{g.window === null ? 'autoencoder only' : `newest ${g.window} of ${g.drawsTotal ?? DASH} ${units(g.unit)}`}</span>
          </td>
          <td className="num hide-sm">
            {g.klHistory === null ? <span className="fnt">{DASH}</span> : fixed(g.klHistory)}
            {g.klUniform !== null ? <span className="sub">vs uniform {fixed(g.klUniform)}</span> : null}
          </td>
          <td className="num hide-sm">
            {g.ae ? signedFixed(g.ae.minZ, 2) : <span className="fnt">{DASH}</span>}
            {g.ae ? (
              <span className="sub">
                lowest z, last {g.ae.window ?? DASH} {units(g.unit)}
                {g.ae.alert ? ' · alert' : ''}
              </span>
            ) : null}
          </td>
          <td className="num">
            {g.q2 ? (
              <>
                {fixed(g.q2.real)} vs {fixed(g.q2.threshold)}
                <span className="sub">
                  {g.q2.status} · {g.q2.seeds ?? DASH} seeds
                  {g.q2.fewWindows ? ` · ${g.q2.windowsTest ?? DASH} test windows (under ${g.q2.minWindows ?? DASH})` : ''}
                </span>
              </>
            ) : (
              <span className="fnt">{DASH}</span>
            )}
          </td>
        </tr>
      ))}
    </>
  );
}

function RandomnessDrawer({ game, meta, onClose }: { game: string; meta: RandomnessView['meta']; onClose: () => void }) {
  const [scope, setScope] = useScope();
  const load = useLoad(() => getAdminRandomnessGame(game, scope), [game, scope]);
  const d = load.kind === 'ok' ? load.data : null;
  const t = meta.tripwires;
  return (
    <Drawer title={game} sub={d ? `${d.group} · randomness watch · ranking ${stamp(d.generatedAt)} · Brussels time` : 'Randomness watch'} onClose={onClose}>
      {load.kind === 'loading' ? <Loading /> : null}
      {load.kind === 'error' ? <Failed error={load.error} retry={load.retry} bare /> : null}
      {d ? (
        <>
          <Section title="Tripwires" aside={<StatusCell g={d} />}>
            {d.entropy !== null ? (
              <p className="line">
                Entropy <b>{fixed(d.entropy, 4)}</b> (1 = even; tripwire below {fixed(t.entropyBelow, 2)}) over the newest {d.window ?? DASH} of {d.drawsTotal ?? DASH}{' '}
                {units(d.unit)}
              </p>
            ) : (
              <p className="line">No entropy or KL record: this game carries the autoencoder only.</p>
            )}
            {d.klHistory !== null ? (
              <p className="line">
                KL vs history <b>{fixed(d.klHistory, 4)}</b> (tripwire above {fixed(t.klHistoryAbove, 2)}) · vs uniform {fixed(d.klUniform, 4)}, no tripwire · same{' '}
                {d.window ?? DASH} {units(d.unit)}
              </p>
            ) : null}
            {d.ae ? (
              <p className="line">
                Autoencoder lowest z <b>{signedFixed(d.ae.minZ, 2)}</b> over the last {d.ae.window ?? DASH} {units(d.unit)} (tripwire below {signedFixed(t.aeBelow, 0)}) ·
                latest z {signedFixed(d.ae.latestZ, 2)} on {day(d.ae.date)}
                {d.ae.alert ? ' · alert' : ''}
              </p>
            ) : null}
          </Section>
          {d.trend.length ? (
            <Section title="Trend">
              <p className="line">Each line: the newest {d.window ?? DASH} {units(d.unit)} up to that day.</p>
              <Scroll>
                <table className="tbl">
                  <thead>
                    <tr>
                      <th>Up to</th>
                      <th className="num">Entropy</th>
                      <th className="num">KL vs history</th>
                      <th className="num">KL vs uniform</th>
                    </tr>
                  </thead>
                  <tbody>
                    {d.trend.map((x, i) => (
                      <tr key={`${x.end}-${i}`}>
                        <td>{day(x.end)}</td>
                        <td className="num">{fixed(x.entropy)}</td>
                        <td className="num">{fixed(x.klHistory, 4)}</td>
                        <td className="num">{fixed(x.klUniform, 4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Scroll>
            </Section>
          ) : null}
          <Section title="Q2 randomness" aside={d.q2 ? <Chip kind={d.q2.status === 'above' ? 'hi' : d.q2.status === 'too few draws' ? 'dim' : 'flat'}>{d.q2.status}</Chip> : null}>
            {d.q2 ? (
              <p className="line">
                {/* k: the record's own, from its seed count (rule 2, 1 Nov 2026 on); 2 under the old rule */}
                <b>{fixed(d.q2.real)}</b> vs threshold <b>{fixed(d.q2.threshold)}</b>
                {d.q2.thresholdSe !== null ? ` ± ${fixed(d.q2.thresholdSe)}` : ''} (null {fixed(d.q2.nullMean)} ± {fixed(d.q2.nullSd)}, +{' '}
                {d.q2.bandK !== null ? fixed(d.q2.bandK, 2) : d.q2.rule >= 2 ? DASH : '2'} sd) · shuffled {fixed(d.q2.shuffled)}
                {d.q2.shuffledThreshold !== null ? `, bound ${fixed(d.q2.shuffledThreshold)}` : ''} · {d.q2.seeds ?? DASH} seeds · {d.q2.windowsTest ?? DASH} test windows
                {d.q2.fewWindows ? ` (under ${d.q2.minWindows ?? DASH}: too few draws)` : ''}
                {d.q2.status === 'failed' ? ' · failed: a comparison scored under 2 seeds' : ''} · {stamp(d.q2.at)}
              </p>
            ) : (
              <p className="line">No Q2 record for this game yet.</p>
            )}
          </Section>
          <Section
            title="Per-model KL"
            aside={
              <>
                <Chip>
                  {d.shown} of {d.of} KL rows{load.busy ? ' · updating' : ''}
                </Chip>
                {d.of ? <ScopeSwitch scope={scope} onChange={setScope} /> : null}
              </>
            }
          >
            {d.of === 0 ? (
              <p className="line">No per-model KL for this game.</p>
            ) : (
              <>
                <p className="line">{meta.modelKl}: information only, lowest first. Over the same {d.window ?? DASH} {units(d.unit)}.</p>
                {d.scope === 'top' && d.topMissing.length ? (
                  <p className="line">
                    {d.topMissing.length} of the top 5 {d.topMissing.length === 1 ? 'has' : 'have'} no KL row over this window: {d.topMissing.map(short).join(', ')}.
                  </p>
                ) : null}
                <Scroll>
                  <table className="tbl">
                    <thead>
                      <tr>
                        <th>Row</th>
                        <th className="num">KL (predicted || real)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {d.models.map((m) => (
                        <tr key={m.row}>
                          <td>
                            {short(m.row)}
                            {m.top && d.scope === 'all' ? <span className="tag">top 5</span> : null}
                          </td>
                          <td className="num">{fixed(m.kl, 4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Scroll>
              </>
            )}
          </Section>
        </>
      ) : null}
    </Drawer>
  );
}
