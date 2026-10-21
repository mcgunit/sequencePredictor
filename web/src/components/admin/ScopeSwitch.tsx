import type { Scope } from '../../api';

// All / Top 5: every row, or only the game's top five (admin.js usersTop: the first five ranked rows in the newest
// day's table - for a lottery game the five a user sees; a market game's users' page follows its best book instead).
export default function ScopeSwitch({ scope, onChange }: { scope: Scope; onChange: (s: Scope) => void }) {
  return (
    <div className="seg" role="group" aria-label="Rows shown">
      <button type="button" aria-pressed={scope === 'all'} onClick={() => onChange('all')}>
        All
      </button>
      <button type="button" aria-pressed={scope === 'top'} onClick={() => onChange('top')}>
        Top 5
      </button>
    </div>
  );
}
