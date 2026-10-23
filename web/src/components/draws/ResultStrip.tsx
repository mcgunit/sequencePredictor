import type { ReactNode } from 'react';
import type { DrawResult, Shape } from '../../workbench';
import { specialsLabel } from './text';

// The mark letters of draws.js, as a cell's class and the words a screen reader hears (the Hits column says the same)
const MARK_CLASS: Record<string, string> = { m: 'm', b: 'b', l: 'l', r: 'r', s: 's', z: 'z', a: 'a' };
const MARK_WORDS: Record<string, string> = { m: 'hit', b: 'bonus hit', l: 'leading run', r: 'trailing run', s: 'special hit', z: 'sign hit', a: 'one bin off' };

// One number as a cell. mark: a draws.js letter ('-' or '' for none); extra: a multi-pick 7th-9th number (dashed).
export function Nb({ v, mark = '', extra, drawn, word }: { v: ReactNode; mark?: string; extra?: boolean; drawn?: boolean; word?: boolean }) {
  const cls = ['nb', drawn ? 'drawn' : '', MARK_CLASS[mark] ?? '', extra ? 'x' : '', word ? 'word' : ''].filter(Boolean).join(' ');
  return (
    <span className={cls}>
      {v}
      {MARK_WORDS[mark] ? <span className="vh"> ({MARK_WORDS[mark]})</span> : null}
    </span>
  );
}

// A positional cell with its slot's name over it (a market's instrument) and an optional line under it (the arrow)
export function Slot({ label, children, under }: { label?: string; children: ReactNode; under?: string }) {
  return (
    <span className="slot">
      {label !== undefined ? <small>{label}</small> : null}
      {children}
      {under !== undefined ? <small className="ar">{under}</small> : null}
    </span>
  );
}

// The drawn row: the mains, then the specials (stars, dream, Viking, Joker+'s sign) or the lotto bonus as their own
// labelled group; a market game's bins under their instruments. compact: one line for the history.
export default function ResultStrip({ shape, result, compact }: { shape: Shape; result: DrawResult | null; compact?: boolean }) {
  if (!result || !result.mains.length) return <span className="mut">no result stored</span>;
  const sm = compact ? ' sm' : '';
  const mains =
    shape.family === 'market' ? (
      <span className={`nums${sm}`}>
        {result.mains.map((v, i) => (
          <Slot key={i} label={shape.slots[i] ?? String(i + 1)}>
            <Nb v={v} drawn />
          </Slot>
        ))}
      </span>
    ) : (
      <span className={`nums${sm}`}>
        {result.mains.map((v, i) => (
          <Nb key={i} v={v} drawn />
        ))}
      </span>
    );
  const extras = shape.family === 'jokerplus' ? (result.signName ? [result.signName] : result.specials) : shape.bonus ? result.bonus : result.specials;
  const label = specialsLabel(shape);
  const second = extras.length ? (
    <span className={`nums${sm}`}>
      {extras.map((v, i) => (
        <Nb key={i} v={v} drawn word={typeof v === 'string'} />
      ))}
    </span>
  ) : null;
  if (compact) {
    return (
      <span className="nums sm">
        {mains}
        {second ? (
          <>
            <span className="gap" />
            <span className="vh">{label}: </span>
            {second}
          </>
        ) : null}
      </span>
    );
  }
  return (
    <div className="rstrip">
      <div className="rgrp">
        <small>{shape.family === 'market' ? 'Bins' : shape.family === 'jokerplus' || shape.family === 'pick3' ? 'Digits' : 'Numbers'}</small>
        {mains}
      </div>
      {second ? (
        <div className="rgrp">
          <small>{label}</small>
          {second}
        </div>
      ) : null}
    </div>
  );
}
