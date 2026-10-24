// The reader's colour scheme (owner, 24 Oct 2026): auto - the system's -, light or dark, kept in this browser
// (localStorage) and applied as data-theme on <html>, which styles.css's tokens already follow. index.html applies a
// stored choice before the first paint; a change here fires 'themechange' so the charts redraw in the new colours.
export type ThemeChoice = 'auto' | 'light' | 'dark';

export const THEME_KEY = 'predictor.theme';
export const THEME_EVENT = 'themechange';

export function readTheme(): ThemeChoice {
  try {
    const v = window.localStorage.getItem(THEME_KEY);
    return v === 'light' || v === 'dark' ? v : 'auto';
  } catch {
    return 'auto';
  }
}

export function setTheme(choice: ThemeChoice): void {
  try {
    if (choice === 'auto') window.localStorage.removeItem(THEME_KEY);
    else window.localStorage.setItem(THEME_KEY, choice);
  } catch {
    /* a private window or blocked storage: the choice holds for this page only */
  }
  const root = document.documentElement;
  if (choice === 'auto') root.removeAttribute('data-theme');
  else root.setAttribute('data-theme', choice);
  window.dispatchEvent(new Event(THEME_EVENT));
}
