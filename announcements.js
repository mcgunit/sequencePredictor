// The one file to edit when a feature ships. Pure data, no requires, so a
// feature author cannot break the server from here.
//
// RULES
//  - `id` is permanent. Never reuse one and never edit one after it is
//    deployed: it is the string every account has recorded as "seen", so
//    changing it makes the note pop up again for everyone.
//  - No HTML in `body` or `title` - every string is escaped when rendered.
//    A link goes in the `link` field.
//  - `level` 'major' opens the dialog once per account; 'minor' only puts a
//    dot on the "What's new" link in the navbar.
//  - `audience` 'all' or 'admin'.
//  - Newest first.
//
// `test/announcements.test.js` checks these rules; run `npm test` before
// deploying a new entry.

const ANNOUNCEMENTS = [
  {
    id: '2026-09-24-tuning-gate',
    date: '2026-09-24',
    level: 'minor',
    audience: 'all',
    title: 'Tuning can no longer be hijacked by one lucky draw',
    body: [
      'The weekly statistical and boosting tuners used to rank their trials by raw profit per bet over 31 days, and to keep the best trial a study had ever seen. One jackpot inside the window decided both: the keno LightGBM parameters served since 10 September scored 5.0 per bet from a single 6/6, the pick3 XGBoost and CatBoost parameters from single straights - each on a window that otherwise lost, and each locked in because no later window could match the luck.',
      'Trials are now scored by a lower confidence bound over the days of the window, with jackpots capped so they count as one good bet; pick3 is scored on digits in the right slot and Joker+ on its leading and trailing runs, which every draw informs. The window is 90 draws instead of 31.',
      'A run\'s best trial replaces the served parameters only if it beats them and the untuned defaults, both re-scored on the same window. Every decision is recorded in the game\'s bestParams file under tuningGate, with the raw profit and the number of lucky strikes next to the score, so a hijack stays visible.',
    ],
    link: { href: '/admin/jobs', label: 'Jobs page' },
  },
  {
    id: '2026-09-23-council-sessions',
    date: '2026-09-23',
    level: 'major',
    audience: 'all',
    title: 'Council conversations are kept, and the table now talks',
    body: [
      'Your council conversations are saved to your account: reload the page or come back tomorrow and they are still there, listed beside the Council page. Start another with New session; delete one you no longer want.',
      'While the council deliberates, each member\'s answer appears at its seat the moment it is ready - a speech bubble on the round table, a line in the voices list under it, and the full text in the conversation - instead of everything arriving at once when the head has finished.',
      'Closing the tab no longer loses an answer: the server keeps polling the council for you and files the answer in your session.',
    ],
    link: { href: '/council', label: 'Open the Council' },
  },
  {
    id: '2026-09-21-foundation-models',
    date: '2026-09-21',
    level: 'minor',
    audience: 'all',
    title: 'A second foundation model joins the predictions',
    body: [
      'Two pretrained time-series models from different labs - Chronos-2 and TimesFM-3 - now each produce a row per game, asked cold for the next value of every drawn position without ever having seen a lottery.',
      'They are shown next to an OrderStatistics Baseline row on purpose. For the games whose numbers come out sorted, the first position is simply the smallest number drawn, so getting its range right is arithmetic rather than prediction: a foundation model is only interesting once it beats that baseline, and where the two models disagree with each other there is probably nothing to find.',
    ],
  },
  {
    id: '2026-09-21-server-schedule',
    date: '2026-09-21',
    level: 'major',
    audience: 'admin',
    title: 'The server keeps the schedule, not crontab',
    body: [
      'The Jobs page now lists the daily predictor and the six jobs of the weekly tuning chain: when each last ran, how long it took, how it ended, at which commit, and a Run now button for each.',
      'The weekly chain no longer waits for a fixed hour - it starts when the Saturday predictor run finishes - and a job that finds the pipeline busy waits its turn instead of skipping the week. A running job survives a deploy and is picked up again when the server comes back.',
      'It starts in dry mode: it records what it would have run and starts nothing, so it is safe next to the crontab entries. To cut over, watch a weekend, remove the three crontab lines, set SCHEDULER=on in .env and restart. To fall back, create config/scheduler.disabled - no restart needed.',
    ],
    link: { href: '/admin/jobs', label: 'Open Jobs' },
  },
  {
    id: '2026-09-20-your-account',
    date: '2026-09-20',
    level: 'major',
    audience: 'all',
    title: 'Your own password, and where you have signed in',
    body: [
      'Your name in the top bar opens your account page. You can change your own password there - you need your current one, and the change signs your other browsers out while keeping this one.',
      'The same page lists the sign-ins recorded for your name, so a session you did not start is visible to you and not only to the administrator.',
    ],
    link: { href: '/account', label: 'Open my account' },
  },
  {
    id: '2026-09-20-draw-dates',
    date: '2026-09-20',
    level: 'major',
    audience: 'all',
    title: 'Every prediction says which draw it is for',
    body: [
      'Each game on the home page now carries the date of the draw its numbers were made for, worked out from that game\'s own draw schedule.',
      'When the date is in the past and shown in red, the newest stored prediction is for a draw that has already taken place - the pipeline has not produced a newer one yet.',
      'While a run is in progress, a banner at the top of the home page says which job is running and for how long.',
    ],
  },
  {
    id: '2026-09-20-jobs-page',
    date: '2026-09-20',
    level: 'major',
    audience: 'admin',
    title: 'The server keeps the Council API running',
    body: [
      'The Jobs page shows the services the web server supervises, starts and restarts them, and tails their logs. The Council API no longer has to be started by hand.',
      'The Activity page lists sign-ins, failed attempts and account changes; the Users page shows when each account was last seen.',
    ],
    link: { href: '/admin/jobs', label: 'Open Jobs' },
  },
  {
    id: '2026-09-17-positional-ensemble',
    date: '2026-09-17',
    level: 'minor',
    audience: 'all',
    title: 'Pick3 and Joker+ have an ensemble row',
    body: [
      'The models\' votes are now combined per digit position for the games where the order matters, so the combined ticket can repeat a digit the way the real draws do.',
    ],
  },
];

// Shown once to an account that has never seen anything here: what the pages
// are, in the order someone would meet them. `audience: 'admin'` steps are
// skipped for a plain user.
const TOUR = {
  title: 'Welcome',
  intro: 'This is a research project: it runs many prediction models against real lottery draws and tracks, honestly, how each one performs. It is not advice, and the models are not expected to beat the games - measuring that is the point.',
  steps: [
    {
      audience: 'all',
      title: 'Predictions',
      body: 'The home page has one card per game. Open a card to see every model\'s ticket for the next draw, and the date of the draw they are for. The chart under a card shows which numbers the models favour together.',
    },
    {
      audience: 'all',
      title: 'History',
      body: 'Every past draw is kept with what each model predicted, how many numbers it hit and what that would have paid. The cards at the top rank the models, show which combinations of models do best together, and watch whether the draws still look random.',
    },
    {
      audience: 'all',
      title: 'Council',
      body: 'A question can be put to several local language models at once; they answer separately and a head model summarises. It needs the model machine to be switched on, and says so when it is not.',
    },
    {
      audience: 'all',
      title: 'Your account',
      body: 'Your name in the top bar opens your account: change your password there, and see the sign-ins recorded for you.',
    },
    {
      audience: 'admin',
      title: 'Users, Jobs and Activity',
      body: 'As administrator you also have the Users page (add or remove accounts, set a password), the Jobs page (the services the server runs, with their logs) and the Activity page (who signed in, and what changed).',
    },
  ],
};

module.exports = { ANNOUNCEMENTS, TOUR };
