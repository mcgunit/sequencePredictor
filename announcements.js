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
