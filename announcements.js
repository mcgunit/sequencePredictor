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
    id: '2026-10-04-market-plan-open-close',
    date: '2026-10-04',
    level: 'minor',
    audience: 'all',
    title: 'Crypto and Shares: a Today\'s plan card, an open-to-close book for shares, one Background card',
    body: [
      'Each market page now opens with Today\'s plan: per coin or share the best book\'s call, the price band the close must land in for it to be right, and the two orders that match how the page scores it, with the Belgian hours for that date - for a share an order that fills at the New York open and one that fills at the close; for a coin buy now and sell at the close of the UTC day. It also says when a position would be held instead of sold, and how many models call the day up.',
      'The shares book has a third rule, open to close: the stake bought at the session\'s open and sold at its close - the one a reader can actually follow, since the ticket is up hours before New York opens. The daily and hold books buy at the previous close, which nobody reading the page can do; the page says so. The six supporting cards now sit inside one collapsed Background card; the charts and the Paper trading card are unchanged.',
    ],
    link: { href: '/markets/shares', label: 'Shares page' },
  },
  {
    id: '2026-10-03-market-rows-tuned',
    date: '2026-10-03',
    level: 'minor',
    audience: 'admin',
    title: 'Crypto and Shares: the GARCH and Regime HMM rows are tuned weekly, by the proper score',
    body: [
      'The Saturday tuning chain now has two strategies for the market games only, Garch and RegimeHmm. Their trials are scored by the log-score of the probability the row gave the bin that happened (uniform is -2.303), not by hit rate, and a new set of parameters is served only when it beats both what is served today and the untuned defaults on the same window - the same gate as every other row. The decision is recorded under tuningGate in bestParams_crypto.json and bestParams_shares.json.',
      'The three Regime HMM rows share one set of knobs, so the two ablation rows stay ablations. The first run is next Saturday, or by hand: python3 HyperoptStatistics.py -g crypto,shares -s Garch,RegimeHmm.',
    ],
    link: { href: '/admin/jobs', label: 'Open Jobs' },
  },
  {
    id: '2026-10-03-lotto-multipick-top5',
    date: '2026-10-03',
    level: 'major',
    audience: 'all',
    title: 'Lotto: three extra numbers per row for a system play, and the top five models by default',
    body: [
      'Every Lotto row still predicts six numbers, now in the order of the model\'s own probability, highest first - not small to large. Next to them, three shaded numbers: the model\'s next most probable ones, for a system play of 7, 8 or 9 numbers (7 grids = 10.50 EUR, 28 grids = 42 EUR, 84 grids = 126 EUR). The hits column shows the hits of the six and, after the dot, the hits among all nine.',
      'More numbers win more by arithmetic alone: nine numbers of any kind hit the smallest prize 8.4% of the time, six numbers 2.4%. The History page has a new card that reads each model against exactly those levels - and how often its 7th, 8th and 9th number were drawn, against the 13% any number gets by luck. That is the research question, not the raw hit count.',
      'The prediction tables now show the top five models of the History ranking by default, with a box to show all; the administrator sees all rows with the box ticked.',
    ],
    link: { href: '/lottery', label: 'New predictions' },
  },
  {
    id: '2026-10-02-paper-trading',
    date: '2026-10-02',
    level: 'minor',
    audience: 'all',
    title: 'Crypto and Shares: the predictions are paper-traded in money',
    body: [
      'Every call is now turned into money with one fixed rule: when a model says a coin or share will go up, 100 USDT or USD is bought at the previous close and sold at the day\'s close, with a 0.1% fee on each leg; when it says flat or down, nothing is bought. The models table shows each model\'s total, win rate and money per trade, and a new Paper trading card draws every model\'s book against the market - buying everything every day with the same stake, which a model has to beat before its book means anything.',
      'A second rule sits next to it on the same card: hold while up - the position is kept as long as the next day\'s call is up again and sold when it stops, so a run of up days pays one fee pair and compounds; its benchmark is buying everything on the first day and holding. It is paper money: no slippage, fills at the close. Betting on falls is the next step.',
    ],
    link: { href: '/markets/crypto', label: 'Crypto page' },
  },
  {
    id: '2026-10-01-three-sections',
    date: '2026-10-01',
    level: 'major',
    audience: 'all',
    title: 'The site is organised in three sections: Lottery games, Crypto and Shares',
    body: [
      'The home page now asks which kind of sequence you want to look at. Lottery games is where the former home page went: every model\'s ticket for the next draw of the seven games, with the History next to it. Crypto and Shares have their own pages, as before, and the top bar shows the three sections.',
      'The market pages explain themselves now. A first card, "How a day becomes a draw", takes the newest day and shows, coin by coin, the real return, the bin it fell in and the interval that bin stood for - so the digits on the game view are readable. A "Day by day" card lists the settled days as returns and bins; open a day for every model\'s ticket with its hits.',
      'The game view of a market (the digit tables under History) is still there, reached from the market page, with the coin or share symbols (BTC, ETH, ... / NVDA, AAPL, ...) as column headings and a note on what the digits are.',
    ],
    link: { href: '/', label: 'Home' },
  },
  {
    id: '2026-09-30-market-rows',
    date: '2026-09-30',
    level: 'minor',
    audience: 'all',
    title: 'Crypto and Shares: two rows built for markets, a proper score, and the controls',
    body: [
      'Two rows now model the returns themselves rather than the bins: a GARCH row, which forecasts each coin\'s or share\'s volatility for the next day and turns it into bin probabilities (it predicts how large the move is, not its direction - the known predictable part of returns), and a Regime HMM row, which reads the whole market as switching between a few regimes - calm, normal, turbulent - and predicts from the regime it believes the market is in. Two stripped-down versions of the regime row run next to it so that the difference between them says what the regimes are made of.',
      'Before any hit rate or paper profit is read, the market pages judge every row by a proper score: how much probability it gave the bin that then happened, over hundreds of past days, with an interval against the GARCH row. A row counts as carrying information beyond volatility only when that interval lies above zero. A new card shows the verdicts, another shows which regime the model believes the market is in today.',
      'The two markets joined the Sunday control experiments: their fair history is a random walk with the same volatility and no memory, so the null band and the randomness verdict appear for them on the History page like for every lottery game, once the first Sunday has run.',
    ],
    link: { href: '/markets/crypto', label: 'Crypto page' },
  },
  {
    id: '2026-09-30-crypto-and-shares',
    date: '2026-09-30',
    level: 'major',
    audience: 'all',
    title: 'Two new sections: Crypto and Shares - the same models, on market prices',
    body: [
      'The research question of roadmap item 4: do the principles that fail to find structure in a fair lottery find any in market prices? Five coins (BTC, ETH, BNB, XRP, SOL against USDT) and four shares (NVDA, AAPL, MSFT, ASML) are tracked as two more games: each instrument\'s next-day return is cut into ten equiprobable bins, and every model predicts one bin per instrument, every day, like a digit of pick3.',
      'The two pages in the top navigation show, per model, how often the exact bin, the adjacent bin and the direction were right against chance (10%, 28%, 50%) and the paper profit of a fixed rule with a fee; per coin or share, the actual price with the predicted course of the best models drawn on it and the next day\'s prediction of every model as a price with its band. Results settle the morning after, when the day\'s bar has closed.',
      'A predictor, not a trading bot: nothing here is advice, and the same controls that judge the lottery rows will judge these before any number above chance is read as predictability. The first daily run builds a month of history for both markets, so the pages show settled results from the first morning; the deep-learning rows join from that day on.',
    ],
    link: { href: '/markets/crypto', label: 'Crypto page' },
  },
  {
    id: '2026-09-28-irrelevant-feature-control',
    date: '2026-09-28',
    level: 'minor',
    audience: 'all',
    title: 'A fourth control: can the meta-learners tell a real base model from noise?',
    body: [
      'Every Sunday, after the randomness test, the meta-learner variants of every game are refitted with three noise columns appended - shuffled copies of real base-model scores, so they look exactly like a base model and mean nothing. A variant that gives them weight is fitting noise, and its held-out numbers are then the same selection effect the null band measures at the row level.',
      'The new Meta-learner feature control card on the History page shows, per game and variant, whether the noise was ignored or fitted, what share of the model\'s attribution went to it, what it cost in held-out AUC, and which base models matter more than noise does - the columns above the noise band.',
      'It reads the table the Saturday chain cached and refits only, so it costs refits, not a backtest. First results on 4 October 2026.',
    ],
    link: { href: '/database', label: 'History page' },
  },
  {
    id: '2026-09-28-classical-svm-control',
    date: '2026-09-28',
    level: 'minor',
    audience: 'all',
    title: 'A classical control row for the quantum kernel',
    body: [
      'ClassicalSVM Model is a new tracked row for every game: the quantum-kernel meta-learner with its one quantum piece replaced by a classical RBF kernel - same features, same reduction, same balanced subsampling and calibrated classifier, with its own weekly-tuned kernel width, C and sample cap. It appears on the History page as the quantum rows do, from the first weekly meta-learner retrain that writes its artifact, and the weekly quantum tuner tunes it alongside them.',
      'Whether the quantum kernel ever separates from this control, and either from the null band, is the question the quantum track was asking; the two rows now answer it side by side over time.',
      'Also this week: the variational classifier\'s gradient is checked against finite differences by the test suite - a check that at once found two rotations the circuit could never use, because it read out the qubit the entangling ring writes last; the readout moved, the check now insists every angle is live, and the VQC rows pick up the corrected circuit at the next weekly retrain. Every meta-learner artifact records the library versions it was written with, and the quantum rows\' encoding-scale search extends down to 0.1, where the tuned values were pressing against the old lower bound.',
    ],
    link: { href: '/database', label: 'History page' },
  },
  {
    id: '2026-09-28-council-head-chairs',
    date: '2026-09-28',
    level: 'minor',
    audience: 'all',
    title: 'The council head judges the members instead of repeating one',
    body: [
      'Asked a question the members answer differently, the small head used to hand back one member\'s answer word for word. Its prompt now says outright that a copied answer is not a chair\'s answer, and that a question about the responder itself has no single answer for a panel of different models.',
      'The orchestrator also checks the head\'s reply against every member\'s answer and, when it is a copy, asks the head once more with the copy named. The second reply stands; if that call fails the first is kept, and a grey note under the verdict says what happened either way. The extra call happens only when the first reply copied.',
      'The head reads the members anonymised and in a shuffled order, so "Member 2" in its verdict meant nothing on the page. Each member\'s card now shows the number the head knows it by, from the moment the head starts reading.',
    ],
    link: { href: '/council', label: 'Council page' },
  },
  {
    id: '2026-09-28-since-the-freeze',
    date: '2026-09-28',
    level: 'minor',
    audience: 'all',
    title: 'The best model per game, counted since the design freeze',
    body: [
      'The Best model per game card now shows, next to the all-history ranking, the best row counted only over the days since a declared date - and each model\'s value and draw count since that date in the expanded ranking.',
      'Nothing was rebuilt. Every stored day was predicted before its draw, so counting from the date the design was frozen turns the record after it into that design\'s own track record, with the same minimum-draws guard as before.',
      'The date is declared in since.json at the repository root. Until one is declared the column reads "not declared"; once declared, a game reads "no scored draw since that date yet" until its first draw on or after the date is scored. The null band is not applied to the since column, because a handful of draws clears a band measured over long histories by luck alone.',
    ],
    link: { href: '/database', label: 'History page' },
  },
  {
    id: '2026-09-28-tuning-controls-calibration-lockbox',
    date: '2026-09-28',
    level: 'minor',
    audience: 'admin',
    title: 'Three research controls: tuning on nothing, calibration metrics, a lockbox',
    body: [
      'TuningControls.py runs the statistical tuner on a history with nothing in it and compares the best trial with the untuned defaults, next to the gate\'s real-history gain: what the hyperopt gains by choosing alone, so a real tuning gain can be read against it.',
      'The weekly meta-learner retrain now reports and stores per-number calibration and ranking metrics - Brier, log-loss, reliability bins, and the top-ticket hits per day against chance - in the artifacts and in meta_learner_metrics.json, instead of a printed accuracy line.',
      'A lockbox period can be declared in lockbox.json: the meta-learner trainer and the quantum tuner never fit or tune on its days, and TrainMetaLearner.py --lockbox-report scores the frozen artifacts on it once.',
    ],
    link: { href: '/admin/jobs', label: 'Jobs page' },
  },
  {
    id: '2026-09-27-control-experiments',
    date: '2026-09-27',
    level: 'minor',
    audience: 'all',
    title: 'The History page now shows what the best row scores on nothing',
    body: [
      'Two control experiments run every Sunday after the predictor. The first scores every tracked row on histories with provably nothing in them - fair synthetic draws and the real draws shuffled - so the Best model per game card can show the null band: the score the best of the rows reaches by selection alone. Rows within it are greyed and marked, like rows with too few draws.',
      'The second asks a suite of classifiers to tell windows of real draws from fair simulated ones, held against the same suite on two fair histories and on the shuffled real draws. Its verdict is a new column on the Randomness watch card, next to the daily entropy tripwire - the controlled test the tripwire cannot give.',
      'Both are measurements of the rows, never inputs to them. They run in their own weekly plan that starts only after Sunday\'s predictor has finished, so they are never part of Saturday\'s tuning chain and never push Sunday\'s predictions behind them; like the chain, they queue in the same order, so a run that overruns into Monday morning makes that day\'s predictor wait for it.',
    ],
    link: { href: '/database', label: 'History page' },
  },
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
  intro: 'This is a research project: it runs many prediction models against real lottery draws and, since autumn 2026, against crypto and share prices, and tracks, honestly, how each one performs. It is not advice, and the models are not expected to beat any of it - measuring that is the point.',
  steps: [
    {
      audience: 'all',
      title: 'Three sections',
      body: 'The home page offers three sections - Lottery games, Crypto and Shares - and the top bar repeats them as Lottery, Crypto and Shares. Each section has its own pages; what is predicted differs, the models and the honesty rules are the same.',
    },
    {
      audience: 'all',
      title: 'Lottery games',
      body: 'New predictions has one card per game. Open a card to see every model\'s ticket for the next draw, and the date of the draw they are for. History keeps every past draw with what each model predicted, how many numbers it hit and what that would have paid; its cards rank the models, show which combinations do best together, and watch whether the draws still look random.',
    },
    {
      audience: 'all',
      title: 'Crypto and Shares',
      body: 'Each coin\'s or share\'s next-day return is cut into ten equally likely bins, and every model predicts one bin per instrument, every day - a draw with one slot per coin. The pages show the price with the predicted course drawn on it, how often each model got the bin, the neighbouring bin and the direction right against chance, a paper profit, and the newest thirty settled days as returns and bins. A first card explains how a day becomes a draw, with the newest day as the example.',
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
