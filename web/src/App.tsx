import { lazy, Suspense } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import WhatsNewDialog from './components/WhatsNewDialog';
import AdminGuard, { Loading } from './components/admin/States';
import AccountPage from './pages/AccountPage';
import CouncilPage from './pages/CouncilPage';
import HomePage from './pages/HomePage';
import LotteryDrawPage from './pages/LotteryDrawPage';
import LotteryGamePage from './pages/LotteryGamePage';
import LotteryPage from './pages/LotteryPage';
import MarketPage from './pages/MarketPage';
import WhatsNewPage from './pages/WhatsNewPage';

// The admin pages load on first use, in their own chunks: a user's download carries the user pages only.
const ControlsPage = lazy(() => import('./pages/admin/ControlsPage'));
const DrawPage = lazy(() => import('./pages/admin/DrawPage'));
const GameHistoryPage = lazy(() => import('./pages/admin/GameHistoryPage'));
const GamesPage = lazy(() => import('./pages/admin/GamesPage'));
const JobsPage = lazy(() => import('./pages/admin/JobsPage'));
const MarketWorkbenchPage = lazy(() => import('./pages/admin/MarketWorkbenchPage'));
const ModelsPage = lazy(() => import('./pages/admin/ModelsPage'));
const PeoplePage = lazy(() => import('./pages/admin/PeoplePage'));
const PhaseShiftPage = lazy(() => import('./pages/admin/PhaseShiftPage'));
const RandomnessPage = lazy(() => import('./pages/admin/RandomnessPage'));
const TuningPage = lazy(() => import('./pages/admin/TuningPage'));
const WeekPage = lazy(() => import('./pages/admin/WeekPage'));

export default function App() {
  return (
    <div className="shell">
      <Sidebar />
      <main className="wrap">
        <Suspense fallback={<Loading />}>
        <Routes>
          {/* Home: the Lottery, Crypto and Shares sections and the Council (the owner's rule of 1 Oct 2026) */}
          <Route path="/" element={<HomePage />} />
          {/* the admin section: before /:market, and only for the administrator (the API answers 403 anyway) */}
          <Route path="/admin" element={<AdminGuard><WeekPage /></AdminGuard>} />
          <Route path="/admin/tuning" element={<AdminGuard><TuningPage /></AdminGuard>} />
          <Route path="/admin/controls" element={<AdminGuard><ControlsPage /></AdminGuard>} />
          <Route path="/admin/models" element={<AdminGuard><ModelsPage /></AdminGuard>} />
          <Route path="/admin/phase-shift" element={<AdminGuard><PhaseShiftPage /></AdminGuard>} />
          <Route path="/admin/randomness" element={<AdminGuard><RandomnessPage /></AdminGuard>} />
          {/* the Workbench: every past draw of a game, one draw, a market's workbench */}
          <Route path="/admin/games" element={<AdminGuard><GamesPage /></AdminGuard>} />
          <Route path="/admin/games/:game" element={<AdminGuard><GameHistoryPage /></AdminGuard>} />
          <Route path="/admin/games/:game/:date" element={<AdminGuard><DrawPage /></AdminGuard>} />
          <Route path="/admin/markets/:market" element={<AdminGuard><MarketWorkbenchPage /></AdminGuard>} />
          <Route path="/admin/jobs" element={<AdminGuard><JobsPage /></AdminGuard>} />
          <Route path="/admin/people" element={<AdminGuard><PeoplePage /></AdminGuard>} />
          <Route path="/admin/*" element={<Navigate to="/admin" replace />} />
          {/* the users' Lottery: today's top 5 per game, each past draw's five as ranked before it (lottery.js) */}
          <Route path="/lottery" element={<LotteryPage />} />
          <Route path="/lottery/:game" element={<LotteryGamePage />} />
          <Route path="/lottery/:game/:date" element={<LotteryDrawPage />} />
          <Route path="/council" element={<CouncilPage />} />
          {/* every signed-in reader's own pages (the API answers only for the reader) */}
          <Route path="/account" element={<AccountPage />} />
          <Route path="/whats-new" element={<WhatsNewPage />} />
          <Route path="/:market" element={<MarketPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
        </Suspense>
      </main>
      {/* the introduction, or the major notes, on whichever page a reader lands (the old pages' dialog) */}
      <WhatsNewDialog />
    </div>
  );
}
