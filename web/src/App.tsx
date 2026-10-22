import { Navigate, Route, Routes } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import WhatsNewDialog from './components/WhatsNewDialog';
import AdminGuard from './components/admin/States';
import AccountPage from './pages/AccountPage';
import MarketPage from './pages/MarketPage';
import WhatsNewPage from './pages/WhatsNewPage';
import ControlsPage from './pages/admin/ControlsPage';
import JobsPage from './pages/admin/JobsPage';
import ModelsPage from './pages/admin/ModelsPage';
import PeoplePage from './pages/admin/PeoplePage';
import PhaseShiftPage from './pages/admin/PhaseShiftPage';
import RandomnessPage from './pages/admin/RandomnessPage';
import TuningPage from './pages/admin/TuningPage';
import WeekPage from './pages/admin/WeekPage';

export default function App() {
  return (
    <div className="shell">
      <Sidebar />
      <main className="wrap">
        <Routes>
          <Route path="/" element={<Navigate to="/crypto" replace />} />
          {/* the admin section: before /:market, and only for the administrator (the API answers 403 anyway) */}
          <Route path="/admin" element={<AdminGuard><WeekPage /></AdminGuard>} />
          <Route path="/admin/tuning" element={<AdminGuard><TuningPage /></AdminGuard>} />
          <Route path="/admin/controls" element={<AdminGuard><ControlsPage /></AdminGuard>} />
          <Route path="/admin/models" element={<AdminGuard><ModelsPage /></AdminGuard>} />
          <Route path="/admin/phase-shift" element={<AdminGuard><PhaseShiftPage /></AdminGuard>} />
          <Route path="/admin/randomness" element={<AdminGuard><RandomnessPage /></AdminGuard>} />
          <Route path="/admin/jobs" element={<AdminGuard><JobsPage /></AdminGuard>} />
          <Route path="/admin/people" element={<AdminGuard><PeoplePage /></AdminGuard>} />
          <Route path="/admin/*" element={<Navigate to="/admin" replace />} />
          {/* every signed-in reader's own pages (the API answers only for the reader) */}
          <Route path="/account" element={<AccountPage />} />
          <Route path="/whats-new" element={<WhatsNewPage />} />
          <Route path="/:market" element={<MarketPage />} />
          <Route path="*" element={<Navigate to="/crypto" replace />} />
        </Routes>
      </main>
      {/* the introduction, or the major notes, on whichever page a reader lands (the old pages' dialog) */}
      <WhatsNewDialog />
    </div>
  );
}
