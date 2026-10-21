import { Navigate, Route, Routes } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import AdminGuard from './components/admin/States';
import MarketPage from './pages/MarketPage';
import ControlsPage from './pages/admin/ControlsPage';
import ModelsPage from './pages/admin/ModelsPage';
import PhaseShiftPage from './pages/admin/PhaseShiftPage';
import RandomnessPage from './pages/admin/RandomnessPage';

export default function App() {
  return (
    <div className="shell">
      <Sidebar />
      <main className="wrap">
        <Routes>
          <Route path="/" element={<Navigate to="/crypto" replace />} />
          {/* the admin section: before /:market, and only for the administrator (the API answers 403 anyway) */}
          <Route path="/admin" element={<Navigate to="/admin/controls" replace />} />
          <Route path="/admin/controls" element={<AdminGuard><ControlsPage /></AdminGuard>} />
          <Route path="/admin/models" element={<AdminGuard><ModelsPage /></AdminGuard>} />
          <Route path="/admin/phase-shift" element={<AdminGuard><PhaseShiftPage /></AdminGuard>} />
          <Route path="/admin/randomness" element={<AdminGuard><RandomnessPage /></AdminGuard>} />
          <Route path="/admin/*" element={<Navigate to="/admin/controls" replace />} />
          <Route path="/:market" element={<MarketPage />} />
          <Route path="*" element={<Navigate to="/crypto" replace />} />
        </Routes>
      </main>
    </div>
  );
}
