import { Navigate, Route, Routes } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import MarketPage from './pages/MarketPage';

export default function App() {
  return (
    <div className="shell">
      <Sidebar />
      <div className="wrap">
        <Routes>
          <Route path="/" element={<Navigate to="/crypto" replace />} />
          <Route path="/:market" element={<MarketPage />} />
          <Route path="*" element={<Navigate to="/crypto" replace />} />
        </Routes>
      </div>
    </div>
  );
}
