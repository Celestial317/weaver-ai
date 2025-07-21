import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import VirtualTryOn from './pages/VirtualTryOn';
import AISearch from './pages/AISearch';
import ImageRecommendations from './pages/ImageRecommendations';
import Stylizer from './pages/Stylizer';
import Catalogue from './pages/Catalogue';
import VisualDesigner from './pages/VisualDesigner';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-900">
        <Navbar />
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/virtual-tryon" element={<VirtualTryOn />} />
          <Route path="/ai-search" element={<AISearch />} />
          <Route path="/image-recommendations" element={<ImageRecommendations />} />
          <Route path="/ai-stylizer" element={<Stylizer />} />
          <Route path="/visual-designer" element={<VisualDesigner />} />
          <Route path="/catalogue" element={<Catalogue />} />
        </Routes>
        <Footer />
      </div>
    </Router>
  );
}

export default App;