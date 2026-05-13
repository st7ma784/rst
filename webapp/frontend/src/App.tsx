import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

import MainLayout from './components/MainLayout';
import HomePage from './pages/HomePage';
import ProcessingPage from './pages/ProcessingPage';
import VisualizationPage from './pages/VisualizationPage';
import RemoteComputePage from './pages/RemoteComputePage';
import JobsPage from './pages/JobsPage';
import ComparisonPage from './pages/ComparisonPage';
import SettingsPage from './pages/SettingsPage';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#2196f3',
    },
    secondary: {
      main: '#f50057',
    },
    background: {
      default: '#0a1929',
      paper: '#132f4c',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 500,
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <MainLayout>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/processing" element={<ProcessingPage />} />
            <Route path="/jobs" element={<JobsPage />} />
            <Route path="/compare" element={<ComparisonPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/visualization/:jobId" element={<VisualizationPage />} />
            <Route path="/remote" element={<RemoteComputePage />} />
          </Routes>
        </MainLayout>
      </Router>
    </ThemeProvider>
  );
}

export default App;
