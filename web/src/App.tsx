import { lazy, Suspense, type ReactNode } from 'react'
import { BrowserRouter, Route, Routes } from 'react-router'
import { AuthProvider } from './auth/AuthContext'
import { RequireAuth } from './auth/RequireAuth'
import { Layout } from './components/Layout'
import { LoadingBlock } from './components/LoadingBlock'
import { HomePage } from './pages/HomePage'
import { LoginPage, RegisterPage } from './pages/AuthPages'
import { NotFoundPage } from './pages/NotFoundPage'

// Feature pages are split out so the first load stays small; Talk in particular
// pulls in the video-call and face-tracking libraries on demand.
const HandwritingPage = lazy(() => import('./pages/HandwritingPage'))
const ReadingPage = lazy(() => import('./pages/ReadingPage'))
const MathPage = lazy(() => import('./pages/MathPage'))
const TalkPage = lazy(() => import('./pages/TalkPage'))
const ProgressPage = lazy(() => import('./pages/ProgressPage'))
const SettingsPage = lazy(() => import('./pages/SettingsPage'))

function Page({ children }: { children: ReactNode }) {
  return <Suspense fallback={<LoadingBlock />}>{children}</Suspense>
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route
            element={
              <RequireAuth>
                <Layout />
              </RequireAuth>
            }
          >
            <Route index element={<HomePage />} />
            <Route path="handwriting" element={<Page><HandwritingPage /></Page>} />
            <Route path="reading" element={<Page><ReadingPage /></Page>} />
            <Route path="math" element={<Page><MathPage /></Page>} />
            <Route path="talk" element={<Page><TalkPage /></Page>} />
            <Route path="progress" element={<Page><ProgressPage /></Page>} />
            <Route path="settings" element={<Page><SettingsPage /></Page>} />
            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  )
}
