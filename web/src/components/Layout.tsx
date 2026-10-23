import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router'
import { useAuth } from '../auth/context'
import { Icon } from './Icon'

const NAV = [
  { to: '/', label: 'Home', end: true },
  { to: '/handwriting', label: 'Handwriting' },
  { to: '/reading', label: 'Reading' },
  { to: '/math', label: 'Math' },
  { to: '/talk', label: 'Talk' },
  { to: '/progress', label: 'Progress' },
  { to: '/settings', label: 'Settings' },
]

export function BrandMark() {
  return (
    <svg viewBox="0 0 64 64" aria-hidden="true" focusable="false">
      <rect width="64" height="64" rx="14" fill="var(--accent)" />
      <path
        d="M18 44c6-2 9-14 12-22s6-8 7-4-4 16-1 20 9-4 12-10"
        fill="none"
        stroke="var(--accent-contrast)"
        strokeWidth="5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

export function Layout() {
  const { user, logout } = useAuth()
  const location = useLocation()
  const mainRef = useRef<HTMLElement>(null)
  const firstRender = useRef(true)
  // On narrow screens the links fold behind a Menu button; wide screens always show them.
  const [menuOpen, setMenuOpen] = useState(false)
  const toggleRef = useRef<HTMLButtonElement>(null)

  const closeMenuOnEscape = (event: KeyboardEvent) => {
    if (event.key === 'Escape' && menuOpen) {
      setMenuOpen(false)
      toggleRef.current?.focus()
    }
  }

  // Move focus to the page content after navigation so screen-reader and
  // keyboard users start at the new page instead of the old link, and show the
  // page from the top (navigation stays in view).
  useEffect(() => {
    if (firstRender.current) {
      firstRender.current = false
      return
    }
    mainRef.current?.focus({ preventScroll: true })
    window.scrollTo({ top: 0 })
  }, [location.pathname])

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to main content
      </a>
      <header className="app-header" onKeyDown={closeMenuOnEscape}>
        <div className="app-header__inner">
          <NavLink to="/" className="brand" aria-label="Legible home" onClick={() => setMenuOpen(false)}>
            <BrandMark />
            <span>Legible</span>
          </NavLink>
          <button
            ref={toggleRef}
            type="button"
            className="btn btn--ghost menu-toggle"
            aria-expanded={menuOpen}
            aria-controls="app-menu"
            onClick={() => setMenuOpen((open) => !open)}
          >
            <Icon name={menuOpen ? 'cross' : 'menu'} />
            Menu
          </button>
          <div id="app-menu" className="app-menu" data-open={menuOpen}>
            <nav className="main-nav" aria-label="Main">
              <ul>
                {NAV.map((item) => (
                  <li key={item.to}>
                    <NavLink to={item.to} end={item.end} onClick={() => setMenuOpen(false)}>
                      {item.label}
                    </NavLink>
                  </li>
                ))}
              </ul>
            </nav>
            {user && (
              <div className="header-user">
                <span className="small">Hi, {user.display_name}</span>
                <button type="button" className="btn btn--ghost" onClick={logout}>
                  Log out
                </button>
              </div>
            )}
          </div>
        </div>
      </header>
      <main id="main" ref={mainRef} tabIndex={-1}>
        <Outlet />
      </main>
    </>
  )
}
