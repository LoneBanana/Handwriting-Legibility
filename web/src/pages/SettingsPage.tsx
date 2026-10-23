import { useEffect, useId, useRef, useState, type FormEvent } from 'react'
import { useAuth } from '../auth/context'
import { ErrorNotice } from '../components/ErrorNotice'
import { DeleteAccount } from '../features/settings/DeleteAccount'
import { HandwritingSamples } from '../features/settings/HandwritingSamples'
import '../features/settings/settings.css'
import {
  draftFrom,
  draftPatch,
  FONT_OPTIONS,
  formatScale,
  formatSpacing,
  isDirty,
  mergeDraft,
  SCALE_RANGE,
  SPACING_RANGE,
  THEME_OPTIONS,
  type DraftKey,
  type SettingsDraft,
} from '../features/settings/settingsForm'
import { VoicePicker } from '../features/settings/VoicePicker'
import { usePageTitle } from '../lib/usePageTitle'

type SaveState = 'idle' | 'saving' | 'saved'

export default function SettingsPage() {
  usePageTitle('Settings')
  const { settings, previewSettings, saveSettings } = useAuth()
  const id = useId()
  const [draft, setDraft] = useState<SettingsDraft>(() => draftFrom(settings))
  const [saveState, setSaveState] = useState<SaveState>('idle')
  const [error, setError] = useState<unknown>(null)
  const saved = useRef(settings)

  useEffect(() => {
    saved.current = settings
  }, [settings])

  // Live preview of the draft. Queued after this commit's effects so it wins
  // over the app re-applying the saved settings (e.g. when the samples consent
  // saves while other changes are still unsaved).
  useEffect(() => {
    const merged = mergeDraft(settings, draft)
    let cancelled = false
    queueMicrotask(() => {
      if (!cancelled) previewSettings(merged)
    })
    return () => {
      cancelled = true
    }
  }, [settings, draft, previewSettings])

  // Leaving without saving puts the saved look back.
  useEffect(() => () => previewSettings(saved.current), [previewSettings])

  const dirty = isDirty(settings, draft)
  const font = FONT_OPTIONS.find((option) => option.value === draft.font) ?? FONT_OPTIONS[0]

  const update = <K extends DraftKey>(key: K, value: SettingsDraft[K]) => {
    setDraft((current) => ({ ...current, [key]: value }))
    setSaveState('idle')
  }

  const save = async (event: FormEvent) => {
    event.preventDefault()
    const patch = draftPatch(settings, draft)
    if (Object.keys(patch).length === 0) return
    setSaveState('saving')
    setError(null)
    try {
      await saveSettings(patch)
      setSaveState('saved')
    } catch (err) {
      setError(err)
      setSaveState('idle')
    }
  }

  const undo = () => {
    setDraft(draftFrom(settings))
    setSaveState('idle')
    setError(null)
  }

  const status =
    saveState === 'saving'
      ? 'Saving…'
      : dirty
        ? 'You have unsaved changes. You can see them on the page now.'
        : saveState === 'saved'
          ? 'Saved. Your settings follow you to any device you sign in on.'
          : 'Everything is saved.'

  return (
    <>
      <div className="page-header">
        <h1>Settings</h1>
        <p>Make Legible comfortable to read and listen to. Changes show straight away; press Save to keep them.</p>
      </div>

      <form className="stack settings-form" onSubmit={save}>
        <section className="card stack" aria-labelledby={`${id}-look`}>
          <h2 id={`${id}-look`}>Reading comfort</h2>

          <fieldset className="field">
            <legend>Font</legend>
            <div className="option-list">
              {FONT_OPTIONS.map((option) => (
                <label key={option.value} className="option font-option" style={{ fontFamily: option.family }}>
                  <input
                    type="radio"
                    name="font"
                    value={option.value}
                    checked={draft.font === option.value}
                    onChange={() => update('font', option.value)}
                  />
                  {option.label}
                </label>
              ))}
            </div>
            <span className="hint">{font.hint}</span>
          </fieldset>

          <div className="settings-preview" aria-labelledby={`${id}-preview`} role="group">
            <span id={`${id}-preview`} className="label small">
              Preview
            </span>
            <p>The quick brown fox jumps over the lazy dog.</p>
            <p>Letters that are easy to mix up: b d p q, m n u, 6 9.</p>
          </div>

          <div className="field">
            <label htmlFor={`${id}-scale`}>Text size</label>
            <div className="range-row">
              <span className="range-row__small" aria-hidden="true">
                A
              </span>
              <input
                id={`${id}-scale`}
                type="range"
                min={SCALE_RANGE.min}
                max={SCALE_RANGE.max}
                step={SCALE_RANGE.step}
                value={draft.font_scale}
                onChange={(event) => update('font_scale', Number(event.target.value))}
                aria-valuetext={formatScale(draft.font_scale)}
              />
              <span className="range-row__large" aria-hidden="true">
                A
              </span>
              <output htmlFor={`${id}-scale`} aria-hidden="true">
                {formatScale(draft.font_scale)}
              </output>
            </div>
          </div>

          <div className="field">
            <label htmlFor={`${id}-spacing`}>Line spacing</label>
            <div className="range-row">
              <input
                id={`${id}-spacing`}
                type="range"
                min={SPACING_RANGE.min}
                max={SPACING_RANGE.max}
                step={SPACING_RANGE.step}
                value={draft.line_spacing}
                onChange={(event) => update('line_spacing', Number(event.target.value))}
                aria-valuetext={`${formatSpacing(draft.line_spacing)} spacing`}
              />
              <output htmlFor={`${id}-spacing`} aria-hidden="true">
                {formatSpacing(draft.line_spacing)}
              </output>
            </div>
          </div>

          <fieldset className="field">
            <legend>Colours</legend>
            <div className="option-list">
              {THEME_OPTIONS.map((option) => (
                <label key={option.value} className="option">
                  <input
                    type="radio"
                    name="theme"
                    value={option.value}
                    checked={draft.theme === option.value}
                    onChange={() => update('theme', option.value)}
                  />
                  <span
                    className="theme-swatch"
                    aria-hidden="true"
                    style={{ background: option.swatch.bg, color: option.swatch.text, borderColor: option.swatch.accent }}
                  >
                    Aa
                  </span>
                  {option.label}
                </label>
              ))}
            </div>
          </fieldset>

          <label className="checkbox">
            <input
              type="checkbox"
              checked={draft.reduce_motion}
              onChange={(event) => update('reduce_motion', event.target.checked)}
            />
            <span>
              <strong>Reduce motion</strong>
              <br />
              <span className="muted small">
                Turns off animations, and the talking face holds still (no blinking or lip movement). Your device’s own
                reduce-motion setting is always respected too.
              </span>
            </span>
          </label>
        </section>

        <section className="card stack" aria-labelledby={`${id}-voice`}>
          <h2 id={`${id}-voice`}>Read-aloud voice</h2>
          <VoicePicker value={draft.tts_voice} onChange={(name) => update('tts_voice', name)} />
        </section>

        <ErrorNotice error={error} />

        <div className="settings-savebar" data-dirty={dirty}>
          <p role="status">{status}</p>
          <div className="row">
            <button type="submit" className="btn" disabled={!dirty || saveState === 'saving'} aria-busy={saveState === 'saving'}>
              {saveState === 'saving' ? 'Saving…' : 'Save changes'}
            </button>
            <button type="button" className="btn btn--secondary" onClick={undo} disabled={!dirty || saveState === 'saving'}>
              Undo changes
            </button>
          </div>
        </div>
      </form>

      <div className="stack">
        <HandwritingSamples />
        <DeleteAccount />
      </div>
    </>
  )
}
