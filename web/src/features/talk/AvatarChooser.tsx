import type { AvatarProvider } from '../../api/types'

const OPTIONS: { value: AvatarProvider; label: string; hint: string }[] = [
  { value: 'local', label: 'Built-in face', hint: 'A drawn face that shows each movement clearly.' },
  { value: 'tavus', label: 'Realistic video', hint: 'A real-looking person on video.' },
]

/** Lets the learner pick the avatar. Switching starts a new conversation. */
export function AvatarChooser({
  value,
  onChange,
  disabled = false,
}: {
  value: AvatarProvider
  onChange: (provider: AvatarProvider) => void
  disabled?: boolean
}) {
  return (
    <fieldset className="field">
      <legend>Avatar</legend>
      <div className="option-list">
        {OPTIONS.map((option) => (
          <label key={option.value} className="option">
            <input
              type="radio"
              name="talk-avatar"
              value={option.value}
              checked={value === option.value}
              onChange={() => onChange(option.value)}
              disabled={disabled}
            />
            {option.label}
          </label>
        ))}
      </div>
      <span className="hint">
        {OPTIONS.find((o) => o.value === value)?.hint} Switching avatars starts a new conversation.
      </span>
    </fieldset>
  )
}
