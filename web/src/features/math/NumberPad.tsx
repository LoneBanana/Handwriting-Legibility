type Props = {
  onDigit: (digit: string) => void
  onBackspace: () => void
  onSubmit: () => void
  canSubmit: boolean
  disabled?: boolean
}

const KEYS = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

export function NumberPad({ onDigit, onBackspace, onSubmit, canSubmit, disabled = false }: Props) {
  return (
    <div className="numpad" role="group" aria-label="Number pad">
      {KEYS.map((key) => (
        <button key={key} type="button" onClick={() => onDigit(key)} disabled={disabled}>
          {key}
        </button>
      ))}
      <button type="button" onClick={onBackspace} disabled={disabled} aria-label="Delete last digit">
        ⌫
      </button>
      <button type="button" onClick={() => onDigit('0')} disabled={disabled}>
        0
      </button>
      <button type="button" className="numpad__go" onClick={onSubmit} disabled={disabled || !canSubmit}>
        OK
      </button>
    </div>
  )
}
