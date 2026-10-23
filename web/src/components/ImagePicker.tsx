import { useEffect, useRef, useState, type ChangeEvent, type DragEvent } from 'react'
import { ACCEPTED_IMAGE_TYPES, validateImage } from '../lib/images'
import { Icon } from './Icon'
import '../features/handwriting/handwriting.css'

// Listing concrete types (not image/*) makes iOS convert HEIC photos to JPEG.
const ACCEPT = ACCEPTED_IMAGE_TYPES.join(',')

type Props = {
  onFile: (file: File) => void
  prompt?: string
  /** Also accept images pasted from the clipboard while the page is open. */
  acceptPaste?: boolean
  compact?: boolean
}

export function ImagePicker({ onFile, prompt = 'Drop a photo here, or choose one', acceptPaste = false, compact = false }: Props) {
  const [dragging, setDragging] = useState(false)
  const [problem, setProblem] = useState<string | null>(null)

  const accept = (file: File | null | undefined) => {
    if (!file) return
    const issue = validateImage(file)
    setProblem(issue)
    if (!issue) onFile(file)
  }
  // The paste listener is registered once; it reads the latest `accept` through a ref.
  const acceptRef = useRef(accept)
  useEffect(() => {
    acceptRef.current = accept
  })

  useEffect(() => {
    if (!acceptPaste) return
    const onPaste = (event: ClipboardEvent) => {
      const item = Array.from(event.clipboardData?.items ?? []).find((i) => i.type.startsWith('image/'))
      const file = item?.getAsFile()
      if (file) {
        event.preventDefault()
        acceptRef.current(file)
      }
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [acceptPaste])

  const onInput = (event: ChangeEvent<HTMLInputElement>) => {
    accept(event.target.files?.[0])
    event.target.value = '' // allow choosing the same file again
  }

  const onDrop = (event: DragEvent) => {
    event.preventDefault()
    setDragging(false)
    accept(event.dataTransfer.files?.[0])
  }

  return (
    <div
      className="dropzone"
      data-dragging={dragging}
      style={compact ? { minHeight: 0, padding: 'var(--space-4)' } : undefined}
      onDragOver={(event) => {
        event.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      {!compact && <Icon name="upload" size={36} />}
      <p style={{ margin: 0 }}>{prompt}</p>
      <div className="row" style={{ justifyContent: 'center' }}>
        <label className="btn file-button">
          <Icon name="upload" />
          Choose a photo
          <input type="file" accept={ACCEPT} onChange={onInput} />
        </label>
        <label className="btn btn--secondary file-button">
          <Icon name="camera" />
          Take a photo
          <input type="file" accept={ACCEPT} capture="environment" onChange={onInput} />
        </label>
      </div>
      {acceptPaste && !compact && <p className="muted small" style={{ margin: 0 }}>You can also paste an image.</p>}
      {problem && (
        <p className="small" role="alert" style={{ color: 'var(--danger)', margin: 0 }}>
          {problem}
        </p>
      )}
    </div>
  )
}
